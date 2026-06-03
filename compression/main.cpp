#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include <gsl/gsl>

#include "uint128.h"

namespace compression {
	namespace {
		using clock = std::chrono::system_clock;
		using seconds = std::chrono::duration<double>;

		template <std::size_t a>
		auto align(std::size_t n)
		{
			constexpr auto mask = a - 1;
			const auto tail = n & mask;
			return n + ((a - tail) & mask);
		}

		auto load_binary(gsl::czstring filename)
		{
			std::ifstream file {filename, std::ifstream::binary | std::ifstream::ate};
			file.exceptions(file.failbit | file.badbit);
			const auto size = file.tellg();
			std::vector<unsigned char> buffer(align<sizeof(std::uint64_t)>(size));
			void* const data_ptr = buffer.data();
			file.seekg(file.beg);
			file.read(static_cast<char*>(data_ptr), size);
			return std::make_pair(buffer, size);
		}

		struct bit_model {
			std::uint64_t ones;
			std::uint64_t total;
		};

		class model_context {
		public:
			model_context(std::uint64_t ctx_mask, std::uint64_t pos_mask) noexcept :
				ctx_mask {ctx_mask},
				pos_mask {pos_mask},
				ctx_bits {gsl::narrow<unsigned int>(_mm_popcnt_u64(ctx_mask))}
			{
			}

			std::uint64_t extract(std::uint64_t nearbits, std::uint64_t bitpos) const noexcept
			{
				return (_pext_u64(bitpos, pos_mask) << ctx_bits) | _pext_u64(nearbits, ctx_mask);
			}

			std::uint64_t bits() const noexcept { return ctx_bits + _mm_popcnt_u64(pos_mask); }

			auto get_masks() const { return std::make_pair(ctx_mask, pos_mask); }

		private:
			std::uint64_t ctx_mask;
			std::uint64_t pos_mask;
			unsigned int ctx_bits;
		};

		constexpr auto bitpos_mask = 63ull;

		// Reads off the leftmost bit
		struct bitreader {
		private:
			gsl::span<const unsigned char> bytes {};
			std::uint64_t bitpos {};
			std::uint64_t window {};
			std::size_t n_words {};

		public:
			bitreader() = default;

			bitreader(gsl::span<const unsigned char> bytes) :
				bytes {bytes},
				bitpos {},
				window {},
				n_words {bytes.size() >> 3}
			{
				Expects(bytes.size() % 8 == 0);
				refill(0);
			}

			std::uint64_t pos() const { return bitpos; }

			std::uint64_t expected() const { return bytes.size() << 3; }

			std::uint64_t next()
			{
				Expects(!is_end());
				const auto b = window >> 63;
				window <<= 1;
				++bitpos;
				if (!(bitpos & bitpos_mask)) {
					const auto word_idx = bitpos >> 6;
					refill(word_idx);
				}

				return b;
			}

			bool is_end() { return (bitpos >> 3) >= bytes.size(); }

			void refill(std::size_t word_idx)
			{
				Expects(word_idx <= n_words);
				if (word_idx == n_words) {
					window = 0;
					return;
				}

				const auto target = bytes.subspan(word_idx << 3, sizeof(window));
				std::memcpy(&window, target.data(), sizeof(window));
			}
		};

		// Needs finalization
		class bitwriter {
		public:
			bitwriter() = default;
			bitwriter(gsl::span<unsigned char> buffer) : buffer {buffer} {}

			void emit(std::uint64_t bit)
			{
				queued <<= 1;
				queued |= bit;
				if (++pos & 63)
					return;

				const auto wordpos = (pos >> 6) - 1;
				push(wordpos);
			}

			void flush()
			{
				const auto tail_bits = pos & bitpos_mask;
				if (!tail_bits)
					return; // Already flushed at the end of the last emit

				const auto pad_bits = 64 - tail_bits;
				const auto wordpos = pos >> 6; // No -1 needed since we're still in the middle of a word;
				queued <<= pad_bits;
				push(wordpos);
				pos += pad_bits;

				Ensures(((pos >> 3) & 7) == 0); // Must be quadword-aligned after flush
			}

			auto getpos() const noexcept { return pos; }

		private:
			std::uint64_t queued {};
			std::uint64_t pos {};
			gsl::span<unsigned char> buffer {};

			void push(std::uint64_t word_idx)
			{
				const auto target = buffer.subspan(word_idx << 3, sizeof(queued));
				std::memcpy(target.data(), &queued, sizeof(queued));
			}
		};

		double bits_for_symbol(const bit_model& model, std::uint64_t bit)
		{
			const auto ones = static_cast<double>(model.ones) / model.total;
			const auto pvalue = bit ? ones : 1.0 - ones;
			const auto bits = std::log2(pvalue);
			return -bits;
		}

		double
		running_entropy(gsl::span<bit_model> dist, gsl::span<const unsigned char> bytes, const model_context& ctx)
		{
			bitreader rdr {bytes};
			std::uint64_t window {};
			std::uint64_t total {};
			for (auto& model : dist) {
				model.total = 2;
				model.ones = 1;
			}

			auto e_total = 0.0;
			while (!rdr.is_end()) {
				const auto idx = ctx.extract(window, total);
				const auto bit = rdr.next();
				auto& model = gsl::at(dist, idx);
				e_total += bits_for_symbol(model, bit);
				if (bit)
					++model.ones;

				++model.total;
				window <<= 1;
				window |= bit;
				++total;
			}

			return e_total;
		}

		/*
			A little bit of context here; at all times:
			- [lbound, rbound] represent the _last_ 64 bits of an interval
			- Based off of the probability that the next bit is set, we pick either the upper or lower subrange
				(how do we deal with remainders? They will contribute upon renormalization)...
			- If the top n bits agree, shift them out
			- If the top bits disagree but it's one of those 0b0111111... 0b10000.... situations, we shift the upper two
				bits into the encoding queue (where they can still be rewritten), and keep track of how many bits are
				added to the tail
				- This is the bulkiest part of encoding / decoding logic imo
			- The goal is to always have all 64 bits of window available to us to build subranges out of
			- In general the subrange width will have a remainder, which we will want to accumulate, concurrently shift,
				and occassionally take bits out of on overflow or getting big enough. But the denominator will be
		   constantly changing (cry)
				- Depending on complexity here, it may be worth simply truncating the model precision
			- Every time 64 bits have been added to the encoding queue, we write the entire queue out. The last queue
				will need to be zero-padded. This can be done by the head of the decompressed code.
		*/

		struct shared_state {
			std::uint64_t lbound; // +0
			std::uint64_t rbound; // +8
			std::uint64_t ctx_mask; // +16
			std::uint64_t pos_mask; // +24
			std::uint64_t ctx; // +32
			std::uint64_t pos; // +40
			bit_model* models; // +48
		};

		class decoder {
		public:
			decoder() = default;
			decoder(
				gsl::span<const unsigned char> input,
				const model_context& context,
				gsl::span<bit_model> models,
				std::uint64_t expected) :
				decoder {}
			{
				rdr = bitreader {input};
				this->context = context;
				this->models = models;
				decoded.resize(expected);
				for (auto& model : models) {
					model.ones = 1;
					model.total = 2;
				}
			}

			void decode(std::uint64_t pos)
			{
				auto& model = gsl::at(models, context.extract(slider, pos));
				const auto range_width = rbound - lbound;
				auto split = uint128_adj(range_width, model.ones, model.total);
				// Clamping to ensure we always predict nonzero probability for each symbol
				split = split == range_width ? split - 1 : split;
				split = split == 0 ? split + 1 : split;

				const auto divider = lbound + split;
				const auto bit = inbound < divider ? 1 : 0;
				slider <<= 1;
				slider |= bit;

				model.ones += bit;
				++model.total;

				lbound = bit ? lbound : divider;
				rbound = bit ? divider : rbound;

				while (true) {
					if (!((lbound ^ rbound) >> 63)) {
						lbound <<= 1;
						rbound <<= 1;
						nextbit();
					}
					else if ((lbound >> 62) == 0b01 && (rbound >> 62) == 0b10) {
						lbound <<= 1;
						lbound &= ~(1ull << 63);
						rbound <<= 1;
						rbound |= 1ull << 63;

						const auto hibit = inbound & (1ull << 63);
						nextbit();
						inbound &= ~(1ull << 63);
						inbound |= hibit;
					}
					else {
						break;
					}
				}
			}

			void nextbit()
			{
				inbound <<= 1;
				inbound |= rdr.next();
				++n_inbound;
			}

			void decode_all(std::uint64_t, std::uint64_t expected_bits)
			{
				for (auto i = 0; i < 64; ++i)
					nextbit();

				std::uint64_t pos {};
				const gsl::span root_span {decoded};
				while (pos < expected_bits) {
					decode(pos);
					++pos;
					if (!(pos & 63)) {
						const auto wordpos = (pos >> 6) - 1;
						const auto target = root_span.subspan(wordpos << 3, sizeof(slider));
						std::memcpy(target.data(), &slider, sizeof(slider));
					}
				}
			}

			void write(gsl::czstring filename, std::size_t actual_size)
			{
				std::ofstream file {filename, std::ofstream::binary};
				file.exceptions(file.badbit | file.failbit);
				const auto valid_bytes = gsl::span {decoded}.subspan(0, actual_size);
				file.write(reinterpret_cast<const char*>(valid_bytes.data()), valid_bytes.size());
			}

		private:
			std::uint64_t lbound {};
			std::uint64_t rbound {~lbound};
			std::uint64_t slider {}; // The sliding window
			std::uint64_t inbound {};
			std::uint64_t n_inbound {}; // How many bits of the outbound are pending
			model_context context {0, 0};
			gsl::span<bit_model> models {};
			std::vector<unsigned char> decoded {};
			bitreader rdr {};
		};

		// This is for assembly-language encoding
		struct encoder_state {
			shared_state shared;
			std::uint64_t outbound; // +56
			std::uint64_t n_outbound; // +64 Encoding shift-out MUST stop every time we get to the next 64-bit block
			std::uint64_t n_trailing; // +72
			std::uint64_t* output_words; // +80
		};

		extern "C" bit_model* get_model(shared_state* state, std::uint64_t bit);
		extern "C" std::uint64_t get_subrange(shared_state* state, std::uint64_t bit, bit_model* model);
		extern "C" void update_model(shared_state* state, std::uint64_t bit, bit_model* model);

		// This desparately needs unit-testing
		// And testing how closely the entropy estimation tracks the encoder (same final model states)
		class encoder {
		public:
			encoder() = default;
			encoder(
				gsl::span<const unsigned char> input,
				const model_context& context,
				gsl::span<bit_model> models,
				bool dry_run) :
				encoder {}
			{
				if (!dry_run)
					encoded.resize(input.size() + 8);

				rdr = bitreader {input};
				wtr = bitwriter {encoded};

				this->context = context;
				this->models = models;
				for (auto& model : models) {
					model.ones = 1;
					model.total = 2;
				}
			}

			// One bit only!
			void encode(std::uint64_t bit)
			{
				const auto idx = context.extract(slider, pos);
				slider = (slider << 1) | bit;
				auto& model = gsl::at(models, idx);
				const auto rwidth = rbound - lbound;
				auto split = uint128_adj(rwidth, model.ones, model.total);
				// Clamping to ensure we always predict nonzero probability for each symbol
				// Of note: numbers in the range (split, split + 1) will never be generated and have no meaning
				// Only way to fix this would be to make the intervals half-open, but that would probably mean making
				// 0 an illegal value.
				split = split == rwidth ? split - 1 : split;
				split = split == 0 ? split + 1 : split;
				model.ones += bit;
				++model.total;

				// what happens when the range collapses hmmmmmmmmm
				// I.e. what if the distribution predicts zero probability for 0?
				lbound = bit ? lbound : lbound + split;
				rbound = bit ? lbound + split : rbound;

				while (true) {
					if (!((lbound ^ rbound) >> 63)) {
						const auto ebit = lbound >> 63;
						wtr.emit(ebit);
						lbound <<= 1;
						rbound <<= 1;
						for (auto i = 0ull; i < n_trailers; ++i)
							wtr.emit(~ebit & 0b1);

						n_trailers = 0;
					}
					else if ((lbound >> 62) == 0b01 && (rbound >> 62) == 0b10) {
						// Congratulations, we have the 0b0111... 0b1000... case
						lbound <<= 1;
						lbound &= ~(1ull << 63);
						rbound <<= 1;
						rbound |= 1ull << 63;
						++n_trailers;
					}
					else {
						break;
					}
				}
			}

			double encode_all()
			{
				while (!rdr.is_end()) {
					const auto bit = rdr.next();
					encode(bit);
					++pos;
				}

				// Now that the valid range is actually [lbound, rbound), correct tail behavior demands that we simply
				// flush the entire implied value of lbound This is correct, because all the bits that were actually
				// locked when encoding up to the final bit have already been decided upon (dubious in the case of
				// having implied bits)
				const auto ebit = lbound >> 63;
				wtr.emit(ebit);
				for (auto i = 0ull; i < n_trailers; ++i)
					wtr.emit(~ebit & 0b1);

				n_trailers = 0;

				lbound <<= 1;
				for (auto i = 0ull; i < 63; ++i, lbound <<= 1)
					wtr.emit(lbound >> 63);

				wtr.flush();

				return static_cast<double>(wtr.getpos()) / rdr.pos();
			}

			void write(gsl::czstring filename)
			{
				std::ofstream file {filename, std::ofstream::binary};
				file.exceptions(file.badbit | file.failbit);
				const auto bytes = wtr.getpos() >> 3;
				file.write(reinterpret_cast<const char*>(encoded.data()), bytes);
			}

			std::pair<gsl::span<const unsigned char>, std::uint64_t> out() { return {encoded, wtr.getpos()}; }

		private:
			std::uint64_t lbound {};
			std::uint64_t rbound {~lbound};
			std::uint64_t slider {}; // The sliding window
			std::uint64_t pos {};
			std::uint64_t n_trailers {};
			model_context context {0, 0};
			gsl::span<bit_model> models {};
			std::vector<unsigned char> encoded {};
			bitwriter wtr {};
			bitreader rdr {};
		};

		class asm_encoder {
		public:
			asm_encoder() = default;
			asm_encoder(
				gsl::span<const unsigned char> input,
				const model_context& context,
				gsl::span<bit_model> models,
				bool dry_run) :
				asm_encoder {}
			{
				if (!dry_run)
					encoded.resize(input.size() + 8);

				rdr = bitreader {input};
				wtr = bitwriter {encoded};

				const auto [ctx_mask, pos_mask] = context.get_masks();
				state.ctx_mask = ctx_mask;
				state.pos_mask = pos_mask;
				state.models = models.data();
				state.rbound = ~state.lbound;
				for (auto& model : models) {
					model.ones = 1;
					model.total = 2;
				}
			}

			// One bit only!
			void encode(std::uint64_t bit)
			{
				const auto model = get_model(&state, bit);
				const auto split = get_subrange(&state, bit, model);
				update_model(&state, bit, model);
				// Clamping to ensure we always predict nonzero probability for each symbol
				// Of note: numbers in the range (split, split + 1) will never be generated and have no meaning
				// Only way to fix this would be to make the intervals half-open, but that would probably mean making
				// 0 an illegal value.

				// what happens when the range collapses hmmmmmmmmm
				// I.e. what if the distribution predicts zero probability for 0?
				state.lbound = bit ? state.lbound : state.lbound + split;
				state.rbound = bit ? state.lbound + split : state.rbound;

				while (true) {
					if (!((state.lbound ^ state.rbound) >> 63)) {
						const auto ebit = state.lbound >> 63;
						wtr.emit(ebit);
						state.lbound <<= 1;
						state.rbound <<= 1;
						for (auto i = 0ull; i < n_trailers; ++i)
							wtr.emit(~ebit & 0b1);

						n_trailers = 0;
					}
					else if ((state.lbound >> 62) == 0b01 && (state.rbound >> 62) == 0b10) {
						// Congratulations, we have the 0b0111... 0b1000... case
						state.lbound <<= 1;
						state.lbound &= ~(1ull << 63);
						state.rbound <<= 1;
						state.rbound |= 1ull << 63;
						++n_trailers;
					}
					else {
						break;
					}
				}
			}

			double encode_all()
			{
				while (!rdr.is_end()) {
					const auto bit = rdr.next();
					encode(bit);
					++state.pos;
				}

				// Now that the valid range is actually [lbound, rbound), correct tail behavior demands that we simply
				// flush the entire implied value of lbound This is correct, because all the bits that were actually
				// locked when encoding up to the final bit have already been decided upon (dubious in the case of
				// having implied bits)
				const auto ebit = state.lbound >> 63;
				wtr.emit(ebit);
				for (auto i = 0ull; i < n_trailers; ++i)
					wtr.emit(~ebit & 0b1);

				n_trailers = 0;

				state.lbound <<= 1;
				for (auto i = 0ull; i < 63; ++i, state.lbound <<= 1)
					wtr.emit(state.lbound >> 63);

				wtr.flush();

				return static_cast<double>(wtr.getpos()) / rdr.pos();
			}

			void write(gsl::czstring filename)
			{
				std::ofstream file {filename, std::ofstream::binary};
				file.exceptions(file.badbit | file.failbit);
				const auto bytes = wtr.getpos() >> 3;
				file.write(reinterpret_cast<const char*>(encoded.data()), bytes);
			}

			std::pair<gsl::span<const unsigned char>, std::uint64_t> out() { return {encoded, wtr.getpos()}; }

		private:
			shared_state state {};
			std::vector<unsigned char> encoded {};
			std::uint64_t n_trailers {};
			bitwriter wtr {};
			bitreader rdr {};
		};

		class asm_decoder {
		public:
			asm_decoder() = default;
			asm_decoder(
				gsl::span<const unsigned char> input,
				const model_context& context,
				gsl::span<bit_model> models,
				std::uint64_t expected) :
				asm_decoder {}
			{
				rdr = bitreader {input};
				const auto [ctx_mask, pos_mask] = context.get_masks();
				state.ctx_mask = ctx_mask;
				state.pos_mask = pos_mask;
				state.rbound = ~state.lbound;
				state.models = models.data();
				decoded.resize(expected);
				for (auto& model : models) {
					model.ones = 1;
					model.total = 2;
				}
			}

			void decode()
			{
				const auto model = get_model(&state, 0);
				const auto split = get_subrange(&state, 0, model);
				const auto divider = state.lbound + split;
				const auto bit = inbound < divider ? 1 : 0;
				update_model(&state, bit, model);

				state.lbound = bit ? state.lbound : divider;
				state.rbound = bit ? divider : state.rbound;

				while (true) {
					if (!((state.lbound ^ state.rbound) >> 63)) {
						state.lbound <<= 1;
						state.rbound <<= 1;
						nextbit();
					}
					else if ((state.lbound >> 62) == 0b01 && (state.rbound >> 62) == 0b10) {
						state.lbound <<= 1;
						state.lbound &= ~(1ull << 63);
						state.rbound <<= 1;
						state.rbound |= 1ull << 63;

						const auto hibit = inbound & (1ull << 63);
						nextbit();
						inbound &= ~(1ull << 63);
						inbound |= hibit;
					}
					else {
						break;
					}
				}
			}

			void nextbit()
			{
				inbound <<= 1;
				inbound |= rdr.next();
				++n_inbound;
			}

			void decode_all(std::uint64_t, std::uint64_t expected_bits)
			{
				for (auto i = 0; i < 64; ++i)
					nextbit();

				const gsl::span root_span {decoded};
				while (state.pos < expected_bits) {
					decode();
					++state.pos;
					if (!(state.pos & 63)) {
						const auto wordpos = (state.pos >> 6) - 1;
						const auto target = root_span.subspan(wordpos << 3, sizeof(state.ctx));
						std::memcpy(target.data(), &state.ctx, sizeof(state.ctx));
					}
				}
			}

			void write(gsl::czstring filename, std::size_t actual_size)
			{
				std::ofstream file {filename, std::ofstream::binary};
				file.exceptions(file.badbit | file.failbit);
				const auto valid_bytes = gsl::span {decoded}.subspan(0, actual_size);
				file.write(reinterpret_cast<const char*>(valid_bytes.data()), valid_bytes.size());
			}

		private:
			shared_state state {};
			std::uint64_t inbound {};
			std::uint64_t n_inbound {};
			std::vector<unsigned char> decoded {};
			bitreader rdr {};
		};

		uint128 vary(std::mt19937& drbg, std::uniform_int_distribution<int>& bit_dist, const uint128& mask)
		{
			const auto bit = bit_dist(drbg);
			const auto select = shl(1ull, bit);
			auto extract = mask;
			bit_and(extract, select);

			auto result = mask;
			bit_xor(result, extract);
			bit_not(extract);
			bit_and(extract, select);
			bit_or(result, extract);

			return result; // Flip the bit
		}

		constexpr auto mask_width = 128;
		constexpr auto maxbits = 20;

		uint128 draw(std::mt19937& drbg)
		{
			std::uniform_int_distribution nbit_dist {0, maxbits};
			const auto nbits = nbit_dist(drbg);

			auto lower = 0;
			uint128 value {};
			for (auto i = 0; i < nbits && lower < mask_width; ++i) {
				std::uniform_int_distribution nxbit_dist {lower, mask_width - 1};
				const auto nxbit = nxbit_dist(drbg);
				bit_or(value, shl(1ull, nxbit));
				lower = nxbit + 1;
			}

			return value;
		}

		void run_worker(
			gsl::span<std::pair<uint128, double>> assignment,
			gsl::span<bit_model> root_span,
			gsl::span<const unsigned char> data,
			std::uint64_t actual_bits)
		{
			for (auto& [mask, score] : assignment) {
				const model_context ctx {mask.lo, mask.hi};
				const auto popcnt = ctx.bits();
				if (popcnt <= maxbits) {
					const auto model_span = root_span.subspan(0, 1ull << popcnt);
					score = running_entropy(model_span, data, ctx) / actual_bits;
				}
				else
					score = static_cast<double>(actual_bits) + 1.0;
			}
		}

		std::uint64_t evolve_for(gsl::span<const unsigned char> data)
		{
			constexpr auto elites = 32;
			constexpr auto relatives = 32;
			static_assert(elites <= relatives, "you're biasing the distribution");
			const auto n_cores = std::thread::hardware_concurrency();
			std::mt19937 drbg {0xdeadbeef};
			std::bernoulli_distribution coin {0.5};
			std::uniform_int_distribution bit_dist {0, mask_width - 1};
			std::vector<std::pair<uint128, double>> pool(elites * relatives);
			gsl::span pool_span {pool};
			std::vector<gsl::span<std::pair<uint128, double>>> assignments(n_cores);
			std::vector<std::vector<bit_model>> models(n_cores);
			const auto n_per_core = pool.size() / n_cores;
			for (auto& gene : pool)
				gene.first = draw(drbg);

			auto assignment_idx = 0ull;
			for (auto i = 0u; i < n_cores - 1; ++i) {
				gsl::at(assignments, i) = pool_span.subspan(assignment_idx, n_per_core);
				assignment_idx += n_per_core;
			}

			assignments.back() = pool_span.subspan(assignment_idx);
			const auto actual_bits = data.size() * 8;

			for (auto& model : models)
				model.resize(1ull << maxbits);

			std::vector<std::thread> workers {};
			workers.reserve(n_cores);
			for (auto i = 0; i < 512; ++i) {
				const auto start = clock::now();
				for (auto j = 0ull; j < n_cores; ++j) {
					const gsl::span model {gsl::at(models, j)};
					const auto assignment = gsl::at(assignments, j);
					workers.emplace_back(
						[model, assignment, data, actual_bits] { run_worker(assignment, model, data, actual_bits); });
				}

				for (auto& thr : workers)
					thr.join();

				const auto time = std::chrono::duration_cast<seconds>(clock::now() - start);
				workers.clear();
				std::sort(pool.begin(), pool.end(), [](auto&& a, auto&& b) {
					if (a.second < b.second)
						return true;

					if (a.second == b.second && popcnt(a.first) < popcnt(b.first))
						return true;

					return false;
				});

				const auto& best = pool.front();
				const auto logline = std::format(
					"Generation {} best score {} (ctx: 0x{:x}, pos: 0x{:x}) took {} ({} per trial)",
					i,
					best.second,
					best.first.lo,
					best.first.hi,
					time,
					time / pool.size());

				std::cout << logline << std::endl;

				for (auto j = 0; j < elites; ++j) {
					const auto& mask = gsl::at(pool, j);
					const auto off = j * relatives;
					gsl::at(pool, off) = mask;
				}

				for (auto j = 0; j < elites; ++j) {
					const auto off = j * relatives;
					const auto& mask = gsl::at(pool, off);
					for (auto jp = 1; jp < relatives; ++jp) {
						auto newmask = vary(drbg, bit_dist, mask.first);
						for (auto retry = 0; retry < mask_width && coin(drbg); ++retry)
							newmask = vary(drbg, bit_dist, newmask);

						gsl::at(pool, off + jp).first = newmask;
					}
				}
			}

			return 0;
		}
	}
}

int main(int argc, char** argv)
{
	using namespace compression;

	const gsl::span args {argv, gsl::narrow<std::size_t>(argc)};
	if (argc != 2) {
		std::cerr << "Usage: compression <input file>" << std::endl;
		return 1;
	}

	const auto [blob, raw_size] = load_binary(args[1]);

	{
		std::vector<bit_model> models(1 << 20);

		const model_context ctx {0xbff, 0x60007};
		// const model_context ctx {0x20ff, 0x3f};
		// const model_context ctx {0xffff, 0x7};
		//  The general pattern from all the evolutionary stuff is that the lower 3 bits of the position, and the
		//  closest N bits of context, are the most important. The surprising thing is that this is _still_ suboptimal
		//  for the case of kernel.bin, which benefits from 0x20ff, 0x3f
		asm_encoder enc {blob, ctx, models, false};
		const auto start = clock::now();
		const auto ratio = enc.encode_all();
		const auto time = std::chrono::duration_cast<seconds>(clock::now() - start);
		std::cout << std::format("Encoded: {}% ({})", 100.0 * ratio, time) << std::endl;

		enc.write("tapeout.bin");
		const auto [encoded, n_bits] = enc.out();

		asm_decoder dec {encoded, ctx, models, blob.size()};
		dec.decode_all(n_bits, blob.size() << 3);
		dec.write("tapeout-rt.bin", raw_size);
	}

	evolve_for(blob);
}
