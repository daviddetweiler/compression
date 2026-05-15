#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <gsl/gsl>

#include "uint128.h"

namespace compression {
	namespace {
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

			void emit(unsigned int bit)
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
				const auto idx = context.extract(slider, pos);
				auto& model = gsl::at(models, idx);
				const auto rwidth = rbound - lbound;
				Expects(rwidth > 1);

				auto tmp = mul(rwidth, model.ones);
				div(tmp, model.total); // Throw away the remainder for now
				auto split = tmp.hi; // Yes, the rounding is bad if the remainder was non-zero
				// Clamping to ensure we always predict nonzero probability for each symbol
				split = split == rwidth ? split - 1 : split;
				split = split == 0 ? split + 1 : split;

				const auto divider = lbound + split;
				// Window's bit-reversed, that's why it's acting weird
				const auto bit = inbound <= divider ? 1 : 0;
				slider <<= 1;
				slider |= bit;

				model.ones += bit;
				++model.total;

				lbound = bit ? lbound : lbound + split;
				rbound = bit ? lbound + split + 1 : rbound;

				while ((~(lbound ^ rbound)) >> 63) {
					lbound <<= 1;
					rbound <<= 1;
					nextbit();
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
					decode(pos++);
					const auto bit = slider & 1;
					trace << lbound << ',' << rbound << ',' << slider << ',' << bit << '\n';
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
			std::ofstream trace {"decode.csv"};
		};

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
				Expects(rwidth > 1);
				auto tmp = mul(rwidth, model.ones);
				div(tmp, model.total); // Throw away the remainder for now
				auto split = tmp.hi; // Yes, the rounding is bad if the remainder was non-zero
				// Clamping to ensure we always predict nonzero probability for each symbol
				split = split == rwidth ? split - 1 : split;
				split = split == 0 ? split + 1 : split;
				model.ones += bit;
				++model.total;

				// what happens when the range collapses hmmmmmmmmm
				// I.e. what if the distribution predicts zero probability for 0?
				lbound = bit ? lbound : lbound + split;
				rbound = bit ? lbound + split + 1 : rbound;

				while ((~(lbound ^ rbound)) >> 63) {
					wtr.emit(lbound >> 63);
					lbound <<= 1;
					rbound <<= 1;
				}
			}

			double encode_all()
			{
				while (!rdr.is_end()) {
					const auto bit = rdr.next();
					encode(bit);
					trace << lbound << ',' << rbound << ',' << slider << ',' << bit << '\n';
					++pos;
				}

				// We must force the choice of some number in the middle of the final range
				// Picking lbound the last bit is always 1 for some reason
				// Picking rbound does things I don't understand.
				// encode(0);
				for (auto i = 0; i < 64; ++i) {
					wtr.emit(lbound >> 63);
					lbound <<= 1;
				}

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
			model_context context {0, 0};
			gsl::span<bit_model> models {};
			std::vector<unsigned char> encoded {};
			bitwriter wtr {};
			bitreader rdr {};
			std::ofstream trace {"encode.csv"};
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

		std::uint64_t evolve_for(gsl::span<const unsigned char> data)
		{
			constexpr auto elites = 32;
			constexpr auto relatives = 32;
			static_assert(elites <= relatives, "you're biasing the distribution");
			std::mt19937 drbg {0xdeadbeef};
			std::bernoulli_distribution coin {0.5};
			std::uniform_int_distribution bit_dist {0, mask_width - 1};
			std::vector<std::pair<uint128, double>> pool(elites * relatives);
			for (auto& gene : pool)
				gene.first = draw(drbg);

			const auto actual_bits = data.size() * 8;
			std::vector<bit_model> models(1ull << maxbits);
			const gsl::span<bit_model> root_span {models};
			for (auto i = 0; i < 512; ++i) {
				for (auto& [mask, score] : pool) {
					const model_context ctx {mask.lo, mask.hi};
					const auto popcnt = ctx.bits();
					if (popcnt <= maxbits) {
						const auto model_span = root_span.subspan(0, 1ull << popcnt);
						score = running_entropy(model_span, data, ctx) / actual_bits;
					}
					else
						score = static_cast<double>(actual_bits) + 1.0;
				}

				std::sort(pool.begin(), pool.end(), [](auto&& a, auto&& b) {
					if (a.second < b.second)
						return true;

					if (a.second == b.second && popcnt(a.first) < popcnt(b.first))
						return true;

					return false;
				});

				const auto& best = pool.front();
				const auto logline = std::format(
					"Generation {} best score {} (ctx: 0x{:x}, pos: 0x{:x})",
					i,
					best.second,
					best.first.lo,
					best.first.hi);

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
		const model_context ctx {0xffff, 0x7};
		// The general pattern from all the evolutionary stuff is that the lower 3 bits of the position, and the closest
		// N bits of context, are the most important.
		encoder enc {blob, ctx, models, false};
		std::cout << "Encoded: " << 100.0 * enc.encode_all() << " %" << std::endl;

		enc.write("tapeout.bin");
		const auto [encoded, n_bits] = enc.out();

		decoder dec {encoded, ctx, models, blob.size()};
		dec.decode_all(n_bits, blob.size() << 3);
		dec.write("tapeout-rt.bin", raw_size);
	}

	//evolve_for(blob);
}
