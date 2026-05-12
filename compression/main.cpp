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

namespace compression {
	namespace {
		struct uint128 {
			std::uint64_t hi;
			std::uint64_t lo;
		};

		bool operator==(const uint128& a, const uint128& b) { return a.hi == b.hi && a.lo == b.lo; }

		uint128 shl(std::uint64_t v, unsigned int n)
		{
			if (n >= 64)
				return {v << (n - 64), {}};

			if (n == 0)
				return {{}, v};

			return {v >> (64 - n), v << n};
		}

		void add(uint128& dst, const uint128& src)
		{
			const auto old = dst.lo;
			dst.lo += src.lo;
			dst.hi += src.hi;
			if (dst.lo < old)
				++dst.hi;
		}

		void neg(uint128& dst)
		{
			constexpr std::uint64_t zero {};
			dst.lo = ~dst.lo;
			dst.hi = ~dst.hi;
			add(dst, uint128 {{}, 1ull});
		}

		uint128 mul(std::uint64_t a, std::uint64_t b)
		{
			uint128 accum {};
			accum.lo = a;

			uint128 result {};
			for (; b; b >>= 1, add(accum, accum)) {
				if (!(b & 1))
					continue;

				add(result, accum);
			}

			return result;
		}

		bool lt(const uint128& a, const uint128& b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }

		bool lte(const uint128& a, const uint128& b) { return a == b || lt(a, b); }

		// hi is quotient, lo is remainder
		void div(uint128& n, std::uint64_t d)
		{
			std::uint64_t q {};
			auto i = 63;
			for (auto i = 63; i >= 0; --i) {
				auto a = shl(d, i);
				if (lte(a, n)) {
					neg(a);
					add(n, a);
					q |= (1ull << i);
				}
			}

			if (n.hi)
				std::abort();

			n.hi = q;
		}

		auto load_binary(gsl::czstring filename)
		{
			std::ifstream file {filename, std::ifstream::binary | std::ifstream::ate};
			file.exceptions(file.failbit | file.badbit);
			const auto size = file.tellg();
			std::vector<unsigned char> buffer(size);
			void* const data_ptr = buffer.data();
			file.seekg(file.beg);
			file.read(static_cast<char*>(data_ptr), size);
			return buffer;
		}

		struct bit_model {
			std::uint64_t ones;
			std::uint64_t total;
		};

		struct model {
			std::uint64_t total;
			std::vector<bit_model> weights;
		};

		constexpr auto bitpos_mask = 63ull;

		// Reads off the leftmost bit
		struct bitreader {
		private:
			gsl::span<const unsigned char> bytes {};
			std::uint64_t bitpos {};
			std::uint64_t window {};

		public:
			bitreader() = default;

			bitreader(gsl::span<const unsigned char> bytes) : bytes {bytes}, bitpos {}, window {}
			{
				std::memcpy(&window, bytes.data(), sizeof(window));
			}

			std::uint64_t pos() { return bitpos; }

			std::uint64_t next()
			{
				const auto b = window >> 63;
				window <<= 1;
				++bitpos;
				if (!(bitpos & bitpos_mask)) {
					const auto idx = bitpos >> 3;
					if (idx != bytes.size())
						std::memcpy(&window, &gsl::at(bytes, idx), sizeof(window));
				}

				return b;
			}

			bool is_end() { return (bitpos >> 3) >= bytes.size(); }
		};

		double contribution(double pvalue)
		{
			const auto c = -pvalue * log2(pvalue);
			return std::isnan(c) ? 0.0 : c;
		}

		double entropy(const bit_model& model)
		{
			const auto total = static_cast<double>(model.total);
			const auto ones = static_cast<double>(model.ones) / total;
			const auto zeroes = 1.0 - ones;
			return contribution(ones) + contribution(zeroes);
		}

		double total_entropy(const model& model)
		{
			auto total = 0.0;
			for (const auto& submodel : model.weights)
				total += entropy(submodel) * submodel.total;

			return total;
		}

		model get_stats(gsl::span<const unsigned char> bytes, std::uint64_t ctx_mask, std::uint64_t pos_mask)
		{
			bitreader rdr {bytes};
			std::uint64_t window {};
			std::uint64_t total {};
			const auto ctx_bits = _mm_popcnt_u64(ctx_mask);
			const auto pos_bits = _mm_popcnt_u64(pos_mask);
			std::vector<bit_model> dist(1ull << (ctx_bits + pos_bits));
			while (!rdr.is_end()) {
				const auto ctx = _pext_u64(window, ctx_mask);
				const auto pos = _pext_u64(total, pos_mask);
				const auto idx = (pos << ctx_bits) | ctx;
				const auto bit = rdr.next();
				auto& model = gsl::at(dist, idx);
				if (bit)
					++model.ones;

				++model.total;
				window <<= 1;
				window |= bit;
				++total;
			}

			return {total, dist};
		}

		double running_entropy(
			gsl::span<bit_model> dist,
			gsl::span<const unsigned char> bytes,
			std::uint64_t ctx_mask,
			std::uint64_t pos_mask = {})
		{
			const auto ctx_bits = _mm_popcnt_u64(ctx_mask);
			bitreader rdr {bytes};
			std::uint64_t window {};
			std::uint64_t total {};
			for (auto& model : dist) {
				model.total = 2;
				model.ones = 1;
			}

			auto e_total = 0.0;
			while (!rdr.is_end()) {
				const auto ctx = _pext_u64(window, ctx_mask);
				const auto pos = _pext_u64(total, pos_mask);
				const auto idx = (pos << ctx_bits) | ctx;
				const auto bit = rdr.next();
				auto& model = gsl::at(dist, idx);
				e_total += entropy(model);
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
				std::uint64_t ctx_mask,
				std::uint64_t pos_mask,
				gsl::span<bit_model> models,
				std::uint64_t expected) :
				decoder {}
			{
				rdr = bitreader {input};
				this->ctx_mask = ctx_mask;
				this->pos_mask = pos_mask;
				ctx_bits = _mm_popcnt_u64(ctx_mask);
				pos_bits = _mm_popcnt_u64(pos_mask);
				this->models = models;
				decoded.resize(expected);
				for (auto& model : models) {
					model.ones = 1;
					model.total = 2;
				}
			}

			void decode(std::uint64_t pos)
			{
				const auto idx = (_pext_u64(pos, pos_mask) << ctx_bits) | _pext_u64(slider, ctx_mask);
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

			void decode_all(std::uint64_t n_bits)
			{
				for (auto i = 0; i < 64; ++i)
					nextbit();

				std::uint64_t pos {};
				while (n_inbound < n_bits) {
					decode(pos++);
					if (!(pos & 63))
						std::memcpy(&gsl::at(decoded, ((pos >> 6) - 1) << 3), &slider, sizeof(slider));
				}

				/*const auto leftover_bits = pos & 63;
				const auto bitpad = (8 - (leftover_bits & 7)) & 7;
				const auto bytes = (leftover_bits + bitpad) >> 3;
				slider <<= bitpad;
				std::memcpy(&gsl::at(decoded, ((pos >> 6) - 1) << 3), &slider, bytes);*/
			}

			void write(gsl::czstring filename)
			{
				std::ofstream file {filename, std::ofstream::binary};
				file.exceptions(file.badbit | file.failbit);
				file.write(reinterpret_cast<const char*>(decoded.data()), decoded.size());
			}

		private:
			std::uint64_t lbound {};
			std::uint64_t rbound {~lbound};
			std::uint64_t slider {}; // The sliding window
			std::uint64_t inbound {};
			std::uint64_t n_inbound {}; // How many bits of the outbound are pending
			std::uint64_t ctx_mask {};
			std::uint64_t pos_mask {};
			std::uint64_t ctx_bits {};
			std::uint64_t pos_bits {};
			gsl::span<bit_model> models {};
			std::vector<unsigned char> decoded {};
			bitreader rdr {};
		};

		class encoder {
		public:
			encoder() = default;
			encoder(
				gsl::span<const unsigned char> input,
				std::uint64_t ctx_mask,
				std::uint64_t pos_mask,
				gsl::span<bit_model> models,
				bool dry_run) :
				encoder {}
			{
				rdr = bitreader {input};
				if (!dry_run)
					encoded.resize(input.size());

				this->ctx_mask = ctx_mask;
				this->pos_mask = pos_mask;
				ctx_bits = _mm_popcnt_u64(ctx_mask);
				pos_bits = _mm_popcnt_u64(pos_mask);
				this->models = models;
				for (auto& model : models) {
					model.ones = 1;
					model.total = 2;
				}
			}

			// One bit only!
			void encode()
			{
				const auto pos = rdr.pos();
				const auto bit = rdr.next();
				const auto idx = (_pext_u64(pos, pos_mask) << ctx_bits) | _pext_u64(slider, ctx_mask);
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
					outbound <<= 1;
					outbound |= (lbound >> 63);
					++n_outbound; // Throw away the output bits for now
					lbound <<= 1;
					rbound <<= 1;

					if (!encoded.empty() && !(n_outbound & 63))
						std::memcpy(&gsl::at(encoded, ((n_outbound >> 6) - 1) << 3), &outbound, sizeof(outbound));
				}
			}

			double encode_all()
			{
				while (!rdr.is_end())
					encode();

				/*const auto leftover_bits = n_outbound & 63;
				const auto bitpad = (64 - leftover_bits) & 63;
				outbound <<= bitpad;
				std::memcpy(&gsl::at(encoded, ((n_outbound >> 6) - 1) << 3), &outbound, sizeof(outbound));
				n_outbound += bitpad;*/

				return static_cast<double>(n_outbound) / rdr.pos();
			}

			void write(gsl::czstring filename)
			{
				std::ofstream file {filename, std::ofstream::binary};
				file.exceptions(file.badbit | file.failbit);
				const auto bytes = n_outbound / 8;
				file.write(reinterpret_cast<const char*>(encoded.data()), n_outbound % 8 ? bytes + 1 : bytes);
			}

			std::pair<gsl::span<const unsigned char>, std::uint64_t> out() { return {encoded, n_outbound}; }

		private:
			std::uint64_t lbound {};
			std::uint64_t rbound {~lbound};
			std::uint64_t slider {}; // The sliding window
			std::uint64_t outbound {};
			std::uint64_t n_outbound {}; // How many bits of the outbound are pending
			std::uint64_t ctx_mask {};
			std::uint64_t pos_mask {};
			std::uint64_t ctx_bits {};
			std::uint64_t pos_bits {};
			gsl::span<bit_model> models {};
			std::vector<unsigned char> encoded {};
			bitreader rdr {};
		};

		constexpr auto mask_width = 128;

		void bit_or(uint128& dst, const uint128& src)
		{
			dst.lo |= src.lo;
			dst.hi |= src.hi;
		}

		void bit_and(uint128& dst, const uint128& src)
		{
			dst.lo &= src.lo;
			dst.hi &= src.hi;
		}

		void bit_xor(uint128& dst, const uint128& src)
		{
			dst.lo ^= src.lo;
			dst.hi ^= src.hi;
		}

		void bit_not(uint128& dst)
		{
			dst.lo = ~dst.lo;
			dst.hi = ~dst.hi;
		}

		uint128 vary(std::mt19937& drbg, const uint128& mask)
		{
			std::uniform_int_distribution bit_dist {0, mask_width - 1};
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

		constexpr auto maxbits = 16;

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

		auto u128_popcnt(const uint128& src) { return _mm_popcnt_u64(src.hi) + _mm_popcnt_u64(src.lo); }

		std::uint64_t evolve_for(gsl::span<const unsigned char> data)
		{
			constexpr auto elites = 32;
			constexpr auto relatives = 32;
			static_assert(elites <= relatives, "you're biasing the distribution");
			std::mt19937 drbg {0xdeadbeef};
			std::vector<std::pair<uint128, double>> pool(elites * relatives);
			for (auto& gene : pool)
				gene.first = draw(drbg);

			const auto actual_bits = data.size() * 8;
			std::vector<bit_model> models(1ull << maxbits);
			for (auto i = 0; i < 512; ++i) {
				for (auto& [mask, score] : pool) {
					const auto ctx_mask = mask.lo;
					const auto pos_mask = mask.hi;
					const auto popcnt = _mm_popcnt_u64(ctx_mask) + _mm_popcnt_u64(pos_mask);
					if (popcnt <= maxbits) {
						auto enc = encoder {
							data,
							ctx_mask,
							pos_mask,
							gsl::span {models}.subspan(0, gsl::narrow_cast<std::size_t>(1ull << popcnt)),
							true};

						score = enc.encode_all();
					}
					else {
						score = static_cast<double>(actual_bits) + 1.0;
					}
				}

				std::sort(pool.begin(), pool.end(), [](auto&& a, auto&& b) {
					if (a.second < b.second)
						return true;

					if (a.second == b.second && u128_popcnt(a.first) < u128_popcnt(b.first))
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
					const auto mask = gsl::at(pool, j);
					const auto off = j * relatives;
					gsl::at(pool, off) = mask;
				}

				for (auto j = 0; j < elites; ++j) {
					const auto off = j * relatives;
					const auto mask = gsl::at(pool, off);
					for (auto jp = 1; jp < relatives; ++jp) {
						std::bernoulli_distribution coin {0.5};
						auto newmask = vary(drbg, mask.first);
						for (auto retry = 0; retry < mask_width && coin(drbg); ++retry)
							newmask = vary(drbg, newmask);

						gsl::at(pool, off + jp).first = newmask;
					}
				}
			}

			return 0;
		}

		void for_mask(gsl::span<const unsigned char> blob, std::uint64_t ctx_mask, std::uint64_t pos_mask = {})
		{
			const auto stats = compression::get_stats(blob, ctx_mask, pos_mask);
			const auto total_percent = 100.0 * compression::total_entropy(stats) / stats.total;

			std::vector<bit_model> models(1ull << (_mm_popcnt_u64(ctx_mask) + _mm_popcnt_u64(pos_mask)));
			const auto running_percent
				= 100.0 * compression::running_entropy(models, blob, ctx_mask, pos_mask) / stats.total;

			std::cout << std::format(
				"{} %\t{} %\t{} %",
				total_percent,
				running_percent,
				100.0 * running_percent / total_percent)
					  << std::endl;
		}
	}
}

int main()
{
	const auto result = compression::mul(~std::uint64_t {}, 0xdeadbeefcafebabeull);
	std::cout << result.hi << " " << result.lo << std::endl;

	const auto max = ~std::uint64_t {};
	auto maxprod = compression::mul(max, max);
	compression::div(maxprod, max);
	std::cout << maxprod.hi << " " << maxprod.lo << std::endl;

	const auto blob = compression::load_binary("C:\\Users\\david\\source\\silicon\\out\\kernel.bin");
	compression::for_mask(blob, 0xff);
	compression::for_mask(blob, 0x7ff);
	compression::for_mask(blob, 0xaa55);
	compression::for_mask(blob, 0xeeee);
	compression::for_mask(blob, 0x333333);
	compression::for_mask(blob, 0xa0000000020081bf); // Best for current kernel.bin
	compression::for_mask(blob, 0xa0ebff); // Best for current kernel.asm

	// With positional context, the algorithm found: 0x800000ff, 0xc00010000017 with a 40% compression ratio from
	// coding alone. For kernel.bin, ofc.
	compression::for_mask(blob, 0x800000ff, 0xc00010000017);
	compression::for_mask(blob, 0x1000ff, 0x30000060007); // and silicon-debug.exe
	compression::for_mask(blob, 0x2bff, 0x100007); // and compression.exe release build
	compression::for_mask(blob, 0x80e3ff, 0x6); // and kernel.asm
	compression::for_mask(blob, 0x8080ff, 0x208000100000007); // and main.cpp
	compression::for_mask(blob, 0x3e3ff, 0x4); // and assorted.tex
	// and compression.exe again with more agressive population diversity
	compression::for_mask(blob, 0x21a047, 0x6006);
	compression::for_mask(blob, 0x20ff, 0x3f); // kernel.bin post- AC encoder

	std::vector<compression::bit_model> models(1 << 16);
	compression::encoder enc {blob, 0x20ff, 0x3f, models, false};
	std::cout << "Encoded: " << 100.0 * enc.encode_all() << " %" << std::endl;

	enc.write("tapeout.bin");
	const auto [encoded, n_bits] = enc.out();

	compression::decoder dec {encoded, 0x20ff, 0x3f, models, blob.size()};
	dec.decode_all(n_bits);
	dec.write("tapeout-rt.bin");

	compression::evolve_for(blob);
}
