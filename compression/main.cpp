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
			if (n > 64)
				std::abort();

			if (n == 64)
				return {v, {}};

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

		struct bitreader {
			gsl::span<const unsigned char> bytes;
			std::uint64_t bitpos;
			std::uint64_t window;

			bitreader(gsl::span<const unsigned char> bytes) : bytes {bytes}, bitpos {}, window {}
			{
				std::memcpy(&window, bytes.data(), sizeof(window));
			}

			std::uint64_t next()
			{
				const auto b = window & 1;
				window >>= 1;
				++bitpos;
				if (!(bitpos & bitpos_mask)) {
					const auto idx = bitpos >> 3;
					if (idx != bytes.size())
						std::memcpy(&window, &gsl::at(bytes, idx), sizeof(window));
				}

				return b;
			}

			bool is_end() { return bitpos >> 3 == bytes.size(); }
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

		model get_stats(gsl::span<const unsigned char> bytes, std::uint64_t ctx_mask)
		{
			bitreader rdr {bytes};
			std::uint64_t window {};
			std::uint64_t total {};
			std::vector<bit_model> dist(1ull << _mm_popcnt_u64(ctx_mask));
			while (!rdr.is_end()) {
				const auto ctx = _pext_u64(window, ctx_mask);
				const auto bit = rdr.next();
				auto& model = gsl::at(dist, ctx);
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
				const auto bit = rdr.next();
				auto& model = gsl::at(dist, ctx);
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

		// Rather ill-advised, but for reasons which are unclear to me
		double mixed_entropy(gsl::span<const unsigned char> bytes, gsl::span<const std::uint64_t> ctx_masks)
		{
			bitreader rdr {bytes};
			std::uint64_t window {};
			std::uint64_t total {};
			std::vector<std::vector<bit_model>> dist_a(ctx_masks.size());
			auto i = 0;
			for (auto& dist : dist_a) {
				const auto ctx_mask = gsl::at(ctx_masks, i);
				dist.resize(1ull << _mm_popcnt_u64(ctx_mask));
				for (auto& model : dist) {
					model.total = 2;
					model.ones = 1;
				}

				++i;
			}

			auto e_total = 0.0;
			while (!rdr.is_end()) {
				const auto bit = rdr.next();
				auto pvalue = 0.0;
				auto i = 0;
				for (auto& dist : dist_a) {
					const auto ctx_mask = gsl::at(ctx_masks, i);
					const auto ctx = _pext_u64(window, ctx_mask);
					auto& model = gsl::at(dist, ctx);
					pvalue += static_cast<double>(model.ones) / model.total;
					if (bit)
						++model.ones;

					++model.total;
					++i;
				}

				pvalue /= dist_a.size();

				e_total += contribution(pvalue) + contribution(1.0 - pvalue);
				window <<= 1;
				window |= bit;
				++total;
			}

			return e_total;
		}

		class encoder {
		public:
		private:
			std::uint64_t lbound {};
			std::uint64_t rbound {~lbound};
			std::uint64_t slider {}; // The sliding window
			std::uint64_t bitpos {}; // Count of bits in use
			std::vector<unsigned char> encoded {};
		};

		constexpr auto mask_width = 64;

		std::uint64_t vary(std::mt19937& drbg, std::uint64_t mask)
		{
			std::uniform_int_distribution bit_dist {0, mask_width - 1};
			const auto bit = bit_dist(drbg);
			const auto extract = mask & (1ull << bit);
			const auto result = (mask ^ extract) | (~extract & (1ull << bit));
			return result; // Flip the bit
		}

		constexpr auto maxbits = 16;

		std::uint64_t draw(std::mt19937& drbg)
		{
			std::uniform_int_distribution nbit_dist {0, maxbits};
			const auto nbits = nbit_dist(drbg);

			auto lower = 0;
			std::uint64_t value {};
			for (auto i = 0; i < nbits && lower < mask_width; ++i) {
				std::uniform_int_distribution nxbit_dist {lower, mask_width - 1};
				const auto nxbit = nxbit_dist(drbg);
				value |= (1ull << nxbit);
				lower = nxbit + 1;
			}

			return value;
		}

		std::uint64_t evolve_for(gsl::span<const unsigned char> data)
		{
			constexpr auto relatives = 32;
			std::mt19937 drbg {0xcafebabe};
			std::vector<std::pair<std::uint64_t, double>> pool(256 * relatives);
			for (auto& gene : pool)
				gene.first = draw(drbg);

			const auto actual_bits = data.size() * 8;
			std::vector<bit_model> models(1ull << maxbits);
			for (auto i = 0; i < 100; ++i) {
				for (auto& [ctx_mask, score] : pool) {
					const auto popcnt = _mm_popcnt_u64(ctx_mask);
					if (popcnt <= maxbits) {
						const auto e = running_entropy(
							gsl::span {models}.subspan(0, gsl::narrow_cast<std::size_t>(1ull << popcnt)),
							data,
							ctx_mask);

						score = e / actual_bits;
					}
					else {
						score = static_cast<double>(actual_bits) + 1.0;
					}
				}

				std::sort(pool.begin(), pool.end(), [](auto&& a, auto&& b) { return a.second < b.second; });
				std::cout
					<< std::format("Generation {} best score {} (0x{:x})", i, pool.front().second, pool.front().first)
					<< std::endl;

				for (auto j = 0; j < relatives; ++j) {
					const auto mask = gsl::at(pool, j);
					const auto off = j * relatives;
					gsl::at(pool, off) = mask;
					for (auto jp = 1; jp < relatives; ++jp) {
						std::uniform_int_distribution coin {0, 1};
						auto newmask = vary(drbg, mask.first);
						if (coin(drbg))
							newmask = vary(drbg, newmask);

						gsl::at(pool, off + jp).first = newmask;
					}
				}
			}

			return 0;
		}

		void for_mask(gsl::span<const unsigned char> blob, std::uint64_t ctx_mask)
		{
			const auto stats = compression::get_stats(blob, ctx_mask);
			std::cout << "Total entropy content: " << 100.0 * compression::total_entropy(stats) / stats.total << " %"
					  << std::endl;

			std::vector<bit_model> models(1ull << _mm_popcnt_u64(ctx_mask));
			std::cout << "Running entropy content: "
					  << 100.0 * compression::running_entropy(models, blob, ctx_mask) / stats.total << " %"
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

	const std::array<std::uint64_t, 2> masks {0xff, 0x700};
	std::cout << "Mixed entropy content: " << compression::mixed_entropy(blob, masks) << " / " << blob.size() * 8
			  << " total bits" << std::endl;

	compression::evolve_for(blob);
}
