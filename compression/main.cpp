#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
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

		double running_entropy(gsl::span<const unsigned char> bytes, std::uint64_t ctx_mask)
		{
			bitreader rdr {bytes};
			std::uint64_t window {};
			std::uint64_t total {};
			std::vector<bit_model> dist(1ull << _mm_popcnt_u64(ctx_mask));
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

		std::uint64_t evolve_for(gsl::span<const unsigned char> data)
		{
			std::vector<std::pair<std::uint64_t, double>> pool(256);
			std::uint64_t k {};
			for (auto& gene : pool)
				gene.first = k;

			const auto actual_bits = data.size() * 8;
			for (auto i = 0; i < 100; ++i) {
				for (auto& [ctx_mask, score] : pool)
					score = running_entropy(data, ctx_mask) / actual_bits;

				std::sort(pool.begin(), pool.end(), [](auto&& a, auto&& b) { return a.second < b.second; });
				std::cout << "Generation " << i << " best score " << pool.front().second << std::endl;
			}

			return 0;
		}

		void for_mask(gsl::span<const unsigned char> blob, std::uint64_t ctx_mask)
		{
			const auto stats = compression::get_stats(blob, ctx_mask);
			std::cout << "Total entropy content: " << compression::total_entropy(stats) << " / " << stats.total
					  << " total bits" << std::endl;

			std::cout << "Running entropy content: " << compression::running_entropy(blob, ctx_mask) << " / "
					  << stats.total << " total bits" << std::endl;
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

	const auto blob = compression::load_binary("C:\\Users\\david\\source\\silicon\\out\\kernel.bin.lzss");
	compression::for_mask(blob, 0xff);
	compression::for_mask(blob, 0x7ff);
	compression::for_mask(blob, 0xaa55);
	compression::for_mask(blob, 0xeeee);
	compression::for_mask(blob, 0x333333);

	const std::array<std::uint64_t, 2> masks {0xff, 0x700};
	std::cout << "Mixed entropy content: " << compression::mixed_entropy(blob, masks) << " / " << blob.size() * 8
			  << " total bits" << std::endl;

	// compression::evolve_for(blob);
}
