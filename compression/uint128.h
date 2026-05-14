#pragma once

#include <cstdint>
#include <cstdlib>

namespace compression {
	struct uint128 {
		std::uint64_t hi;
		std::uint64_t lo;
	};

	inline bool operator==(const uint128& a, const uint128& b) { return a.hi == b.hi && a.lo == b.lo; }

	inline uint128 shl(std::uint64_t v, unsigned int n)
	{
		if (n >= 64)
			return {v << (n - 64), {}};

		if (n == 0)
			return {{}, v};

		return {v >> (64 - n), v << n};
	}

	inline void add(uint128& dst, const uint128& src)
	{
		const auto old = dst.lo;
		dst.lo += src.lo;
		dst.hi += src.hi;
		if (dst.lo < old)
			++dst.hi;
	}

	inline void neg(uint128& dst)
	{
		constexpr std::uint64_t zero {};
		dst.lo = ~dst.lo;
		dst.hi = ~dst.hi;
		add(dst, uint128 {{}, 1ull});
	}

	inline uint128 mul(std::uint64_t a, std::uint64_t b)
	{
		constexpr auto mask32 = (1ull << 32) - 1;
		const auto a0 = a & mask32;
		const auto b0 = b & mask32;
		const auto a1 = a >> 32;
		const auto b1 = b >> 32;

		uint128 prod = shl(a0 * b0, 0);
		add(prod, shl(a0 * b1, 32));
		add(prod, shl(a1 * b0, 32));
		add(prod, shl(a1 * b1, 64));

		return prod;
	}

	inline bool lt(const uint128& a, const uint128& b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }

	inline bool lte(const uint128& a, const uint128& b) { return a == b || lt(a, b); }

	// hi is quotient, lo is remainder
	inline void div(uint128& n, std::uint64_t d)
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

	inline void bit_or(uint128& dst, const uint128& src)
	{
		dst.lo |= src.lo;
		dst.hi |= src.hi;
	}

	inline void bit_and(uint128& dst, const uint128& src)
	{
		dst.lo &= src.lo;
		dst.hi &= src.hi;
	}

	inline void bit_xor(uint128& dst, const uint128& src)
	{
		dst.lo ^= src.lo;
		dst.hi ^= src.hi;
	}

	inline void bit_not(uint128& dst)
	{
		dst.lo = ~dst.lo;
		dst.hi = ~dst.hi;
	}

	inline auto popcnt(const uint128& src) { return _mm_popcnt_u64(src.hi) + _mm_popcnt_u64(src.lo); }
}
