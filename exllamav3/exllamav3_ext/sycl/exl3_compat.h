#pragma once

// Shared helpers for the SYCL backend (port of the CUDA compat.cuh pieces the kernels need).
// Hand-written SYCL 2020, no dpct-isms: everything is plain inline C++.

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

// -------------------------------------------------------------------------------------------------
// fp16 helpers. CUDA __hadd/__hmul/__hfma compute in fp16 with one rounding at the end; sycl::half
// arithmetic promotes to float, computes, and rounds back once. Since every fp16 value is exact in
// fp32, the promoted result is the correctly-rounded fp16 of the exact value, which is
// bit-identical to the fp16-in/fp16-out intrinsic for add/mul/fma. The one exception to keep
// integer is the mul1 byte-sum (dp4a_ones below) and the uint16 -> half reinterpret (bit copy).
// -------------------------------------------------------------------------------------------------

inline float h2f(sycl::half h) { return static_cast<float>(h); }

inline sycl::half f2h(float f) { return sycl::half(f); }

// Bit copies (CUDA __ushort_as_half / half_as_ushort)
inline sycl::half ushort_as_half(uint16_t u) { return sycl::bit_cast<sycl::half>(u); }
inline uint16_t half_as_ushort(sycl::half h) { return sycl::bit_cast<uint16_t>(h); }

inline sycl::half hadd(sycl::half a, sycl::half b) { return sycl::half(float(a) + float(b)); }
inline sycl::half hsub(sycl::half a, sycl::half b) { return sycl::half(float(a) - float(b)); }
inline sycl::half hmul(sycl::half a, sycl::half b) { return sycl::half(float(a) * float(b)); }
inline sycl::half hfma(sycl::half a, sycl::half b, sycl::half c)
{
    return sycl::half(float(a) * float(b) + float(c));
}

// bf16 (CUDA __nv_bfloat16): same argument as above, bit copies via uint16
inline sycl::ext::oneapi::bfloat16 ushort_as_bf16(uint16_t u)
{ return sycl::ext::oneapi::bfloat16(sycl::bit_cast<float, uint32_t>(uint32_t(u) << 16)); }
inline uint16_t bf16_as_ushort(sycl::ext::oneapi::bfloat16 b)
{ return uint16_t(sycl::bit_cast<uint32_t, float>(float(b)) >> 16); }

inline float bf2f(sycl::ext::oneapi::bfloat16 b) { return float(b); }
inline sycl::ext::oneapi::bfloat16 f2bf(float f) { return sycl::ext::oneapi::bfloat16(f); }

// -------------------------------------------------------------------------------------------------
// Bitstream extraction. fshift replicates __funnelshift_r / exl3_dq.cuh's fshift: the 64-bit
// value (a:b) shifted right by `shift` (shift <= 63). Note the tile bitstream is read MSB-first
// within each uint32 word (stream bit s is bit 31 - (s % 32) of word s / 32); the decode formulas
// in exl3_decode.h are lifted verbatim from exl3_dq.cuh and account for this.
// -------------------------------------------------------------------------------------------------

inline uint32_t fshift(uint32_t b, uint32_t a, int shift)
{
    uint64_t merged = ((uint64_t) a << 32) | (uint64_t) b;
    return (uint32_t) (merged >> shift);
}

// dp4a(x, 0x01010101, acc): sum of the four bytes of x plus acc, all uint32. The mul1 codebook's
// byte-sum must stay integer; this is the portable form of __dp4a.
inline uint32_t dp4a_ones(uint32_t x, uint32_t acc)
{
    return (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + ((x >> 24) & 0xffu) + acc;
}

// EXL3_CB_LOP3(x) = (x & 0x8fff8fff) ^ 0x3b603b60, the portable form of the lop3.b32
inline uint32_t exl3_cb_lop3(uint32_t x) { return (x & 0x8fff8fffu) ^ 0x3b603b60u; }

// -------------------------------------------------------------------------------------------------
// Queue / error handling
// -------------------------------------------------------------------------------------------------

#define SYCL_CHECK(expr) \
    do { \
        try { (expr); } \
        catch (const sycl::exception& e) { \
            throw std::runtime_error(std::string("SYCL error: ") + e.what() + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

// The extension is built against torch XPU: kernels run on the current torch XPU stream so they
// stay ordered with torch ops. Falls back to a module-owned default queue when torch is absent
// (standalone test builds).
sycl::queue& exl3_sycl_queue();
