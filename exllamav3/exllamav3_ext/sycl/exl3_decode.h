#pragma once

// EXL3 decode chain, ported from quant/exl3_dq.cuh + quant/codebook.cuh + quant/bits_k.cuh into
// portable SYCL. The decode formulas are lifted verbatim from the CUDA sources; the only structural
// differences are that these read one 16-bit window per call instead of fragment batches, and the
// tile words are addressed through a small loader (see PackedPtr below).

#include "exl3_compat.h"

// -------------------------------------------------------------------------------------------------
// bits_from_K (quant/bits_k.cuh). K arrives as a float from the checkpoint (trellis.shape[-1]/16):
// integer 1..8, or half-integer 1.5 / 2.5 / 3.5 (mul1 codebook only)
// -------------------------------------------------------------------------------------------------

struct BitsK
{
    int bits;
    bool half;
};

inline BitsK bits_from_K(float K)
{
    const int bits = (int) K;
    const float f = K - (float) bits;
    if (!(bits >= 1 && bits <= 8 && (f == 0.0f || (f == 0.5f && bits <= 3))))
        throw std::runtime_error("Unsupported EXL3 bitrate (integer 1..8, or 1.5 / 2.5 / 3.5)");
    return { bits, f == 0.5f };
}

inline int k2_from_K(float K)
{
    const BitsK b = bits_from_K(K);
    return 2 * b.bits + (b.half ? 1 : 0);
}

// -------------------------------------------------------------------------------------------------
// Tile word loaders. The CUDA kernels stage the uint16 tile into shared memory as uint32 words
// (a verbatim byte copy), so word i = tile16[2i] | tile16[2i+1] << 16 and the bitstream is read
// MSB-first within each word (stream bit s is bit 31 - (s % 32) of word s / 32). Both loaders
// below reproduce that bit order; word indices wrap modulo the tile size (the virtual wrap the
// decoders rely on).
// -------------------------------------------------------------------------------------------------

// Tile staged in local memory as uint32 words (reconstruct / had kernels)
struct PackedPtrU32
{
    const uint32_t* p;
    int words;
    inline uint32_t operator[](int i) const
    {
        i = ((i % words) + words) % words;
        return p[i];
    }
};

// Tile read directly from global uint16 memory (GEMV): word i = tile16[2i] | tile16[2i+1] << 16
struct PackedPtrU16
{
    const uint16_t* p;
    int words;
    inline uint32_t operator[](int i) const
    {
        i = ((i % words) + words) % words;
        return (uint32_t) p[2 * i] | ((uint32_t) p[2 * i + 1] << 16);
    }
};

// Power-of-two-tile variant: the wrap becomes a single AND instead of two runtime integer
// divisions (a GPU division is a ~20-50 instruction software sequence; the GEMV inner loops
// issue 8 of them per k-tile per lane, which dominated the whole kernel). words must be a power
// of two — words = 8*K for integer K in {1,2,4,8} and 4*(2*K+1) = 16 for K = 1.5
template <bool POW2>
struct PackedPtrU16Pow2
{
    const uint16_t* p;
    int words;
    inline uint32_t operator[](int i) const
    {
        i &= words - 1;
        return (uint32_t) p[2 * i] | ((uint32_t) p[2 * i + 1] << 16);
    }
};

// Tile staged in local memory through a local_accessor (fused reconstruct_had), captured by
// value in the kernel; `base` is this subtile's word offset into the accessor
template <typename ACC>
struct PackedPtrLocal
{
    ACC acc;
    int base;
    int words;
    inline uint32_t operator[](int i) const
    {
        i = ((i % words) + words) % words;
        return acc[base + i];
    }
};

// uint32 words per tile: integer K -> K * 8; half-integer K = KA + 0.5 -> 4 * (2*KA + 1)
inline constexpr int tile_words(int K, bool half) { return half ? (4 * (2 * K + 1)) : (K * 8); }

// Stage a uint16 tile (packed_size uint16s) into a uint32 local buffer
inline void stage_tile(uint32_t* dst, const uint16_t* src, int packed_size)
{
    const int words = packed_size / 2;
    for (int i = 0; i < words; ++i)
        dst[i] = (uint32_t) src[2 * i] | ((uint32_t) src[2 * i + 1] << 16);
}

// -------------------------------------------------------------------------------------------------
// Codebooks (quant/codebook.cuh decode_3inst<cb>, constants verbatim). Input is the 16-bit window;
// the result is an fp16 weight.
// -------------------------------------------------------------------------------------------------

template <int cb>
inline sycl::half decode_3inst(uint32_t x)
{
    if constexpr (cb == 0)
    {
        x *= 89226354u;
        x += 64248484u;
        x = exl3_cb_lop3(x);
        sycl::half lo = ushort_as_half((uint16_t) (x & 0xffffu));
        sycl::half hi = ushort_as_half((uint16_t) (x >> 16));
        return hadd(lo, hi);
    }
    if constexpr (cb == 1)
    {
        x *= 0xCBAC1FEDu;
        x = exl3_cb_lop3(x);
        sycl::half lo = ushort_as_half((uint16_t) (x & 0xffffu));
        sycl::half hi = ushort_as_half((uint16_t) (x >> 16));
        return hadd(lo, hi);
    }
    if constexpr (cb == 2)
    {
        x *= 0x83DCD12Du;
        const uint32_t acc = 0x6400u;  // 0x6400 -> 1024.0 ..  0x67FF -> 2047.0
        uint32_t sum = dp4a_ones(x, acc);   // byte-sum of x + 1024, integer
        sycl::half h = ushort_as_half((uint16_t) sum);   // bit copy, NOT a numeric conversion
        const sycl::half k_inv_h = ushort_as_half(0x1eee);  //  0.00677 = 1/147.7
        const sycl::half k_bias_h = ushort_as_half(0xc931); // -10.39 = (-1024.0 - 510.0) * k_inv_h
        return hfma(h, k_inv_h, k_bias_h);
    }
}

// -------------------------------------------------------------------------------------------------
// Generic integer-K decode (exl3_dq.cuh dq<bits, cb>). For weight t in a tile whose bitstream
// starts at ptr: the 16-bit window of weight t ends at (t+1)*bits (+ 256*bits virtual wrap) and
// its value is a funnel shift of the two words containing it. Generic for every integer K 1..8
// (the CUDA kernel has aligned fast paths for K in {1,2,4}; not needed here).
// -------------------------------------------------------------------------------------------------

template <int bits, int cb, typename PTR>
inline sycl::half dq(PTR ptr, int t_offset)
{
    int b0 = t_offset * bits + bits - 16 + 256 * bits;  // start of the 16-bit window
    int b1 = b0 + 16;                                   // end (exclusive)
    int i0 = b0 / 32;
    int i1 = (b1 - 1) / 32;
    int s0 = (i1 + 1) * 32 - b1;
    uint32_t a = ptr[i0];
    uint32_t b = ptr[i1];
    uint32_t w0 = fshift(b, a, s0) & 0xffffu;
    return decode_3inst<cb>(w0);
}

// Four consecutive weights (exl3_dq.cuh dq4): the group spans 3*bits + 16 <= 40 bits
template <int bits, int cb, typename PTR>
inline void dq4(PTR ptr, int t_offset, sycl::half* out /* [4] */)
{
    int b0 = (t_offset + 257) * bits - 16;   // start of first window of the 4
    int b1 = b0 + 3 * bits;                  // start of last window
    int b2 = b1 + 16;                        // end of last window
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[i0];
    uint32_t b = ptr[i2];
    uint32_t w3 = fshift(b, a, s2)             & 0xffffu;
    uint32_t w2 = fshift(b, a, s2 + bits)      & 0xffffu;
    uint32_t w1 = fshift(b, a, s2 + bits * 2)  & 0xffffu;
    uint32_t w0 = fshift(b, a, s2 + bits * 3)  & 0xffffu;
    out[0] = decode_3inst<cb>(w0);
    out[1] = decode_3inst<cb>(w1);
    out[2] = decode_3inst<cb>(w2);
    out[3] = decode_3inst<cb>(w3);
}

// Eight consecutive weights: the windows are contiguous, so a per-weight loop of the generic form
// is correct for every integer bits (the 16-bit window of each weight spans at most two words)
template <int bits, int cb, typename PTR>
inline void dq8(PTR ptr, int t_offset, sycl::half* out /* [8] */)
{
    for (int j = 0; j < 8; ++j)
        out[j] = dq<bits, cb>(ptr, t_offset + j);
}

// -------------------------------------------------------------------------------------------------
// Half-integer bitrate decode (exl3_dq.cuh dq8_half): K = KA + 0.5, mul1 codebook only. Positions
// alternate KA (even t) / KA+1 (odd t) bits; bits2 = 2*KA + 1 bits per two positions; the tile is
// 128*bits2 bits = 4*bits2 uint32 words. The 16-bit window of weight t ends at
// ((t>>1)+1)*bits2 - (t odd ? 0 : KA+1) + 128*bits2.
// -------------------------------------------------------------------------------------------------

template <int KA, int cb, typename PTR>
inline sycl::half dq_half(PTR ptr, int t_offset)
{
    constexpr int bits2 = 2 * KA + 1;
    int end = ((t_offset >> 1) + 1) * bits2 - ((t_offset & 1) ? 0 : KA + 1) + 128 * bits2;
    int start = end - 16;
    int i0 = start / 32;
    int i1 = (end - 1) / 32;
    int s = (i1 + 1) * 32 - end;
    uint32_t a = ptr[i0];
    uint32_t b = ptr[i1];
    uint32_t w0 = fshift(b, a, s) & 0xffffu;
    return decode_3inst<cb>(w0);
}

// Eight consecutive weights starting at an even index (port of dq8_half: two groups of four, one
// funnel shift per group, the other three windows derived by shifting)
template <int KA, int cb, typename PTR>
inline void dq8_half(PTR ptr, int t_offset, sycl::half* out /* [8] */)
{
    constexpr int bits2 = 2 * KA + 1;
    constexpr int gspan = 18 + 3 * KA;                     // bits spanned by four windows
    const int e7 = ((t_offset >> 1) + 4) * bits2 + 128 * bits2;  // end bit of window 7 (+ one tile for the wrap)
    const int e3 = e7 - 2 * bits2;                               // end bit of window 3
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    {
        int hi = (e7 - 1) / 32, lo = (e7 - gspan) / 32, s = (hi + 1) * 32 - e7;
        uint32_t a = ptr[lo], b = ptr[hi];
        w7 = fshift(b, a, s); w6 = w7 >> (KA + 1); w5 = w6 >> KA; w4 = w5 >> (KA + 1);
    }
    {
        int hi = (e3 - 1) / 32, lo = (e3 - gspan) / 32, s = (hi + 1) * 32 - e3;
        uint32_t a = ptr[lo], b = ptr[hi];
        w3 = fshift(b, a, s); w2 = w3 >> (KA + 1); w1 = w2 >> KA; w0 = w1 >> (KA + 1);
    }
    out[0] = decode_3inst<cb>(w0 & 0xffffu);
    out[1] = decode_3inst<cb>(w1 & 0xffffu);
    out[2] = decode_3inst<cb>(w2 & 0xffffu);
    out[3] = decode_3inst<cb>(w3 & 0xffffu);
    out[4] = decode_3inst<cb>(w4 & 0xffffu);
    out[5] = decode_3inst<cb>(w5 & 0xffffu);
    out[6] = decode_3inst<cb>(w6 & 0xffffu);
    out[7] = decode_3inst<cb>(w7 & 0xffffu);
}

// -------------------------------------------------------------------------------------------------
// Weight position mapping of the 16x16 subtile (verified against the CUDA reconstruct shuffle
// network): t = l*8 + j with lane l in 0..31, j in 0..7 maps to subtile position (R, C), and the
// inverse finds t for a given (R, C) (the GEMV uses this)
// -------------------------------------------------------------------------------------------------

inline void weight_t_to_rc(int t, int* R, int* C)
{
    int l = t >> 3;
    int j = t & 7;
    *R = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1);
    *C = 2 * (l >> 3) + 8 * (j >> 2) + ((l >> 2) & 1);
}

inline int weight_rc_to_t(int R, int C)
{
    int c = R >> 3, d = R & 1, p = (R & 6) >> 1;
    int l = ((C & 7) >> 1) * 8 + ((C & 7) & 1) * 4 + p;
    int j = (C >> 3) * 4 + c * 2 + d;
    return l * 8 + j;
}
