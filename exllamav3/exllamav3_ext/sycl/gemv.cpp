// Decode GEMV, port of quant/exl3_gemv.cu + exl3_gemv_kernel.cuh into SYCL (no tensor cores).
//
// SYCL has no grid-wide sync, so the CUDA kernel's fused flow becomes three launches:
//   1. input Hadamard (pre = suh, 1/sqrt(128))  ->  A_had
//   2. GEMV: one work item per (m, n) output, scalar dot over k with decode-on-the-fly
//   3. output Hadamard (post = svh, 1/sqrt(128)) on C in place
//
// The GEMV decodes each weight from its packed subtile with the inverse position mapping
// (weight_rc_to_t), which is correct for every m; the decode redundancy grows with m, which is
// acceptable in the m <= 144 decode regime this path serves.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <ATen/Tensor.h>
#include <cstdlib>

#include "exl3_compat.h"
#include "exl3_decode.h"
#include "sycl_kernels.h"

namespace {

template <int K, int cb, bool HALF>
class GemvKernelName;

using GemvFn = void (*)(sycl::queue&, const sycl::half*, const uint16_t*, void*, int, int, int, int, bool);

static const GemvFn* gemv_table();   // forward decl, defined below

template <int K, int cb, bool HALF>
void launch_gemv(sycl::queue& q, const sycl::half* A, const uint16_t* packed, void* C,
                 int size_m, int size_k, int size_n, int stride, bool c_fp32)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);
    constexpr int words = tile_words(K, HALF);
    // ntiles is the n-tile STRIDE between k-tiles in the trellis [k/16, n_full/16, words] layout:
    // for a sliced source (stride > 0, the source's full output width) it must be the source's
    // full tile count, not the slice width, or every k-tile after the first reads the wrong columns
    const int ntiles = (stride > 0 ? stride : size_n) / 16;
    const int row_stride = stride > 0 ? stride : size_n;
    // No extra scale here: the separate input/output Hadamard launches each fold 1/sqrt(128),
    // which together with the raw trellis decode yields C = x @ W (the CUDA kernel fuses the two
    // Hadamards with the same normalization instead)

    q.parallel_for<GemvKernelName<K, cb, HALF>>(sycl::range<1>((size_t) size_m * size_n),
        [=](sycl::item<1> item)
    {
        const size_t idx = item.get_linear_id();
        const int mi = (int) (idx / size_n);
        const int ni = (int) (idx % size_n);

        const int kb_n = ni >> 4;                   // subtile index along n
        const int n_in_tile = ni & 15;              // column within the subtile

        float acc = 0.0f;
        const sycl::half* xrow = A + (size_t) mi * size_k;

        for (int k = 0; k < size_k; ++k)
        {
            const uint16_t* tile = packed + (size_t) ((k >> 4) * ntiles + kb_n) * packed_size;
            PackedPtrU16 ptr{ tile, words };
            const int t = weight_rc_to_t(k & 15, n_in_tile);
            sycl::half w;
            if constexpr (HALF)
                w = dq_half<K, cb>(ptr, t);
            else
                w = dq<K, cb>(ptr, t);
            acc = fmaf((float) xrow[k], (float) w, acc);
        }

        const size_t out_idx = (size_t) mi * row_stride + ni;
        if (c_fp32) ((float*) C)[out_idx] = acc;
        else        ((sycl::half*) C)[out_idx] = sycl::half(acc);
    });
}

// Tile-once decode GEMV for the m == 1 decode path. The previous split-K kernel ran one work-group
// PER OUTPUT COLUMN with lanes striding k-tiles: every subtile's tile stream was then re-read by
// the 16 column work-groups, 16x L2-amplified traffic which microbenchmarking showed to be THE
// bound (dropping the decode chain changed nothing; restructuring the work-group layout gave 3.7x
// cold on the lm_head shape). Here ONE work-group of LANES lanes per n-subtile (16 columns): all
// lanes iterate the k-tiles together, each lane decodes 256/LANES consecutive trellis weights per
// tile (every weight decoded exactly once per tile, each tile read by one work-group), and keeps
// its 256/LANES slot accumulators in REGISTERS (no per-weight local-memory RMW; measured ~14%
// over a local-tile RMW) which are written to the local 16x16 tile once at the end; the columns
// are then reduced over the tile rows. LANES is picked by the host from the subtile count: more
// lanes help when there are few subtiles (the hidden layers), fewer when there are many
// (the lm_head). Numerics: partial sums reduce in fp32 (order differs from the serial kernel's,
// fine for fp16-weight dots).
template <int K, int cb, bool HALF, int LANES>
class GemvSplitKKernelName;

template <int K, int cb, bool HALF, int LANES>
void launch_gemv_splitk(sycl::queue& q, const sycl::half* A, const uint16_t* packed, void* C,
                        int size_m, int size_k, int size_n, int stride, bool c_fp32)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);
    constexpr int words = tile_words(K, HALF);
    // ntiles is the n-tile STRIDE between k-tiles in the trellis [k/16, n_full/16, words] layout:
    // for a sliced source (stride > 0) it must be the source's full tile count. The WORK extent,
    // though, is the slice's own subtile count (size_n/16) — the launch extent and the trellis
    // k-stride are different quantities (launching stride/16 groups would compute past the slice
    // and overwrite the sibling sources' output columns in the fused row)
    const int ntiles = (stride > 0 ? stride : size_n) / 16;
    const int ngroups = size_n / 16;
    const int ktiles = size_k / 16;
    constexpr int T_PER_LANE = 256 / LANES;

    // m == 1 (the caller gates on it): one group of LANES per n-subtile
    q.submit([&](sycl::handler& h)
    {
        sycl::local_accessor<float, 2> lout(sycl::range<2>(16, 16), h);   // [col][row] accumulators
        h.parallel_for<GemvSplitKKernelName<K, cb, HALF, LANES>>(
            sycl::nd_range<1>(sycl::range<1>((size_t) ngroups * LANES), sycl::range<1>(LANES)),
            [=](sycl::nd_item<1> it)
        {
            const int kb_n = (int) it.get_group_linear_id();
            const int lane = (int) it.get_local_id(0);

            // hoisted per-lane constants (t0 is loop-invariant across k-tiles)
            int tmap[T_PER_LANE], Rrow[T_PER_LANE], Ccol[T_PER_LANE];
            #pragma unroll
            for (int u = 0; u < T_PER_LANE; ++u)
            {
                const int t = lane * T_PER_LANE + u;
                tmap[u] = t;
                // inverse of weight_rc_to_t: l = t>>3, j = t&7 (j < 8, so the split is exact)
                const int l = t >> 3, j = t & 7;
                const int p = l & 3, c1 = (l >> 2) & 1, c8 = l >> 3;
                const int d = j & 1, cc = (j >> 1) & 1, h8 = j >> 2;
                Rrow[u] = cc * 8 + p * 2 + d;          // k-row within the ktile
                Ccol[u] = h8 * 8 + c8 * 2 + c1;        // column within the subtile
            }

            // register accumulators: no per-weight local-memory RMW (measured ~14% on the
            // hidden-layer shape); every (col, row) slot is written exactly once at the end
            float acc[T_PER_LANE];
            #pragma unroll
            for (int u = 0; u < T_PER_LANE; ++u) acc[u] = 0.0f;

            for (int kt = 0; kt < ktiles; ++kt)
            {
                const uint16_t* tile = packed + (size_t) (kt * ntiles + kb_n) * packed_size;
                PackedPtrU16 ptr{ tile, words };
                const int k0 = kt * 16;
                #pragma unroll
                for (int u = 0; u < T_PER_LANE; ++u)
                {
                    sycl::half w;
                    if constexpr (HALF)
                        w = dq_half<K, cb>(ptr, tmap[u]);
                    else
                        w = dq<K, cb>(ptr, tmap[u]);
                    acc[u] = fmaf((float) A[k0 + Rrow[u]], (float) w, acc[u]);
                }
            }
            #pragma unroll
            for (int u = 0; u < T_PER_LANE; ++u) lout[Ccol[u]][Rrow[u]] = acc[u];
            sycl::group_barrier(it.get_group());
            // each output column is the sum over the tile's 16 rows (the k dim)
            for (int col = lane; col < 16; col += LANES)
            {
                float s = 0.0f;
                for (int row = 0; row < 16; ++row) s += lout[col][row];
                const size_t out_idx = (size_t) kb_n * 16 + col;   // m == 1: row 0
                if (c_fp32) ((float*) C)[out_idx] = s;
                else        ((sycl::half*) C)[out_idx] = sycl::half(s);
            }
        });
    });
}

// fp32 partial sums for the KS-split tile-once kernel (KS slices of size_n each). The split only
// engages for parallelism-starved shapes (ngroups < 2560, i.e. size_n < 40960), so a fixed scratch
// covers every split shape
static float* splitk_scratch(sycl::queue& q)
{
    constexpr size_t CAP = 32 * 262144;  // KS_MAX(32) x max size_n (the 27B lm_head, 248320) floats
    static float* buf = nullptr;
    if (!buf) buf = sycl::malloc_device<float>(CAP, q);
    return buf;
}

// KS-split tile-once GEMV for parallelism-starved shapes: the one-group-per-subtile layout leaves
// the GPU under-occupied when a layer has few n-subtiles (e.g. the 27B's down_proj/out_proj:
// 320 subtiles x 64 lanes = 20K threads, measured ~22 GB/s cold). Splitting the k-tiles across KS
// work-groups per subtile multiplies the threads in flight (KS=8, LANES=64, measured on the cold
// production shapes, bench27b.cpp: down k=17408 n=5120 2.2x, out_proj k=6144 n=5120 2x,
// mlp gate/up k=5120 n=17408 1.4x, GDN bundle k=5120 n=20480 1.25x over the unsplit kernel).
// Each (subtile, k-chunk) group decodes its chunk's weights exactly once into fp32 partials in a
// scratch buffer (disjoint writes — no atomics, no memset); a tiny reduce kernel then sums the KS
// slices into C. The in-order queue orders partial -> reduce and consecutive calls, so the shared
// scratch is safe. m == 1 only (the dispatch gates on it)
template <int K, int cb, bool HALF, int LANES, int KS>
class GemvSplitKsKernelName;
template <int K, int cb, bool HALF, int LANES, int KS>
class GemvSplitKsReduceName;

template <int K, int cb, bool HALF, int LANES, int KS>
void launch_gemv_splitk_ks(sycl::queue& q, const sycl::half* A, const uint16_t* packed, void* C,
                           int size_m, int size_k, int size_n, int stride, bool c_fp32)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);
    constexpr int words = tile_words(K, HALF);
    // ntiles: the n-tile STRIDE between k-tiles in the trellis (the source's full tile count for a
    // sliced source); ngroups: the slice's own subtile count (the work extent)
    const int ntiles = (stride > 0 ? stride : size_n) / 16;
    const int ngroups = size_n / 16;
    const int ktiles = size_k / 16;
    constexpr int T_PER_LANE = 256 / LANES;
    const int chunk = (ktiles + KS - 1) / KS;
    float* scratch = splitk_scratch(q);

    q.submit([&](sycl::handler& h)
    {
        sycl::local_accessor<float, 2> lout(sycl::range<2>(16, 16), h);
        h.parallel_for<GemvSplitKsKernelName<K, cb, HALF, LANES, KS>>(
            sycl::nd_range<1>(sycl::range<1>((size_t) ngroups * KS * LANES), sycl::range<1>(LANES)),
            [=](sycl::nd_item<1> it)
        {
            const int gid = (int) it.get_group_linear_id();
            const int kb_n = gid / KS;              // subtile along n
            const int kc = gid % KS;                // k-chunk within the subtile
            const int lane = (int) it.get_local_id(0);

            int tmap[T_PER_LANE], Rrow[T_PER_LANE], Ccol[T_PER_LANE];
            #pragma unroll
            for (int u = 0; u < T_PER_LANE; ++u)
            {
                const int t = lane * T_PER_LANE + u;
                tmap[u] = t;
                // inverse of weight_rc_to_t: l = t>>3, j = t&7 (j < 8, so the split is exact)
                const int l = t >> 3, j = t & 7;
                const int p = l & 3, c1 = (l >> 2) & 1, c8 = l >> 3;
                const int d = j & 1, cc = (j >> 1) & 1, h8 = j >> 2;
                Rrow[u] = cc * 8 + p * 2 + d;          // k-row within the ktile
                Ccol[u] = h8 * 8 + c8 * 2 + c1;        // column within the subtile
            }

            float acc[T_PER_LANE];
            #pragma unroll
            for (int u = 0; u < T_PER_LANE; ++u) acc[u] = 0.0f;

            const int kt0 = kc * chunk;
            const int kt1 = sycl::min(kt0 + chunk, ktiles);
            for (int kt = kt0; kt < kt1; ++kt)
            {
                const uint16_t* tile = packed + (size_t) (kt * ntiles + kb_n) * packed_size;
                PackedPtrU16 ptr{ tile, words };
                const int k0 = kt * 16;
                #pragma unroll
                for (int u = 0; u < T_PER_LANE; ++u)
                {
                    sycl::half w;
                    if constexpr (HALF)
                        w = dq_half<K, cb>(ptr, tmap[u]);
                    else
                        w = dq<K, cb>(ptr, tmap[u]);
                    acc[u] = fmaf((float) A[k0 + Rrow[u]], (float) w, acc[u]);
                }
            }
            #pragma unroll
            for (int u = 0; u < T_PER_LANE; ++u) lout[Ccol[u]][Rrow[u]] = acc[u];
            sycl::group_barrier(it.get_group());
            // each output column is the sum over the tile's 16 rows (the k dim);
            // disjoint fp32 partial slices — slice kc, element (kb_n*16 + col) = output index i
            for (int col = lane; col < 16; col += LANES)
            {
                float s = 0.0f;
                for (int row = 0; row < 16; ++row) s += lout[col][row];
                scratch[(size_t) kc * size_n + kb_n * 16 + col] = s;
            }
        });
    });

    // reduce: sum the KS fp32 partial slices into C (m == 1: row 0)
    q.submit([&](sycl::handler& h)
    {
        h.parallel_for<GemvSplitKsReduceName<K, cb, HALF, LANES, KS>>(
            sycl::range<1>((size_t) size_n), [=](sycl::item<1> item)
        {
            const int i = (int) item.get_linear_id();
            float s = 0.0f;
            #pragma unroll
            for (int c = 0; c < KS; ++c) s += scratch[(size_t) c * size_n + i];
            if (c_fp32) ((float*) C)[i] = s;
            else        ((sycl::half*) C)[i] = sycl::half(s);
        });
    });
}

// -------------------------------------------------------------------------------------------------
// MT2: two adjacent n-tiles per work-group. The pair's tiles are packed_size apart in the trellis,
// so each lane's two dq4 word-pairs fall in one 128-byte region — 2x the DRAM burst of the
// one-subtile split kernel. With the dq4 decode (4 consecutive weights from 2 word loads instead
// of 4 x 2) the benched throughput on the 27B's cold production shapes is 64-66 GB/s vs 49-56 for
// the KS=8 split (bench27b.cpp: down 1.31x, gate/up 1.16x, out_proj 1.26x, GDN bundle 1.18x).
// Integer K <= 5 only (dq4's 4-window group spans 3*bits+16 bits and must fit one word pair); the
// group = (n-tile PAIR, k-chunk), partials in the shared splitk scratch, then the same KS-reduce
// kernel as launch_gemv_splitk_ks. m == 1 only (the dispatch gates on it)
// -------------------------------------------------------------------------------------------------

template <int K, int cb, int LANES, int KS>
class GemvMt2KernelName;

template <int K, int cb, int LANES, int KS>
void launch_gemv_mt2(sycl::queue& q, const sycl::half* A, const uint16_t* packed, void* C,
                     int size_m, int size_k, int size_n, int stride, bool c_fp32)
{
    constexpr int packed_size = 16 * K;
    constexpr int words = tile_words(K, false);
    const int ntiles = (stride > 0 ? stride : size_n) / 16;
    const int ngroups = size_n / 32;         // tile PAIRS along n
    const int ktiles = size_k / 16;
    constexpr int T_PER_LANE = 256 / LANES;
    const int chunk = (ktiles + KS - 1) / KS;
    float* scratch = splitk_scratch(q);

    q.submit([&](sycl::handler& h)
    {
        sycl::local_accessor<float, 2> lout(sycl::range<2>(32, 16), h);
        h.parallel_for<GemvMt2KernelName<K, cb, LANES, KS>>(
            sycl::nd_range<1>(sycl::range<1>((size_t) ngroups * KS * LANES), sycl::range<1>(LANES)),
            [=](sycl::nd_item<1> it)
        {
            const int gid = (int) it.get_group_linear_id();
            const int kb_n = gid / KS;              // tile pair along n
            const int kc = gid % KS;                // k-chunk
            const int lane = (int) it.get_local_id(0);

            // Vectorized-A mapping (T_PER_LANE == 4): the lane's 4 t-slots (t = lane*4 + u) map to
            // k-rows {p*2, p*2+1, 8+p*2, 8+p*2+1} with per-lane constants (l = lane>>1) — two
            // ADJACENT pairs, so the 8 scalar 2B A loads per k-tile (one per weight, shared by both
            // adjacent tiles) become 2 vector 4B loads. The columns are ch*8 + cbase with
            // ch = (lane&1) | (u>>2): odd lanes map all 4 weights to ONE column, even lanes to two
            int Rrow[T_PER_LANE], Ccol[T_PER_LANE];
            int r0 = 0, cbase = 0;
            if constexpr (T_PER_LANE == 4)
            {
                const int l = lane >> 1;
                const int p = l & 3, c1 = (l >> 2) & 1, c8 = l >> 3;
                r0 = p * 2;                        // first adjacent k-row pair
                cbase = c8 * 2 + c1;
            }
            else
            {
                #pragma unroll
                for (int u = 0; u < T_PER_LANE; ++u)
                {
                    const int t = lane * T_PER_LANE + u;
                    const int l = t >> 3, j = t & 7;
                    const int p = l & 3, c1 = (l >> 2) & 1, c8 = l >> 3;
                    const int d = j & 1, cc = (j >> 1) & 1, h8 = j >> 2;
                    Rrow[u] = cc * 8 + p * 2 + d;      // k-row within the ktile
                    Ccol[u] = h8 * 8 + c8 * 2 + c1;    // column within the subtile
                }
            }

            float acc[2][T_PER_LANE];
            #pragma unroll
            for (int s = 0; s < 2; ++s)
                #pragma unroll
                for (int u = 0; u < T_PER_LANE; ++u) acc[s][u] = 0.0f;

            const int kt0 = kc * chunk;
            const int kt1 = sycl::min(kt0 + chunk, ktiles);
            for (int kt = kt0; kt < kt1; ++kt)
            {
                const uint16_t* tile = packed + (size_t) (kt * ntiles + kb_n * 2) * packed_size;
                PackedPtrU16 ptr0{ tile, words };
                PackedPtrU16 ptr1{ tile + packed_size, words };
                const int k0 = kt * 16;
                sycl::half w0[4], w1[4];
                if constexpr (T_PER_LANE == 4)
                {
                    dq4<K, cb>(ptr0, lane * 4, w0);
                    dq4<K, cb>(ptr1, lane * 4, w1);
                }
                else
                {
                    #pragma unroll
                    for (int u = 0; u < T_PER_LANE; ++u)
                    {
                        w0[u] = dq<K, cb>(ptr0, lane * T_PER_LANE + u);
                        w1[u] = dq<K, cb>(ptr1, lane * T_PER_LANE + u);
                    }
                }
                if constexpr (T_PER_LANE == 4)
                {
                    // 2 vector loads serve both tiles (the pair shares the same k-rows)
                    const uint32_t v0 = *(const uint32_t*) (A + k0 + r0);
                    const uint32_t v1 = *(const uint32_t*) (A + k0 + 8 + r0);
                    const float a[4] = { ushort_as_half(v0 & 0xffff), ushort_as_half(v0 >> 16),
                                         ushort_as_half(v1 & 0xffff), ushort_as_half(v1 >> 16) };
                    #pragma unroll
                    for (int u = 0; u < 4; ++u)
                    {
                        acc[0][u] = fmaf(a[u], (float) w0[u], acc[0][u]);
                        acc[1][u] = fmaf(a[u], (float) w1[u], acc[1][u]);
                    }
                }
                else
                {
                    #pragma unroll
                    for (int u = 0; u < T_PER_LANE; ++u)
                    {
                        acc[0][u] = fmaf((float) A[k0 + Rrow[u]], (float) w0[u], acc[0][u]);
                        acc[1][u] = fmaf((float) A[k0 + Rrow[u]], (float) w1[u], acc[1][u]);
                    }
                }
            }
            #pragma unroll
            for (int u = 0; u < T_PER_LANE; ++u)
            {
                if constexpr (T_PER_LANE == 4)
                {
                    const int ch = (lane & 1) | (u >> 2);
                    const int R = (u >> 1) * 8 + r0 + (u & 1);
                    lout[ch * 8 + cbase][R] = acc[0][u];
                    lout[16 + ch * 8 + cbase][R] = acc[1][u];
                }
                else
                {
                    lout[Ccol[u]][Rrow[u]] = acc[0][u];
                    lout[16 + Ccol[u]][Rrow[u]] = acc[1][u];
                }
            }
            sycl::group_barrier(it.get_group());
            // each output column sums the tile's 16 k-rows; the pair covers outputs
            // [kb_n*32, kb_n*32+32) — disjoint fp32 partial slices, slice kc
            for (int col = lane; col < 32; col += LANES)
            {
                float s = 0.0f;
                for (int row = 0; row < 16; ++row) s += lout[col][row];
                scratch[(size_t) kc * size_n + kb_n * 32 + col] = s;
            }
        });
    });

    // reduce: sum the KS fp32 partial slices into C (m == 1: row 0) — same kernel as the
    // one-subtile split (the partial layout is identical)
    q.submit([&](sycl::handler& h)
    {
        h.parallel_for<GemvSplitKsReduceName<K, cb, false, LANES, KS>>(
            sycl::range<1>((size_t) size_n), [=](sycl::item<1> item)
        {
            const int i = (int) item.get_linear_id();
            float s = 0.0f;
            #pragma unroll
            for (int c = 0; c < KS; ++c) s += scratch[(size_t) c * size_n + i];
            if (c_fp32) ((float*) C)[i] = s;
            else        ((sycl::half*) C)[i] = sycl::half(s);
        });
    });
}

// Fused split-K reduce + output Hadamard (replaces the separate reduce kernel and the caller's
// post-Hadamard: one launch instead of two, and C is never round-tripped through memory twice).
// One work-group per 128-column block: sum the KS fp32 partial slices, H128 butterfly in SLM,
// scale by svh, write C. The butterfly is the same natural-order Sylvester transform (and sign
// convention) as hadamard.cpp's had_r_128, with r_scale = 1/sqrt(128).
static bool g_esimd_defer_reduce = false;

template <int KS, typename OUT>
class GemvReduceHadName;

template <int KS, typename OUT>
static void launch_gemv_reduce_had(sycl::queue& q, const float* scratch, OUT* C,
                                   const sycl::half* svh, int size_n)
{
    const int chunks = size_n / 128;
    q.submit([&](sycl::handler& h)
    {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(128), h);
        h.parallel_for<GemvReduceHadName<KS, OUT>>(
            sycl::nd_range<1>(sycl::range<1>((size_t) chunks * 128), sycl::range<1>(128)),
            [=](sycl::nd_item<1> it)
        {
            const int chunk = (int) it.get_group_linear_id();
            const int p = (int) it.get_local_id(0);
            const int col = chunk * 128 + p;
            float v = 0.0f;
            #pragma unroll
            for (int c = 0; c < KS; ++c) v += scratch[(size_t) c * size_n + col];
            // Match the unfused reduce's dtype: the separate reduce writes (OUT) before the
            // Hadamard, so for a half output the butterfly input is the fp16-rounded sum
            if constexpr (std::is_same_v<OUT, sycl::half>)
                v = (float) (sycl::half) v;
            buf[p] = v;
            sycl::group_barrier(it.get_group());
            for (int st = 0; st < 7; ++st)
            {
                float pv = buf[p ^ (1 << st)];
                sycl::group_barrier(it.get_group());
                buf[p] = (((p >> st) & 1) ? (pv - buf[p]) : (buf[p] + pv));
                sycl::group_barrier(it.get_group());
            }
            C[col] = (OUT) (buf[p] * 0.08838834764831845f * (float) svh[col]);
        });
    });
}

// Proper ESIMD tile-once GEMV (SYCL_ESIMD_KERNEL), mul1 (cb=2) only: each work item = 8
// consecutive virtual lanes (simd<8>), the mul1 decode via the native dp4a (1 instruction vs the
// ~11-op dp4a_ones byte-sum), the window extraction via ONE 32-bit funnel decomposition
// ((a << (31-s2)) << 1 | (b >> s2) == low 32 of (a:b) >> s2, safe for s2 in [0,31]) + constant
// shifts for the 4 windows (bits <= 5 keeps them in the low 32), the cross-lane reduction in
// registers via scalar element access (the SLM is broken in nd_range ESIMD kernels and strided
// selects miscompile — probed empirically, test_esimd22-26), disjoint scratch writes + the split-K
// reduce kernel (the production epilogue pattern). Benched on the 27B's cold production shapes
// (bench27b.cpp variant 41-43): KS=16, 1.07-1.20x over the vec kernel, exact correctness on all
// four shapes. Half-integer rates and cb 0/1 keep the vec kernel (the lop3 decode needs a uint16
// -> half bit-reinterpret on simd, which miscompiles here).
template <int K, int KS>
class GemvEsimdKernelName;
template <int K, int KS>
class GemvEsimdReduceName;

template <int K, int KS>
void launch_gemv_esimd(sycl::queue& q, const sycl::half* A, const uint16_t* packed, void* C,
                       int size_m, int size_k, int size_n, int stride, bool c_fp32)
{
    using namespace sycl::ext::intel::esimd;
    constexpr int packed_size = 16 * K;
    constexpr int words = tile_words(K, false);
    const int ntiles = (stride > 0 ? stride : size_n) / 16;
    const int ngroups = size_n / 32;         // tile PAIRS along n
    const int ktiles = size_k / 16;
    const int chunk = (ktiles + KS - 1) / KS;
    float* scratch = splitk_scratch(q);
    const float k_inv = float(ushort_as_half(0x1eee));
    const float k_bias = float(ushort_as_half(0xc931));

    q.submit([&](sycl::handler& h)
    {
        h.parallel_for<GemvEsimdKernelName<K, KS>>(sycl::nd_range<1>(
            sycl::range<1>((size_t) ngroups * KS * 8), sycl::range<1>(8)),
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL
        {
            const int gid = (int) it.get_group_linear_id();
            const int kb_n = gid / KS;              // tile pair along n
            const int kc = gid % KS;                // k-chunk
            const int j = (int) it.get_local_id(0); // work item covers lanes 8j..8j+7

            // simd<8> over the 8 virtual lanes e = 0..7, lane = 8*j + e; the lane's 4 t-slots
            // (t = lane*4 + u) map to k-rows {r0, r0+1, 8+r0, 8+r0+1} and column ch*8 + cbase
            // with ch = lane&1, cbase = j (constant across the work item)
            simd<uint32_t, 8> ev(0u);
            for (int i = 0; i < 8; ++i) ev[i] = (uint32_t) i;
            const simd<uint32_t, 8> lane = ev + 8u * (uint32_t) j;
            const simd<uint32_t, 8> l = lane >> 1;
            const simd<uint32_t, 8> p = l & 3u;
            const simd<uint32_t, 8> r0 = p << 1;               // first adjacent k-row pair
            constexpr uint32_t bits = K;
            const simd<uint32_t, 8> t_off = lane << 2;         // lane * 4
            const simd<uint32_t, 8> b0 = (t_off + 257u) * bits - 16u;
            const simd<uint32_t, 8> b2 = b0 + 3u * bits + 16u;
            const simd<uint32_t, 8> i0 = b0 >> 5;
            const simd<uint32_t, 8> i2 = (b2 - 1u) >> 5;
            const simd<uint32_t, 8> s2 = (i2 + 1u) * 32u - b2;
            const simd<uint32_t, 8> sh_a = 31u - s2;           // (a << (31-s2)) << 1 == a << (32-s2)
            const simd<uint32_t, 8> i0w = i0 % words;          // wrap (b0 > 0 always)
            const simd<uint32_t, 8> i2w = i2 % words;

            const simd<uint32_t, 8> c0101(0x01010101u);
            const simd<uint32_t, 8> c6400(0x6400u);

            simd<float, 8> acc[2][4];
            for (int s = 0; s < 2; ++s)
                for (int u = 0; u < 4; ++u) acc[s][u] = simd<float, 8>(0.0f);

            const int kt0 = kc * chunk;
            const int kt1 = kt0 + chunk < ktiles ? kt0 + chunk : ktiles;
            for (int kt = kt0; kt < kt1; ++kt)
            {
                const uint16_t* tile0 = packed + (size_t) (kt * ntiles + kb_n * 2) * packed_size;
                const uint32_t* w0p = (const uint32_t*) tile0;
                const uint32_t* w1p = (const uint32_t*) (tile0 + packed_size);
                const int k0 = kt * 16;
                // word gathers: the PackedPtr word i = p[2i] | p[2i+1]<<16, uint32 at byte 4i
                const simd<uint32_t, 8> a0 = gather<uint32_t, 8>(w0p, i0w * 4u);
                const simd<uint32_t, 8> b0v = gather<uint32_t, 8>(w0p, i2w * 4u);
                const simd<uint32_t, 8> a1 = gather<uint32_t, 8>(w1p, i0w * 4u);
                const simd<uint32_t, 8> b1v = gather<uint32_t, 8>(w1p, i2w * 4u);
                // funnel decomposition + the 4 windows per tile (bits <= 5: all in the low 32)
                const simd<uint32_t, 8> win0 = ((a0 << sh_a) << 1u) | (b0v >> s2);
                const simd<uint32_t, 8> win1 = ((a1 << sh_a) << 1u) | (b1v >> s2);
                // A: the lane's 4 t-slots map to k-rows {r0, r0+1, 8+r0, 8+r0+1} (both tiles)
                simd<float, 8> af[4];
                for (int u = 0; u < 4; ++u)
                {
                    const uint32_t krow = (uint32_t) k0 + (u & 1) + 8u * (u >> 1);
                    af[u] = convert<float, sycl::half, 8>(
                        gather<sycl::half, 8>(A, 2u * (krow + r0)));
                }
                for (int u = 0; u < 4; ++u)
                {
                    // tile 0: the u-th window sits at shift bits*(3-u) within win
                    simd<uint32_t, 8> x = ((win0 >> (bits * (3u - (uint32_t) u))) & 0xffffu)
                                          * 0x83DCD12Du;
                    simd<uint32_t, 8> sum = dp4a<uint32_t>(c6400, x, c0101);
                    simd<float, 8> hv = convert<float, uint32_t, 8>(sum) - 24576.0f;
                    simd<float, 8> wf = convert<float, sycl::half, 8>(
                        convert<sycl::half, float, 8>(hv * k_inv + k_bias));
                    acc[0][u] += af[u] * wf;
                    // tile 1
                    simd<uint32_t, 8> x1 = ((win1 >> (bits * (3u - (uint32_t) u))) & 0xffffu)
                                           * 0x83DCD12Du;
                    simd<uint32_t, 8> sum1 = dp4a<uint32_t>(c6400, x1, c0101);
                    simd<float, 8> hv1 = convert<float, uint32_t, 8>(sum1) - 24576.0f;
                    simd<float, 8> wf1 = convert<float, sycl::half, 8>(
                        convert<sycl::half, float, 8>(hv1 * k_inv + k_bias));
                    acc[1][u] += af[u] * wf1;
                }
            }
            // cross-lane reduction in registers: the work item's 8 lanes cover column j (even
            // lanes) and 8+j (odd) for tile 0, +16 for tile 1; the column partial sums the 16 rows
            float colp[4];
            for (int s = 0; s < 2; ++s)
                for (int par = 0; par < 2; ++par)
                {
                    float t = 0.0f;
                    for (int u = 0; u < 4; ++u)
                        for (int e = par; e < 8; e += 2) t += (float) acc[s][u][e];
                    colp[2 * s + par] = t;
                }
            // disjoint scratch writes: the pair's columns {j, 8+j, 16+j, 24+j}
            for (int c = 0; c < 4; ++c)
                scatter(scratch, simd<uint32_t, 1>(
                    4u * (uint32_t) (kc * size_n + kb_n * 32 + j + 8 * c)),
                    simd<float, 1>(colp[c]));
        });
    });

    // reduce: sum the KS fp32 partial slices into C (the same kernel as the other splits);
    // skipped when the caller fuses it with the output Hadamard (sycl_gemv_mul1_had below)
    if (g_esimd_defer_reduce) return;
    q.submit([&](sycl::handler& h)
    {
        h.parallel_for<GemvEsimdReduceName<K, KS>>(sycl::range<1>((size_t) size_n),
            [=](sycl::item<1> item)
        {
            const int i = (int) item.get_linear_id();
            float s = 0.0f;
            #pragma unroll
            for (int c = 0; c < KS; ++c) s += scratch[(size_t) c * size_n + i];
            if (c_fp32) ((float*) C)[i] = s;
            else        ((sycl::half*) C)[i] = sycl::half(s);
        });
    });
}

// mul1 (cb=2) GEMV with the output Hadamard fused into the split-K reduce: runs the ESIMD tile
// kernel (deferred reduce) then the fused reduce+H128 kernel. Falls back to the normal path
// (returns false) whenever the shape is outside the ESIMD dispatch. Only K in [1,5] is wired.
bool sycl_gemv_mul1_had(sycl::queue& q, const sycl::half* A, const uint16_t* packed, void* C,
                        bool c_fp32, const sycl::half* svh, int size_k, int size_n, int K)
{
    if (K < 1 || K > 5 || size_k < 4096 || size_n < 4096 || size_n >= 40960 || size_n % 128 != 0 || size_k % 16 != 0)
        return false;
    const char* e = std::getenv("EXL3_FUSED_HAD");
    if (e && atoi(e) == 0) return false;

    g_esimd_defer_reduce = true;
    switch (K)
    {
        case 1: launch_gemv_esimd<1, 16>(q, A, packed, C, 1, size_k, size_n, 0, c_fp32); break;
        case 2: launch_gemv_esimd<2, 16>(q, A, packed, C, 1, size_k, size_n, 0, c_fp32); break;
        case 3: launch_gemv_esimd<3, 16>(q, A, packed, C, 1, size_k, size_n, 0, c_fp32); break;
        case 4: launch_gemv_esimd<4, 16>(q, A, packed, C, 1, size_k, size_n, 0, c_fp32); break;
        case 5: launch_gemv_esimd<5, 16>(q, A, packed, C, 1, size_k, size_n, 0, c_fp32); break;
    }
    g_esimd_defer_reduce = false;
    if (c_fp32)
        launch_gemv_reduce_had<16, float>(q, splitk_scratch(q), (float*) C, svh, size_n);
    else
        launch_gemv_reduce_had<16, sycl::half>(q, splitk_scratch(q), (sycl::half*) C, svh, size_n);
    return true;
}

// Kernel-instance table (cbi = K - 1; +8 for mcg, +16 for mul1, 24 + K - 1 for the half-integer
// rates), used by the host GEMV entry points
template <int K>
inline void expand_cb_gemv(GemvFn (&table)[27])
{
    table[K - 1]      = &launch_gemv<K, 0, false>;
    table[K - 1 + 8]  = &launch_gemv<K, 1, false>;
    table[K - 1 + 16] = &launch_gemv<K, 2, false>;
}

template <int K, int LANES>
inline void expand_cb_gemv_splitk(GemvFn (&table)[27])
{
    table[K - 1]      = &launch_gemv_splitk<K, 0, false, LANES>;
    table[K - 1 + 8]  = &launch_gemv_splitk<K, 1, false, LANES>;
    table[K - 1 + 16] = &launch_gemv_splitk<K, 2, false, LANES>;
}

static const GemvFn* gemv_table()
{
    static GemvFn table[27] = {};
    static bool init = false;
    if (!init)
    {
        expand_cb_gemv<1>(table); expand_cb_gemv<2>(table); expand_cb_gemv<3>(table);
        expand_cb_gemv<4>(table); expand_cb_gemv<5>(table); expand_cb_gemv<6>(table);
        expand_cb_gemv<7>(table); expand_cb_gemv<8>(table);
        table[24] = &launch_gemv<1, 2, true>;
        table[25] = &launch_gemv<2, 2, true>;
        table[26] = &launch_gemv<3, 2, true>;
        init = true;
    }
    return table;
}

template <int LANES>
static const GemvFn* gemv_splitk_table()
{
    static GemvFn table[27] = {};
    static bool init = false;
    if (!init)
    {
        expand_cb_gemv_splitk<1, LANES>(table); expand_cb_gemv_splitk<2, LANES>(table);
        expand_cb_gemv_splitk<3, LANES>(table); expand_cb_gemv_splitk<4, LANES>(table);
        expand_cb_gemv_splitk<5, LANES>(table); expand_cb_gemv_splitk<6, LANES>(table);
        expand_cb_gemv_splitk<7, LANES>(table); expand_cb_gemv_splitk<8, LANES>(table);
        table[24] = &launch_gemv_splitk<1, 2, true, LANES>;
        table[25] = &launch_gemv_splitk<2, 2, true, LANES>;
        table[26] = &launch_gemv_splitk<3, 2, true, LANES>;
        init = true;
    }
    return table;
}

template <int K, int LANES, int KS>
inline void expand_cb_gemv_splitks(GemvFn (&table)[27])
{
    table[K - 1]      = &launch_gemv_splitk_ks<K, 0, false, LANES, KS>;
    table[K - 1 + 8]  = &launch_gemv_splitk_ks<K, 1, false, LANES, KS>;
    table[K - 1 + 16] = &launch_gemv_splitk_ks<K, 2, false, LANES, KS>;
}

template <int LANES, int KS>
static const GemvFn* gemv_splitks_table()
{
    static GemvFn table[27] = {};
    static bool init = false;
    if (!init)
    {
        expand_cb_gemv_splitks<1, LANES, KS>(table); expand_cb_gemv_splitks<2, LANES, KS>(table);
        expand_cb_gemv_splitks<3, LANES, KS>(table); expand_cb_gemv_splitks<4, LANES, KS>(table);
        expand_cb_gemv_splitks<5, LANES, KS>(table); expand_cb_gemv_splitks<6, LANES, KS>(table);
        expand_cb_gemv_splitks<7, LANES, KS>(table); expand_cb_gemv_splitks<8, LANES, KS>(table);
        table[24] = &launch_gemv_splitk_ks<1, 2, true, LANES, KS>;
        table[25] = &launch_gemv_splitk_ks<2, 2, true, LANES, KS>;
        table[26] = &launch_gemv_splitk_ks<3, 2, true, LANES, KS>;
        init = true;
    }
    return table;
}

template <int K, int LANES, int KS>
inline void expand_cb_gemv_mt2(GemvFn (&table)[27])
{
    table[K - 1]      = &launch_gemv_mt2<K, 0, LANES, KS>;
    table[K - 1 + 8]  = &launch_gemv_mt2<K, 1, LANES, KS>;
    table[K - 1 + 16] = &launch_gemv_mt2<K, 2, LANES, KS>;
}

template <int LANES, int KS>
static const GemvFn* gemv_mt2_table()
{
    static GemvFn table[27] = {};
    static bool init = false;
    if (!init)
    {
        expand_cb_gemv_mt2<1, LANES, KS>(table); expand_cb_gemv_mt2<2, LANES, KS>(table);
        expand_cb_gemv_mt2<3, LANES, KS>(table); expand_cb_gemv_mt2<4, LANES, KS>(table);
        expand_cb_gemv_mt2<5, LANES, KS>(table);
        init = true;
    }
    return table;
}

// ESIMD tile-once table: starts from the MT2 table (fallback rows), overrides the mul1 rows
// (K 1..5) with the ESIMD kernel at KS=16
template <int K, int KS>
inline void expand_esimd_gemv(GemvFn (&table)[27])
{
    table[K - 1 + 16] = &launch_gemv_esimd<K, KS>;
}

template <int KS>
static const GemvFn* gemv_esimd_table()
{
    static GemvFn table[27] = {};
    static bool init = false;
    if (!init)
    {
        __builtin_memcpy(table, gemv_mt2_table<64, 32>(), sizeof(table));
        expand_esimd_gemv<1, KS>(table); expand_esimd_gemv<2, KS>(table);
        expand_esimd_gemv<3, KS>(table); expand_esimd_gemv<4, KS>(table);
        expand_esimd_gemv<5, KS>(table);
        init = true;
    }
    return table;
}

// m == 3 (the MTP verify's row count, draft_num_tokens 2 + 1): the mt2 layout with M
// accumulator rows. The trellis decodes ONCE per (ktile, lane) and per row does 2 vector 4B
// A-loads (the 4 t-slots map to k-rows {r0, r0+1, 8+r0, 8+r0+1}, both tiles sharing them), so
// the decode — 84% of the m=1 GEMV — amortizes over M rows. The generic kernel re-decodes the
// tile per output row (M x traffic amplification, unoptimized decode), which made the verify
// pass ~10x slower per GEMV and MTP drafting ~1.8 tok/s despite a 46-70% acceptance rate.
// KS k-chunks write disjoint fp32 partial slices; a reduce kernel sums them into C. Integer
// K <= 5 (dq4), cb <= 2 (the table covers cb 0..2), LANES 64 (T_PER_LANE == 4: the
// vectorized-A mapping), size_n % 32 == 0 (the tile-pair extent).
template <int M, int K, int cb, int LANES, int KS>
class GemvMtpKernelName;
template <int M, int K, int cb, int LANES, int KS>
class GemvMtpReduceName;

template <int M, int K, int cb, int LANES, int KS>
void launch_gemv_mtp(sycl::queue& q, const sycl::half* A, const uint16_t* packed, void* C,
                     int size_m, int size_k, int size_n, int stride, bool c_fp32)
{
    constexpr int packed_size = 16 * K;
    constexpr int words = tile_words(K, false);
    const int ntiles = (stride > 0 ? stride : size_n) / 16;
    const int ngroups = size_n / 32;         // tile PAIRS along n
    const int ktiles = size_k / 16;
    const int chunk = (ktiles + KS - 1) / KS;
    float* scratch = splitk_scratch(q);

    q.submit([&](sycl::handler& h)
    {
        // tiny per-lane staging (768 B/group): the 6 KB lout SLM cost an occupancy cliff
        // (M=1 2KB 0.34ms, M=2 4KB 0.37ms, M=3 6KB 0.61ms at n=17408 k=5120 — the m=2->m=3
        // step is a cliff, not linear work)
        sycl::local_accessor<float, 1> lsum(sycl::range<1>((size_t) M * 2 * LANES), h);
        h.parallel_for<GemvMtpKernelName<M, K, cb, LANES, KS>>(
            sycl::nd_range<1>(sycl::range<1>((size_t) ngroups * KS * LANES), sycl::range<1>(LANES)),
            [=](sycl::nd_item<1> it)
        {
            const int gid = (int) it.get_group_linear_id();
            const int kb_n = gid / KS;              // tile pair along n
            const int kc = gid % KS;                // k-chunk
            const int lane = (int) it.get_local_id(0);

            const int l = lane >> 1;
            const int p = l & 3, c1 = (l >> 2) & 1, c8 = l >> 3;
            const int r0 = p * 2;                   // first adjacent k-row pair
            const int cbase = c8 * 2 + c1;

            float acc[M][2][4];
            #pragma unroll
            for (int r = 0; r < M; ++r)
                #pragma unroll
                for (int s = 0; s < 2; ++s)
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) acc[r][s][u] = 0.0f;

            const int kt0 = kc * chunk;
            const int kt1 = sycl::min(kt0 + chunk, ktiles);
            constexpr bool wp2 = (words & (words - 1)) == 0;
            for (int kt = kt0; kt < kt1; ++kt)
            {
                const uint16_t* tile = packed + (size_t) (kt * ntiles + kb_n * 2) * packed_size;
                using PP = std::conditional_t<wp2, PackedPtrU16Pow2<true>, PackedPtrU16>;
                PP ptr0{ tile, words };
                PP ptr1{ tile + packed_size, words };
                const int k0 = kt * 16;
                sycl::half w0[4], w1[4];
                dq4<K, cb>(ptr0, lane * 4, w0);
                dq4<K, cb>(ptr1, lane * 4, w1);
                #pragma unroll
                for (int r = 0; r < M; ++r)
                {
                    // 2 vector loads serve both tiles (the pair shares the same k-rows)
                    const sycl::half* xr = A + (size_t) r * size_k;
                    const uint32_t v00 = *(const uint32_t*) (xr + k0 + r0);
                    const uint32_t v01 = *(const uint32_t*) (xr + k0 + 8 + r0);
                    const float a[4] = { ushort_as_half(v00 & 0xffff), ushort_as_half(v00 >> 16),
                                         ushort_as_half(v01 & 0xffff), ushort_as_half(v01 >> 16) };
                    #pragma unroll
                    for (int u = 0; u < 4; ++u)
                    {
                        acc[r][0][u] = fmaf(a[u], (float) w0[u], acc[r][0][u]);
                        acc[r][1][u] = fmaf(a[u], (float) w1[u], acc[r][1][u]);
                    }
                }
            }
            // per-lane 4-row sums staged tiny: each lane's acc covers 4 rows of ITS column
            // (rows r0..r0+1, 8+r0..8+r0+1), and the 4 owner lanes (base+2p, p=0..3, same
            // ch/cbase) cover all 16 rows — so the column partial needs only 4 lane sums
            for (int r = 0; r < M; ++r)
                for (int s = 0; s < 2; ++s)
                    lsum[(size_t) lane * (M * 2) + r * 2 + s] =
                        acc[r][s][0] + acc[r][s][1] + acc[r][s][2] + acc[r][s][3];
            sycl::group_barrier(it.get_group());
            // each lane reduces its own column from the 4 owner lanes
            const int base = 2 * (c8 * 8 + c1 * 4) + (lane & 1);
            for (int r = 0; r < M; ++r)
                for (int s = 0; s < 2; ++s)
                {
                    float t = 0.0f;
                    #pragma unroll
                    for (int pp = 0; pp < 4; ++pp)
                        t += lsum[(size_t) (base + 2 * pp) * (M * 2) + r * 2 + s];
                    scratch[(size_t) (r * KS + kc) * size_n + kb_n * 32 + s * 16 + (lane & 1) * 8 + cbase] = t;
                }
        });
    });

    // reduce: sum the KS fp32 partial slices per (row, col) into C
    q.submit([&](sycl::handler& h)
    {
        h.parallel_for<GemvMtpReduceName<M, K, cb, LANES, KS>>(
            sycl::range<1>((size_t) M * size_n), [=](sycl::item<1> item)
        {
            const int idx = (int) item.get_linear_id();
            const int r = idx / size_n;
            const int i = idx - r * size_n;
            float s = 0.0f;
            #pragma unroll
            for (int c = 0; c < KS; ++c) s += scratch[(size_t) (r * KS + c) * size_n + i];
            if (c_fp32) ((float*) C)[idx] = s;
            else        ((sycl::half*) C)[idx] = sycl::half(s);
        });
    });
}

// MTP verify GEMV table: the mt2-style rows (K 1..5, cb 0..2) pointing at launch_gemv_mtp
template <int K, int M, int LANES, int KS>
inline void expand_cb_gemv_mtp(GemvFn (&table)[27])
{
    table[K - 1]      = &launch_gemv_mtp<M, K, 0, LANES, KS>;
    table[K - 1 + 8]  = &launch_gemv_mtp<M, K, 1, LANES, KS>;
    table[K - 1 + 16] = &launch_gemv_mtp<M, K, 2, LANES, KS>;
}

template <int M, int KS>
static const GemvFn* gemv_mtp_table()
{
    static GemvFn table[27] = {};
    static bool init = false;
    if (!init)
    {
        expand_cb_gemv_mtp<1, M, 64, KS>(table); expand_cb_gemv_mtp<2, M, 64, KS>(table);
        expand_cb_gemv_mtp<3, M, 64, KS>(table); expand_cb_gemv_mtp<4, M, 64, KS>(table);
        expand_cb_gemv_mtp<5, M, 64, KS>(table);
        init = true;
    }
    return table;
}

// Decode GEMV dispatch: the tile-once kernel for the m == 1 decode path where either
// (a) there are enough n-subtiles to keep the one-group-per-subtile layout occupied
// (size_n/16 >= 256, i.e. size_n >= 4096 — at k=2560 measured 1.85x at n=4096, 2.8x at n=9216,
// and 14.8x on the 4B's K=8 lm_head n=248320, whose serial launch was reading the packed stream
// 16x-amplified at only ~20 GB/s), or (b) size_k >= 4096, where the serial kernel's per-column
// layout re-reads every subtile's tile stream 16x (the 27B's shapes; 2.1x end-to-end there).
// Below both (short-k small-n shapes, e.g. k=2560 n=1024 = 64 subtiles) the per-output kernel
// wins: the tile-once layout doesn't fill the machine and its group reduction only adds overhead.
// Measured on the exact 4B shapes (bench4b.cpp): k=2560 n=1024 serial 2x FASTER, n=2560 k=4096
// serial slightly faster, n=2560 k=9216 tile-once 1.21x, n=4096 tile-once 1.85x, n=9216 2.8x.
// Layer shapes with few subtiles but a long k chain (under ~5120 hardware threads at 64
// lanes/group) go to the KS-split tile-once instead: measured 1.25-2.2x over the unsplit
// tile-once on the 27B's cold production shapes (bench27b.cpp); the lm_head (15k+ subtiles)
// stays unsplit
static inline GemvFn gemv_dispatch(int size_m, int size_k, int size_n, int cbi)
{
    if (size_m != 1)
    {
        // MTP verify (draft_num_tokens 2 + 1 = 3 rows): the mt2-style kernel decodes the trellis
        // once per (ktile, lane) and per row does 2 vector A loads — the generic kernel
        // re-decodes the tile per output row (M x traffic amplification). Needs integer K <= 5
        // (dq4's 4-window group), cb <= 2 (no half-integer rates: cbi >= 24 has no dq4), and
        // size_n % 32 (the tile-pair extent). KS=32 k-chunks for the hidden layers (parallelism-
        // starved shapes, scratch layout [row][chunk] x size_n fits the 32x262144-float cap at
        // size_n <= 40960); the lm_head (n = 248320, 15k+ subtiles) has ample parallelism and
        // takes KS=1 (scratch 3 x 248320 floats)
        if (size_m == 3 && cbi < 24 && (cbi % 8) + 1 <= 5 && size_n % 32 == 0)
        {
            if (size_n <= 40960) return gemv_mtp_table<3, 32>()[cbi];
            return gemv_mtp_table<3, 1>()[cbi];
        }
        return gemv_table()[cbi];
    }
    if (size_k < 4096 && size_n / 16 < 256) return gemv_table()[cbi];
    const int ngroups = size_n / 16;
    if (ngroups * 64 < 163840)
    {
        // MT2: two adjacent n-tiles per group (128B DRAM bursts) + the dq4 decode, KS=32 k-chunks —
        // benched 1.16-1.31x over the KS=8 split on the 27B's cold production shapes (bench27b.cpp).
        // Needs the tile PAIRS to be exact (size_n % 32 == 0), integer K <= 5 (dq4's 4-window group
        // spans 3*bits+16 bits and must fit one word pair), and cb < 2 (the table covers cb 0..2;
        // cbi >= 24 is the half-integer rates, which have no dq4)
        if (size_n % 32 == 0 && cbi < 24 && (cbi % 8) + 1 <= 5)
        {
            // mul1 (cb == 2, cbi 16..23): the ESIMD tile-once GEMV at KS=16 — benched 1.07-1.20x
            // over the vec kernel on the 27B's cold production shapes (k >= 4096, n >= 4096),
            // exact correctness. Outside that range the fixed epilogue/reduce cost dominates
            // (the 2B's k=2048 gate/up measured ~3% slower on the endpoint), and cb 0/1 need the
            // vec kernel anyway (the lop3 decode needs a uint16->half bit reinterpret on simd,
            // which miscompiles here)
            if (cbi >= 16 && size_k >= 4096 && size_n >= 4096)
                return gemv_esimd_table<16>()[cbi];
            return gemv_mt2_table<64, 32>()[cbi];
        }
        return gemv_splitks_table<64, 8>()[cbi];
    }
    // mul1 (cbi 16..23) with K <= 5: the ESIMD tile-once kernel also wins on the many-subtile
    // shapes it was previously excluded from by the scratch cap (the 27B lm_head, n=248320
    // k=2048: 1.16x over the tile-once scalar kernel, bench27b.cpp; the ESIMD reduce writes
    // KS x size_n partials, covered by the enlarged scratch). Needs size_n % 32 (the tile-pair
    // extent). cbi < 24 keeps the half-integer rates on the scalar kernel (no ESIMD row)
    if (cbi >= 16 && cbi < 24 && (cbi % 8) + 1 <= 5 && size_n % 32 == 0)
        return gemv_esimd_table<16>()[cbi];
    // 64 lanes when there are many subtiles (the lm_head: per-lane efficiency wins), 128 when
    // few (the hidden layers: more lanes in flight wins)
    return ((ngroups) >= 4096 ? gemv_splitk_table<64>() : gemv_splitk_table<128>())[cbi];
}

} // namespace

void sycl_exl3_gemm_impl(sycl::queue& q,
                         const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
                         const c10::optional<at::Tensor>& suh,
                         const c10::optional<at::Tensor>& A_had,
                         const c10::optional<at::Tensor>& svh,
                         int64_t force_shape_idx, bool mcg, bool mul1, int64_t force_num_sms)
{
    (void) force_shape_idx;   // unused on SYCL (no per-shape kernel table)
    (void) force_num_sms;     // unused on SYCL (no SM-count heuristic)

    TORCH_CHECK(B.dim() == 3, "exl3_gemm: trellis must be 3D");
    TORCH_CHECK(A.size(-1) == B.size(0) * 16, "exl3_gemm: A's last dim must be trellis K dim * 16");
    TORCH_CHECK(C.size(-1) == B.size(1) * 16, "exl3_gemm: C's last dim must be trellis N dim * 16");
    TORCH_CHECK(A.scalar_type() == at::kHalf, "exl3_gemm: A must be float16");
    TORCH_CHECK(B.scalar_type() == at::kShort, "exl3_gemm: trellis must be uint16");
    const bool c_fp32 = C.scalar_type() == at::kFloat;
    if (!c_fp32) TORCH_CHECK(C.scalar_type() == at::kHalf, "exl3_gemm: C must be float16 or float32");
    TORCH_CHECK(!(mcg && mul1), "exl3_gemm: specified both mcg and mul1");
    TORCH_CHECK(suh.has_value() && A_had.has_value() && svh.has_value(), "exl3_gemm requires suh, A_had and svh");

    int size_m = 1;
    for (int d = 0; d < A.dim() - 1; ++d) size_m *= (int) A.size(d);
    const int size_k = (int) A.size(-1);
    const int size_n = (int) B.size(1) * 16;
    TORCH_CHECK(C.numel() >= (int64_t) size_m * size_n, "exl3_gemm: C must hold one output row per row of A");

    const int tile_u16 = (int) B.size(2);
    const bool half_k = (tile_u16 % 16) != 0;
    const int K = tile_u16 / 16;
    TORCH_CHECK(!half_k || (tile_u16 % 16 == 8 && mul1),
                "exl3_gemm: half-integer bitrates require the mul1 codebook");

    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    // No rows or no output columns: nothing to compute. Empty reduction (k = 0): the product is zero
    if (!size_m || !size_n) return;
    if (!size_k)
    {
        C.zero_();
        return;
    }

    TORCH_CHECK(size_k % 128 == 0, "exl3_gemm: K dimension must be divisible by 128");
    TORCH_CHECK(size_n % 128 == 0, "exl3_gemm: N dimension must be divisible by 128");

    // 1. Input Hadamard: A_had = (A . suh) . H / sqrt(128). The bsz-1 BC scratch follows the
    // linear's out_dtype (fp32 for the GDN projections); like the CUDA kernel, treat the
    // buffer as raw half storage (the fp32 allocation holds at least twice the needed bytes)
    const sycl::half* A_ptr = (const sycl::half*) A.data_ptr();
    if (A_had->scalar_type() == at::kFloat)
        sycl_had_h_raw(q, A_ptr, (sycl::half*) A_had->data_ptr(),
                       (const sycl::half*) suh->data_ptr(), nullptr, 1.0f, size_m, size_k / 128);
    else
        sycl_had_r_128(A, A_had.value(), suh, c10::nullopt, 1.0f);

    // 2. GEMV: C = A_had @ W(hadamard basis)
    int cbi = K - 1;
    if (half_k) cbi += 24;
    else if (cb == 1) cbi += 8;
    else if (cb == 2) cbi += 16;
    TORCH_CHECK(cbi >= 0 && cbi < 27, "exl3_gemm: kernel index out of range: ", cbi);

    // mul1 decode + the fused reduce/output-Hadamard path (one launch instead of the separate
    // reduce and post-Hadamard): same shape gate as the ESIMD dispatch -- K in [1,5] and the
    // split-starved region (size_n < 40960); the lm_head (size_n >= 40960) keeps its splitk kernel
    if (!half_k && cb == 2 && size_m == 1 && K <= 5 && size_n < 40960 && size_n % 32 == 0)
    {
        if (sycl_gemv_mul1_had(q, (const sycl::half*) A_had->data_ptr(),
                               (const uint16_t*) B.data_ptr(), C.data_ptr(), c_fp32,
                               (const sycl::half*) svh->data_ptr(), size_k, size_n, K))
            return;
    }

    gemv_dispatch(size_m, size_k, size_n, cbi)(q,
               (const sycl::half*) A_had->data_ptr(),
               (const uint16_t*) B.data_ptr(),
               C.data_ptr(),
               size_m, size_k, size_n, 0, c_fp32);

    // 3. Output Hadamard in place: C = C . H / sqrt(128) . svh
    if (c_fp32)
        sycl_had_r_128_f(C, C, c10::nullopt, svh, 1.0f);
    else
        sycl_had_r_128_h(C, C, c10::nullopt, svh, 1.0f);
}

// Entry point matching the CUDA binding surface: had(pre=suh) -> GEMV -> had(post=svh)
void sycl_exl3_gemm(const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
                    const c10::optional<at::Tensor>& suh,
                    const c10::optional<at::Tensor>& A_had,
                    const c10::optional<at::Tensor>& svh,
                    int64_t force_shape_idx, bool mcg, bool mul1, int64_t force_num_sms)
{
    sycl_exl3_gemm_impl(exl3_sycl_queue(), A, B, C, suh, A_had, svh,
                        force_shape_idx, mcg, mul1, force_num_sms);
}

// Same flow without the (unused on SYCL) heuristic arguments
void sycl_exl3_gemv(const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
                    const c10::optional<at::Tensor>& suh,
                    const c10::optional<at::Tensor>& A_had,
                    const c10::optional<at::Tensor>& svh, bool mcg, bool mul1)
{
    sycl_exl3_gemm_impl(exl3_sycl_queue(), A, B, C, suh, A_had, svh, -1, mcg, mul1, 0);
}

// GEMV + output Hadamard with an optional row stride (sliced mgemm mode); the input Hadamard
// is the caller's responsibility
void sycl_exl3_gemm_raw2(sycl::queue& q, const sycl::half* A_had,
                         const uint16_t* trellis, void* C, const sycl::half* svh,
                         int size_m, int size_k, int size_n, int stride, bool c_fp32,
                         int K, int cb, bool half_k)
{
    if (!size_m || !size_n || !size_k) return;
    TORCH_CHECK(size_k % 128 == 0 && size_n % 128 == 0, "exl3_gemm: dims must be divisible by 128");

    int cbi = K - 1;
    if (half_k) cbi += 24;
    else if (cb == 1) cbi += 8;
    else if (cb == 2) cbi += 16;
    TORCH_CHECK(cbi >= 0 && cbi < 27, "exl3_gemm: kernel index out of range: ", cbi);
    gemv_dispatch(size_m, size_k, size_n, cbi)(q, A_had, trellis, C, size_m, size_k, size_n, stride, c_fp32);

    if (c_fp32)
        sycl_had_f_raw(q, (const float*) C, (float*) C, nullptr, svh, 1.0f, size_m, size_n / 128, stride);
    else
        sycl_had_h_raw(q, (const sycl::half*) C, (sycl::half*) C, nullptr, svh, 1.0f, size_m, size_n / 128, stride);
}

void sycl_exl3_gemm_raw(sycl::queue& q, const sycl::half* A, sycl::half* A_had,
                        const uint16_t* trellis, void* C, const sycl::half* suh,
                        const sycl::half* svh, int size_m, int size_k, int size_n, bool c_fp32,
                        int K, int cb, bool half_k)
{
    if (!size_m || !size_n || !size_k) return;
    TORCH_CHECK(size_k % 128 == 0 && size_n % 128 == 0, "exl3_gemm: dims must be divisible by 128");

    // 1. Input Hadamard
    sycl_had_h_raw(q, A, A_had, suh, nullptr, 1.0f, size_m, size_k / 128);

    // 2-3. GEMV + output Hadamard
    sycl_exl3_gemm_raw2(q, A_had, trellis, C, svh, size_m, size_k, size_n, 0, c_fp32, K, cb, half_k);
}
