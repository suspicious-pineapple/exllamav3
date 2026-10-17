// EXL3 reconstruct kernels, port of quant/reconstruct.cu into SYCL.
//
// reconstruct / reconstruct_slice: one work-group per (16-row k-block x 128-col n group), 256
// work items. Each item decodes the 8 weights of one subtile (warp_id -> subtile, lane*8 -> t)
// and scatters them directly into the row-major unpacked output with the verified position
// mapping (no shuffles or fragment layout needed).
//
// reconstruct_had_slice: one work-group per 128x128 tile, 256 items. The tile is decoded into a
// local half[128][128] (32 KB), then two 7-stage natural-order Sylvester butterflies (columns
// then rows, each folding a /sqrt(128)) produce W = diag(suh) . H V H /128 . diag(svh), the
// original-basis weights. The butterfly arithmetic runs in fp32 with fp16 storage between
// stages, which rounds slightly differently from the CUDA shuffle version (noted in README.md).

#include <sycl/sycl.hpp>
#include <ATen/Tensor.h>

#include "exl3_compat.h"
#include "exl3_decode.h"

namespace {

constexpr float RH_R_SCALE = 0.08838834764831845f;   // 1/sqrt(128)

// --- plain reconstruct: 256 items per (16-row k block x 128-col n group) --------------------------

template <int K, int cb, bool HALF>
class ReconstructKernelName;

template <int K, int cb, bool HALF>
void launch_reconstruct(sycl::queue& q, sycl::half* unpacked, const uint16_t* packed,
                        int packed_blocks_n, int packed_n_offset, int N,
                        int n_groups, int k_blocks)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);
    constexpr int words = tile_words(K, HALF);
    sycl::range<2> global((size_t) n_groups * 256, (size_t) k_blocks);
    sycl::range<2> local(256, 1);
    q.parallel_for<ReconstructKernelName<K, cb, HALF>>(sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> it)
    {
        const int nb = (int) it.get_group(0);      // 128-wide n group
        const int kb = (int) it.get_group(1);      // 16-row k block
        const int tid = (int) it.get_local_id(0);
        const int w = tid / 32;                    // subtile index 0..7 within the group
        const int lane = tid % 32;
        const int n_sub = nb * 8 + w;              // absolute subtile index along n

        const uint16_t* tile = packed + (size_t) (kb * packed_blocks_n + packed_n_offset + n_sub) * packed_size;
        PackedPtrU16 ptr{ tile, words };

        sycl::half vals[8];
        if constexpr (HALF)
            dq8_half<K, cb>(ptr, lane * 8, vals);
        else
            dq8<K, cb>(ptr, lane * 8, vals);

        const int n_base = nb * 128 + w * 16;
        for (int j = 0; j < 8; ++j)
        {
            int R, C;
            weight_t_to_rc(lane * 8 + j, &R, &C);
            unpacked[(size_t) (kb * 16 + R) * N + n_base + C] = vals[j];
        }
    });
}

// --- fused reconstruct + both-side Hadamard: 256 items per 128x128 tile ---------------------------

template <int K, int cb, bool HALF>
class ReconstructHadKernelName;

// One in-place 7-stage natural-order Sylvester butterfly stage along `axis` of the 128x128 local
// tile (axis 0 = row index / k, 1 = column index / n); `scale` is folded into the last stage
template <typename TILE>
inline void had_stage(TILE& tile, int axis, int s, float scale,
                      sycl::nd_item<2> it, int tid)
{
    const bool last = (s == 6);
    const int b = 1 << s;
    // Pairs (a, a + b) with bit s of `a` clear: 128 lines x 64 pairs per stage
    for (int idx = tid; idx < 128 * 64; idx += 256)
    {
        int line = idx >> 6;           // the non-transformed coordinate
        int pp = idx & 63;             // pair index within the line
        int a = ((pp >> s) << (s + 1)) | (pp & (b - 1));
        int c2 = a | b;
        float va, vb;
        if (axis == 0)
        {
            va = (float) tile[a][line];
            vb = (float) tile[c2][line];
        }
        else
        {
            va = (float) tile[line][a];
            vb = (float) tile[line][c2];
        }
        if (last)
        {
            if (axis == 0)
            {
                tile[a][line]  = sycl::half((va + vb) * scale);
                tile[c2][line] = sycl::half((va - vb) * scale);
            }
            else
            {
                tile[line][a]  = sycl::half((va + vb) * scale);
                tile[line][c2] = sycl::half((va - vb) * scale);
            }
        }
        else
        {
            if (axis == 0)
            {
                tile[a][line]  = sycl::half(va + vb);
                tile[c2][line] = sycl::half(va - vb);
            }
            else
            {
                tile[line][a]  = sycl::half(va + vb);
                tile[line][c2] = sycl::half(va - vb);
            }
        }
    }
}

template <int K, int cb, bool HALF>
void launch_reconstruct_had(sycl::queue& q, sycl::half* unpacked, const uint16_t* packed,
                            const sycl::half* suh, const sycl::half* svh,
                            int packed_blocks_n, int packed_n_offset, int N,
                            int n_groups, int k_groups)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);
    constexpr int words = tile_words(K, HALF);
    sycl::range<2> global((size_t) n_groups * 256, (size_t) k_groups);
    sycl::range<2> local(256, 1);
    q.submit([&](sycl::handler& cgh)
    {
        sycl::local_accessor<uint32_t, 1> s_packed(sycl::range<1>(64 * words), cgh);
        sycl::local_accessor<sycl::half, 2> tile(sycl::range<2>(128, 128), cgh);
        cgh.parallel_for<ReconstructHadKernelName<K, cb, HALF>>(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it)
        {
            const int nb = (int) it.get_group(0);
            const int kb = (int) it.get_group(1);
            const int tid = (int) it.get_local_id(0);
            const int w = tid / 32;
            const int lane = tid % 32;

            // Stage the packed words: subtile (u, w) of the tile is packed block (kb*8 + u,
            // packed_n_offset + nb*8 + w)
            for (int idx = tid; idx < 64 * words; idx += 256)
            {
                int sub = idx / words;
                int wi = idx % words;
                int u = sub / 8, wv = sub % 8;
                const uint16_t* t16 = packed +
                    (size_t) ((kb * 8 + u) * packed_blocks_n + packed_n_offset + nb * 8 + wv) * packed_size;
                s_packed[idx] = (uint32_t) t16[2 * wi] | ((uint32_t) t16[2 * wi + 1] << 16);
            }
            sycl::group_barrier(it.get_group());

            // Decode: each item decodes the 8 weights of subtile (u, w) at t = lane*8 + 0..7, for
            // u = 0..7, scattering with the verified mapping into the local tile
            for (int u = 0; u < 8; ++u)
            {
                PackedPtrLocal<sycl::local_accessor<uint32_t, 1>> ptr{ s_packed, (u * 8 + w) * words, words };
                sycl::half vals[8];
                if constexpr (HALF)
                    dq8_half<K, cb>(ptr, lane * 8, vals);
                else
                    dq8<K, cb>(ptr, lane * 8, vals);
                for (int j = 0; j < 8; ++j)
                {
                    int R, C;
                    weight_t_to_rc(lane * 8 + j, &R, &C);
                    tile[u * 16 + R][w * 16 + C] = vals[j];
                }
            }
            sycl::group_barrier(it.get_group());

            // Column transform (along n), then row transform (along k), each with a folded /sqrt(128).
            // H is symmetric, so the order is immaterial in exact arithmetic
            for (int s = 0; s < 7; ++s)
            {
                had_stage(tile, 1, s, RH_R_SCALE, it, tid);
                sycl::group_barrier(it.get_group());
            }
            for (int s = 0; s < 7; ++s)
            {
                had_stage(tile, 0, s, RH_R_SCALE, it, tid);
                sycl::group_barrier(it.get_group());
            }

            // Scale by the sign/scale vectors and store rows coalesced
            for (int idx = tid; idx < 128 * 128; idx += 256)
            {
                int r = idx / 128;
                int c = idx % 128;
                sycl::half v = hmul(tile[r][c], suh[kb * 128 + r]);
                v = hmul(v, svh[nb * 128 + c]);
                unpacked[(size_t) (kb * 128 + r) * N + nb * 128 + c] = v;
            }
        });
    });
}

// --- kernel-instance tables (cbi = cb*8 + K - 1; 24 + K - 1 for the half-integer rates) -----------

using ReconFn = void (*)(sycl::queue&, sycl::half*, const uint16_t*, int, int, int, int, int);
using ReconHadFn = void (*)(sycl::queue&, sycl::half*, const uint16_t*, const sycl::half*,
                            const sycl::half*, int, int, int, int, int);

template <int K, int cb> struct ReconInst { static constexpr ReconFn fn = launch_reconstruct<K, cb, false>; };
template <int K, int cb> struct ReconHadInst { static constexpr ReconHadFn fn = launch_reconstruct_had<K, cb, false>; };

template <int K> struct ReconHalfInst { static constexpr ReconFn fn = launch_reconstruct<K, 2, true>; };
template <int K> struct ReconHalfHadInst { static constexpr ReconHadFn fn = launch_reconstruct_had<K, 2, true>; };

template <int... Ks, int... Cs, typename F>
constexpr auto make_table(F f, std::integer_sequence<int, Ks...>, std::integer_sequence<int, Cs...>)
{
    return f(ReconInst<Ks, Cs>::fn...);
}

template <int K>
constexpr void expand_cb_recon(ReconFn (&table)[27])
{
    table[K - 1]        = launch_reconstruct<K, 0, false>;
    table[K - 1 + 8]    = launch_reconstruct<K, 1, false>;
    table[K - 1 + 16]   = launch_reconstruct<K, 2, false>;
}

template <int K>
constexpr void expand_cb_had(ReconHadFn (&table)[27])
{
    table[K - 1]        = launch_reconstruct_had<K, 0, false>;
    table[K - 1 + 8]    = launch_reconstruct_had<K, 1, false>;
    table[K - 1 + 16]   = launch_reconstruct_had<K, 2, false>;
}

} // namespace

void sycl_reconstruct_impl(sycl::queue& q, at::Tensor unpacked, at::Tensor packed,
                           float K_, bool mcg, bool mul1, int64_t n_offset);
void sycl_reconstruct_had_impl(sycl::queue& q, at::Tensor unpacked, at::Tensor packed,
                               at::Tensor suh, at::Tensor svh, float K_, bool mcg, bool mul1,
                               int64_t n_offset);
void sycl_reconstruct_slice(
    at::Tensor unpacked, at::Tensor packed, float K_, bool mcg, bool mul1, int64_t n_offset);
void sycl_reconstruct(at::Tensor unpacked, at::Tensor packed, float K_, bool mcg, bool mul1);
void sycl_reconstruct_had_slice(
    at::Tensor unpacked, at::Tensor packed, at::Tensor suh, at::Tensor svh,
    float K_, bool mcg, bool mul1, int64_t n_offset);

void sycl_reconstruct(at::Tensor unpacked, at::Tensor packed, float K_, bool mcg, bool mul1)
{
    sycl_reconstruct_impl(exl3_sycl_queue(), unpacked, packed, K_, mcg, mul1, 0);
}

void sycl_reconstruct_had_slice(at::Tensor unpacked, at::Tensor packed, at::Tensor suh,
                                at::Tensor svh, float K_, bool mcg, bool mul1, int64_t n_offset)
{
    sycl_reconstruct_had_impl(exl3_sycl_queue(), unpacked, packed, suh, svh, K_, mcg, mul1, n_offset);
}

void sycl_reconstruct_impl(sycl::queue& q, at::Tensor unpacked, at::Tensor packed,
                           float K_, bool mcg, bool mul1, int64_t n_offset)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    TORCH_CHECK(!bk.half || mul1, "reconstruct: half-integer bitrates require the mul1 codebook");
    TORCH_CHECK(unpacked.size(0) == packed.size(0) * 16, "reconstruct: unpacked K dim must be packed K dim * 16");
    TORCH_CHECK(packed.size(2) == 16 * K + (bk.half ? 8 : 0), "reconstruct: packed tile size mismatch");
    TORCH_CHECK(unpacked.scalar_type() == at::kHalf, "reconstruct: unpacked must be float16");

    const int rows = (int) packed.size(0);
    const int packed_cols = (int) packed.size(1);

    TORCH_CHECK(unpacked.size(1) % 128 == 0, "reconstruct: unpacked N dimension must be divisible by 128");
    TORCH_CHECK(n_offset % 128 == 0, "reconstruct: n_offset must be divisible by 128");
    TORCH_CHECK(n_offset >= 0, "reconstruct: n_offset must be non-negative");
    TORCH_CHECK(n_offset + unpacked.size(1) <= packed.size(1) * 16, "reconstruct: slice exceeds packed tensor bounds");

    if (unpacked.numel() == 0)
        return;

    const int n_groups = (int) unpacked.size(1) / 128;
    const int k_blocks = rows;
    const int packed_n_offset = (int) (n_offset / 16);

    int cbi = K - 1;
    if (bk.half) cbi += 24;
    else if (mcg) cbi += 8;
    else if (mul1) cbi += 16;
    TORCH_CHECK(cbi >= 0 && cbi < 27, "reconstruct: kernel index out of range: ", cbi);

    static ReconFn table[27] = {};
    static bool init = false;
    if (!init)
    {
        expand_cb_recon<1>(table); expand_cb_recon<2>(table); expand_cb_recon<3>(table);
        expand_cb_recon<4>(table); expand_cb_recon<5>(table); expand_cb_recon<6>(table);
        expand_cb_recon<7>(table); expand_cb_recon<8>(table);
        table[24] = launch_reconstruct<1, 2, true>;
        table[25] = launch_reconstruct<2, 2, true>;
        table[26] = launch_reconstruct<3, 2, true>;
        init = true;
    }
    table[cbi](q, (sycl::half*) unpacked.data_ptr(), (const uint16_t*) packed.data_ptr(),
               packed_cols, packed_n_offset, (int) unpacked.size(1), n_groups, k_blocks);
}

void sycl_reconstruct_had_impl(sycl::queue& q, at::Tensor unpacked, at::Tensor packed,
                               at::Tensor suh, at::Tensor svh, float K_, bool mcg, bool mul1,
                               int64_t n_offset)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    TORCH_CHECK(!bk.half || mul1, "reconstruct_had: half-integer bitrates require the mul1 codebook");
    TORCH_CHECK(unpacked.size(0) == packed.size(0) * 16, "reconstruct_had: unpacked K dim must be packed K dim * 16");
    TORCH_CHECK(packed.size(2) == 16 * K + (bk.half ? 8 : 0), "reconstruct_had: packed tile size mismatch");
    TORCH_CHECK(unpacked.scalar_type() == at::kHalf, "reconstruct_had: unpacked must be float16");
    TORCH_CHECK(suh.scalar_type() == at::kHalf && svh.scalar_type() == at::kHalf, "reconstruct_had: suh/svh must be float16");

    TORCH_CHECK(unpacked.size(0) % 128 == 0, "reconstruct_had: K dimension must be divisible by 128");
    TORCH_CHECK(unpacked.size(1) % 128 == 0, "reconstruct_had: N dimension must be divisible by 128");
    TORCH_CHECK(n_offset % 128 == 0, "reconstruct_had: n_offset must be divisible by 128");
    TORCH_CHECK(n_offset >= 0, "reconstruct_had: n_offset must be non-negative");
    TORCH_CHECK(n_offset + unpacked.size(1) <= packed.size(1) * 16, "reconstruct_had: slice exceeds packed tensor bounds");
    TORCH_CHECK(suh.numel() >= unpacked.size(0), "reconstruct_had: suh size");
    TORCH_CHECK(svh.numel() >= unpacked.size(1), "reconstruct_had: svh size");

    if (unpacked.numel() == 0)
        return;

    const int n_groups = (int) unpacked.size(1) / 128;
    const int k_groups = (int) unpacked.size(0) / 128;

    int cbi = K - 1;
    if (bk.half) cbi += 24;
    else if (mcg) cbi += 8;
    else if (mul1) cbi += 16;
    TORCH_CHECK(cbi >= 0 && cbi < 27, "reconstruct_had: kernel index out of range: ", cbi);

    static ReconHadFn table[27] = {};
    static bool init = false;
    if (!init)
    {
        expand_cb_had<1>(table); expand_cb_had<2>(table); expand_cb_had<3>(table);
        expand_cb_had<4>(table); expand_cb_had<5>(table); expand_cb_had<6>(table);
        expand_cb_had<7>(table); expand_cb_had<8>(table);
        table[24] = launch_reconstruct_had<1, 2, true>;
        table[25] = launch_reconstruct_had<2, 2, true>;
        table[26] = launch_reconstruct_had<3, 2, true>;
        init = true;
    }
    table[cbi](q, (sycl::half*) unpacked.data_ptr(), (const uint16_t*) packed.data_ptr(),
               (const sycl::half*) suh.data_ptr(), (const sycl::half*) svh.data_ptr(),
               (int) packed.size(1), (int) (n_offset / 16), (int) unpacked.size(1),
               n_groups, k_groups);
}
