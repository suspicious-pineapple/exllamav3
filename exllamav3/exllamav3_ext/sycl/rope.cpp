// RoPE (rotary position embeddings), SYCL port of quant/rope.cu rope_kernel.
//
// One work-item per (batch, token, head); each work-item processes a whole head in fp32:
// optional per-head RMS norm (q_norm/k_norm), then the rotation (NEOX pairs (i, i + D/2) or
// GPTJ interleaved pairs), then the Llama 4 position-dependent scale (queries only). The
// head lives in global memory between passes, so no local memory is needed and out_q may
// alias q (in-place apply): every element is read exactly once before its single write.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <sycl/sycl.hpp>

#include "exl3_compat.h"
#include "sycl_kernels.h"

#define ROPESTYLE_NONE 0
#define ROPESTYLE_GPTJ 1
#define ROPESTYLE_NEOX 2
#define MAX_ROTATE_DIMS 4

namespace
{
    template <int rope_mode> class RopeKernelName;

    inline void sincos_accurate(float x, float* s, float* c)
    {
        *s = sycl::sincos(x, c);
    }

    template <int rope_mode>
    void rope_impl(
        sycl::queue& q,
        const sycl::half* q_ptr, sycl::half* out_q_ptr,
        const sycl::half* k_ptr, sycl::half* out_k_ptr,
        const float* inv_freq,
        int bsz, int seq_len, int num_heads_q, int num_heads_k,
        int head_dim, int q_head_stride, int k_head_stride,
        int partial_head_dim,
        int position,
        const uint32_t* positions,
        const uint32_t* position_ids,
        int position_ids_stride,
        float attn_factor,
        const void* q_norm, const void* k_norm,
        bool norm_bf16,
        float norm_eps, float norm_constant_bias,
        bool inv_freq_table, int inv_freq_stride,
        float llama_4_scaling_beta, int llama_4_scaling_original,
        int rotate_dims, int rotate_offset)
    {
        const long total_heads = (long) bsz * seq_len * (num_heads_q + num_heads_k);
        if (!total_heads) return;
        constexpr int NT = 64;
        const size_t groups = ((size_t) total_heads + NT - 1) / NT;
        const int Dp = partial_head_dim;
        const int pairs = Dp / 2;

        q.parallel_for<RopeKernelName<rope_mode>>(sycl::range<1>(groups * NT), [=](sycl::item<1> item)
        {
            const long idx = item.get_linear_id();
            if (idx >= total_heads) return;

            // Unpack (batch, token, head); heads walk q first, then k
            const int heads_per_tok = num_heads_q + num_heads_k;
            const long tok = idx / heads_per_tok;
            const int bi = (int) (tok / seq_len);
            const int si = (int) (tok % seq_len);
            const int hi = (int) (idx % heads_per_tok);

            const bool is_q = hi < num_heads_q;
            const sycl::half* src = is_q
                ? q_ptr + (tok * num_heads_q + hi) * q_head_stride
                : k_ptr + (tok * num_heads_k + hi - num_heads_q) * k_head_stride;
            sycl::half* dst = is_q
                ? out_q_ptr + (tok * num_heads_q + hi) * q_head_stride
                : out_k_ptr + (tok * num_heads_k + hi - num_heads_q) * k_head_stride;

            // Position for each rotate dim (mrope position_ids vary by rdim)
            int pos_cache[MAX_ROTATE_DIMS];
            for (int rdim = 0; rdim < rotate_dims; ++rdim)
            {
                int pos = si + position;
                if (positions)
                    pos = si + (int) positions[bi];
                else if (position_ids)
                {
                    long pidx = ((long) bi * seq_len + si) * position_ids_stride;
                    if (position_ids_stride > 1) pidx += rdim;
                    pos = (int) position_ids[pidx];
                }
                pos_cache[rdim] = pos;
            }

            // Per-head RMS norm: sum of squares over the whole head
            float rmf = 1.0f;
            const bool do_norm = is_q ? (q_norm != nullptr) : (k_norm != nullptr);
            if (do_norm)
            {
                float sum = 0.0f;
                for (int i = 0; i < head_dim; ++i)
                {
                    float v = (float) src[i];
                    sum += v * v;
                }
                rmf = sycl::rsqrt(sum / (float) head_dim + norm_eps);
            }

            // Norm scale for element i: rmf * (weight[i] + bias). The CUDA kernel multiplies
            // the normed value by the weight in the weight's precision; this port keeps fp32
            auto norm_scale = [&](int i) -> float
            {
                float w = norm_constant_bias;
                if (norm_bf16)
                {
                    const sycl::ext::oneapi::bfloat16* wptr =
                        (const sycl::ext::oneapi::bfloat16*) (is_q ? q_norm : k_norm);
                    w += (float) wptr[i];
                }
                else
                {
                    const sycl::half* wptr = (const sycl::half*) (is_q ? q_norm : k_norm);
                    w += (float) wptr[i];
                }
                return rmf * w;
            };

            // Llama 4 position-dependent scale, queries only (keys enter the cache unscaled)
            float l4_scaling = 1.0f;
            if (llama_4_scaling_beta > 0.0f && is_q)
                l4_scaling = 1.0f + llama_4_scaling_beta *
                    sycl::log(1.0f + (float) (pos_cache[0] / llama_4_scaling_original));

            // Rotate and write. Pairs (offset + t, offset + t + Dp/2) cover each rotated
            // element exactly once, so reads never see this work-item's own writes
            for (int rdim = 0; rdim < rotate_dims; ++rdim)
            {
                const int offset = rotate_offset + Dp * rdim;
                float sin = 0.0f, cos = 0.0f;
                if (inv_freq_table)
                {
                    // Table mode (mrope): precomputed sincos per position
                    const int p = pos_cache[rdim];
                    sincos_accurate(1.0f, &sin, &cos);   // placeholder, overwritten below
                    for (int t = 0; t < pairs; ++t)
                    {
                        float fr = inv_freq[(long) p * pairs + t];
                        sincos_accurate(fr, &sin, &cos);
                        sin *= attn_factor;
                        cos *= attn_factor;
                        float v1 = (float) src[offset + t];
                        float v2 = (float) src[offset + t + pairs];
                        if (do_norm)
                        {
                            v1 *= norm_scale(offset + t);
                            v2 *= norm_scale(offset + t + pairs);
                        }
                        v1 *= l4_scaling;
                        v2 *= l4_scaling;
                        if constexpr (rope_mode == ROPESTYLE_NEOX)
                        {
                            dst[offset + t] = sycl::half(v1 * cos - v2 * sin);
                            dst[offset + t + pairs] = sycl::half(v2 * cos + v1 * sin);
                        }
                        else
                        {
                            dst[offset + t] = sycl::half(v1 * cos - v2 * sin);
                            dst[offset + t + pairs] = sycl::half(v1 * sin + v2 * cos);
                        }
                    }
                    continue;
                }

                // Scalar inv_freq: fr = inv_freq[t], angle = fr * pos
                for (int t = 0; t < pairs; ++t)
                {
                    float fr = inv_freq[t];
                    float pf = (float) pos_cache[rdim];
                    sincos_accurate(fr * pf, &sin, &cos);
                    sin *= attn_factor;
                    cos *= attn_factor;
                    float v1 = (float) src[offset + t];
                    float v2 = (float) src[offset + t + pairs];
                    if (do_norm)
                    {
                        v1 *= norm_scale(offset + t);
                        v2 *= norm_scale(offset + t + pairs);
                    }
                    v1 *= l4_scaling;
                    v2 *= l4_scaling;
                    dst[offset + t] = sycl::half(v1 * cos - v2 * sin);
                    dst[offset + t + pairs] = sycl::half(v2 * cos + v1 * sin);
                }
            }

            // Elements outside the rotated range: norm still applies to the whole head
            const int rotated = rotate_offset + Dp * rotate_dims;
            if (rotated < head_dim)
            {
                for (int i = 0; i < head_dim; ++i)
                {
                    if (i >= rotate_offset && i < rotated) continue;
                    float v = (float) src[i];
                    if (do_norm) v *= norm_scale(i);
                    v *= l4_scaling;
                    dst[i] = sycl::half(v);
                }
            }
        });
    }
}

void sycl_rope(
    const at::Tensor& q, at::Tensor& out_q,
    const c10::optional<at::Tensor>& k, c10::optional<at::Tensor>& out_k,
    const at::Tensor& inv_freq,
    int64_t position,
    const c10::optional<at::Tensor>& positions,
    const c10::optional<at::Tensor>& position_ids,
    int64_t rope_mode,
    float attn_factor,
    const c10::optional<at::Tensor>& q_norm,
    const c10::optional<at::Tensor>& k_norm,
    float norm_eps,
    float norm_constant_bias,
    float llama_4_scaling_beta,
    int64_t llama_4_scaling_original,
    int64_t rotate_dims,
    int64_t rotate_offset)
{
    TORCH_CHECK(q.scalar_type() == at::kHalf, "rope: q must be float16");
    TORCH_CHECK(q.dim() == 4, "rope: q must be [B, S, H, D]");
    TORCH_CHECK(k.has_value() == out_k.has_value(), "rope: k and out_k must be given together");
    TORCH_CHECK(rope_mode == ROPESTYLE_GPTJ || rope_mode == ROPESTYLE_NEOX, "rope: invalid rope_mode");
    TORCH_CHECK(rotate_dims > 0 && rotate_dims <= MAX_ROTATE_DIMS, "rope: rotate_dims out of range");

    const int bsz = (int) q.size(0);
    const int seq_len = (int) q.size(1);
    const int num_heads_q = (int) q.size(2);
    const int head_dim = (int) q.size(3);
    const int q_head_stride = (int) q.stride(2);
    const int k_head_stride = k.has_value() ? (int) k->stride(2) : 0;
    const int partial_head_dim = (int) inv_freq.size(-1) * 2;
    TORCH_CHECK(head_dim % 2 == 0, "rope: head_dim must be even");
    TORCH_CHECK(rotate_dims == 1 || head_dim == partial_head_dim * (int) rotate_dims,
                "rope: rotate_dims is inconsistent with inv_freq and head_dim");
    TORCH_CHECK(rotate_offset >= 0 && rotate_offset + partial_head_dim * (int) rotate_dims <= head_dim,
                "rope: rotate_offset out of range");

    int num_heads_k = 0;
    if (k.has_value())
    {
        TORCH_CHECK(k->scalar_type() == at::kHalf, "rope: k must be float16");
        TORCH_CHECK(k->dim() == 4, "rope: k must be [B, S, H, D]");
        TORCH_CHECK(k->size(0) == bsz && k->size(1) == seq_len && k->size(3) == head_dim,
                    "rope: k is incorrect shape");
        num_heads_k = (int) k->size(2);
    }

    TORCH_CHECK(inv_freq.scalar_type() == at::kFloat, "rope: inv_freq must be float32");
    bool inv_freq_table = inv_freq.dim() > 1;
    int inv_freq_stride = 0;
    if (inv_freq_table)
        inv_freq_stride = (int) inv_freq.size(-1) * inv_freq.size(-2);

    TORCH_CHECK((positions.has_value() ? 1 : 0) + (position_ids.has_value() ? 1 : 0) <= 1,
                "rope: positions and position_ids are mutually exclusive");
    if (positions.has_value())
    {
        TORCH_CHECK(positions->dim() == 1 && positions->size(0) == bsz, "rope: positions shape");
        TORCH_CHECK(positions->scalar_type() == at::kInt, "rope: positions must be int32");
    }
    int position_ids_stride = 1;
    if (position_ids.has_value())
    {
        TORCH_CHECK(position_ids->scalar_type() == at::kInt, "rope: position_ids must be int32");
        const int rd = position_ids->dim();
        TORCH_CHECK(rd == 2 || (rd == 3 && position_ids->size(-1) == rotate_dims),
                    "rope: position_ids wrong number of dims");
        TORCH_CHECK(position_ids->size(0) == bsz && position_ids->size(1) == seq_len,
                    "rope: position_ids is incorrect shape");
        if (rd == 3) position_ids_stride = (int) rotate_dims;
    }

    // Per-head RMS norm weights (fp16 or bf16), both or neither
    TORCH_CHECK(q_norm.has_value() || !k_norm.has_value(), "rope: k_norm requires q_norm");
    TORCH_CHECK(!q_norm.has_value() || !k.has_value() || k_norm.has_value(),
                "rope: q_norm with k requires k_norm");
    bool norm_bf16 = false;
    if (q_norm.has_value())
    {
        TORCH_CHECK(q_norm->dim() == 1 && q_norm->size(0) == head_dim, "rope: q_norm is incorrect size");
        norm_bf16 = q_norm->scalar_type() == at::kBFloat16;
        TORCH_CHECK(norm_bf16 || q_norm->scalar_type() == at::kHalf, "rope: unsupported norm dtype");
        if (k_norm.has_value())
            TORCH_CHECK(k_norm->scalar_type() == q_norm->scalar_type() &&
                        k_norm->dim() == 1 && k_norm->size(0) == head_dim,
                        "rope: k_norm is incorrect size or dtype");
    }

    if (!bsz || !seq_len || !(num_heads_q + num_heads_k) || !head_dim) return;

    sycl::queue& q_queue = exl3_sycl_queue();
    const sycl::half* q_ptr = (const sycl::half*) q.data_ptr();
    sycl::half* out_q_ptr = (sycl::half*) out_q.data_ptr();
    const sycl::half* k_ptr = k.has_value() ? (const sycl::half*) k->data_ptr() : nullptr;
    sycl::half* out_k_ptr = out_k.has_value() ? (sycl::half*) out_k->data_ptr() : nullptr;
    const float* inv_freq_ptr = (const float*) inv_freq.data_ptr();
    const uint32_t* positions_ptr = positions.has_value() ? (const uint32_t*) positions->data_ptr() : nullptr;
    const uint32_t* position_ids_ptr = position_ids.has_value() ? (const uint32_t*) position_ids->data_ptr() : nullptr;
    const void* q_norm_ptr = q_norm.has_value() ? (const void*) q_norm->data_ptr() : nullptr;
    const void* k_norm_ptr = k_norm.has_value() ? (const void*) k_norm->data_ptr() : nullptr;

    if (rope_mode == ROPESTYLE_NEOX)
        rope_impl<ROPESTYLE_NEOX>(q_queue, q_ptr, out_q_ptr, k_ptr, out_k_ptr, inv_freq_ptr,
            bsz, seq_len, num_heads_q, num_heads_k, head_dim, q_head_stride, k_head_stride,
            partial_head_dim, (int) position, positions_ptr, position_ids_ptr, position_ids_stride,
            attn_factor, q_norm_ptr, k_norm_ptr, norm_bf16, norm_eps, norm_constant_bias,
            inv_freq_table, inv_freq_stride, llama_4_scaling_beta, (int) llama_4_scaling_original,
            (int) rotate_dims, (int) rotate_offset);
    else
        rope_impl<ROPESTYLE_GPTJ>(q_queue, q_ptr, out_q_ptr, k_ptr, out_k_ptr, inv_freq_ptr,
            bsz, seq_len, num_heads_q, num_heads_k, head_dim, q_head_stride, k_head_stride,
            partial_head_dim, (int) position, positions_ptr, position_ids_ptr, position_ids_stride,
            attn_factor, q_norm_ptr, k_norm_ptr, norm_bf16, norm_eps, norm_constant_bias,
            inv_freq_table, inv_freq_stride, llama_4_scaling_beta, (int) llama_4_scaling_original,
            (int) rotate_dims, (int) rotate_offset);
}
