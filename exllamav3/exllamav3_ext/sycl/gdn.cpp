// Gated Delta Net kernels, ports of gdn.cu into SYCL.
//
// gated_delta_net_fused_op / gated_delta_net_fused_op_2: the elementwise split/gate helpers the
// GDN torch path calls before the conv and the delta-rule scan.
//
// recurrent_gated_delta_rule: the short-sequence (seqlen < num_v_heads, i.e. decode and SD
// verification) gated delta-rule scan, per (batch row, v head, v chunk) work-group, sequential
// over the sequence. The CUDA kernel's warp-shuffle reductions become local-memory reductions;
// the 128x128 head dims (Qwen3.5 / KDA class models) are the supported case, per-head and
// per-k-channel (KDA) decay both. Mamba2 mode and the scan-replay entry are not ported.

#include <sycl/sycl.hpp>
#include <ATen/Tensor.h>
#include <cmath>
#include <algorithm>

#include "exl3_compat.h"
#include "sycl_kernels.h"

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int GDN_SUBK = 4;

inline float gdn_softplus(float x)
{
    if (x > 20.0f) return x;
    return sycl::log1p(sycl::exp(x));
}

// -------------------------------------------------------------------------------------------------
// gated_delta_net_fused_op: split projected qkvz + ba GDN inputs, produce the conv input in the
// channel-major layout [B, F, S], the z gate, beta = sigmoid(b) * beta_scale and
// g = -softplus(a + dt_bias) * exp(a_log). One work item per (t, linear), linear = (b*S + s)*Nk + kh.
// -------------------------------------------------------------------------------------------------

template <int T_MAX>
class GdnFusedOpKernelName;

template <int T_MAX>
void launch_gdn_fused_op(sycl::queue& q,
                         const float* in_qkvz, const float* in_ba,
                         const bf16* dt_bias, const bf16* a_log,
                         bf16* out_qkv, bf16* out_z, bf16* out_beta, float* out_g,
                         size_t B, size_t S, size_t Nk, size_t Ng, size_t Hk, size_t Hv,
                         float beta_scale)
{
    const size_t Nv = Nk * Ng;
    const size_t Fseg = 2 * Hk + 2 * Ng * Hv;
    const size_t Fba = 2 * Ng;
    const size_t Nlin = B * S * Nk;
    const size_t Fout = 2 * Nk * Hk + Nv * Hv;
    const size_t T = (size_t) T_MAX;

    q.parallel_for<GdnFusedOpKernelName<T_MAX>>(sycl::range<2>(T, Nlin),
        [=](sycl::item<2> item)
    {
        const int t = (int) item.get_id(0);
        const size_t linear = item.get_id(1);
        const size_t kh = linear % Nk;
        const size_t s = (linear / Nk) % S;
        const size_t b = (linear / Nk) / S;

        const size_t base_qkvz = (((b * S) + s) * Nk + kh) * Fseg;
        const size_t base_ba = (((b * S) + s) * Nk + kh) * Fba;

        // q block: out_qkv features [kh*Hk, kh*Hk + Hk)
        if (t < (int) Hk)
        {
            const float vq = in_qkvz[base_qkvz + t];
            out_qkv[((b * Fout) + kh * Hk + t) * S + s] = bf16(vq);
        }

        // k block: out_qkv features [Nk*Hk + kh*Hk, + Hk)
        if (t < (int) Hk)
        {
            const float vk = in_qkvz[base_qkvz + Hk + t];
            out_qkv[((b * Fout) + Nk * Hk + kh * Hk + t) * S + s] = bf16(vk);
        }

        // v and z blocks: each Ng*Hv long; source offsets [2*Hk .. 2*Hk + Ng*Hv) and
        // [2*Hk + Ng*Hv .. 2*Hk + 2*Ng*Hv)
        if (t < (int) Hv)
        {
            for (size_t g = 0; g < Ng; ++g)
            {
                const size_t vhead = kh * Ng + g;
                const float vv = in_qkvz[base_qkvz + 2 * Hk + g * Hv + t];
                out_qkv[((b * Fout) + 2 * Nk * Hk + vhead * Hv + t) * S + s] = bf16(vv);

                const float vz = in_qkvz[base_qkvz + 2 * Hk + Ng * Hv + g * Hv + t];
                out_z[(((b * S) + s) * Nv + vhead) * Hv + t] = bf16(vz);
            }
        }

        // b and a from mixed_ba (each Ng long) -> beta, g
        if (t < (int) Ng)
        {
            const size_t vhead = kh * Ng + t;
            const size_t out_va_off = ((b * S) + s) * Nv + vhead;

            const float b = in_ba[base_ba + t];
            out_beta[out_va_off] = bf16(1.0f / (1.0f + sycl::exp(-b)) * beta_scale);

            const float a = in_ba[base_ba + Ng + t];
            const float bi = (float) dt_bias[vhead];
            const float al = (float) a_log[vhead];
            out_g[out_va_off] = -gdn_softplus(a + bi) * sycl::exp(al);
        }
    });
}

// -------------------------------------------------------------------------------------------------
// gated_delta_net_fused_op_2: b/a projections -> beta, g (per element)
// -------------------------------------------------------------------------------------------------

template <typename a_log_T>
class GdnFusedOp2KernelName;

template <typename a_log_T>
void launch_gdn_fused_op_2(sycl::queue& q, const float* in_b, const float* in_a,
                           const bf16* in_dt_bias, const a_log_T* in_a_log,
                           bf16* out_beta, float* out_g, size_t BS, size_t H, float beta_scale)
{
    const size_t N = BS * H;
    q.parallel_for<GdnFusedOp2KernelName<a_log_T>>(sycl::range<1>(N), [=](sycl::item<1> item)
    {
        const size_t i = item.get_linear_id();
        const size_t h = i % H;
        const float beta = 1.0f / (1.0f + sycl::exp(-in_b[i])) * beta_scale;
        const float g = -gdn_softplus(in_a[i] + (float) in_dt_bias[h]) * sycl::exp((float) in_a_log[h]);
        out_beta[i] = bf16(beta);
        out_g[i] = g;
    });
}

// -------------------------------------------------------------------------------------------------
// recurrent gated delta rule, 128x128 head dims. Work-group = (v_chunk_dim, SUBK) items; one
// group per (batch row, v head, v chunk), sequential over the sequence with the state in global
// memory. CHANNELWISE: per-k-channel log-decay (KDA); otherwise per-head decay (GDN).
// -------------------------------------------------------------------------------------------------

template <int V_SPLIT, bool CHANNELWISE>
class GdnScanKernelName;

template <int V_SPLIT, bool CHANNELWISE>
void launch_gdn_scan(sycl::queue& q,
                     const bf16* mixed_qkv, const float* g, const bf16* beta,
                     float* recurrent_state, bf16* core_attn_out,
                     int bsz, int seqlen, int num_k_heads, int num_v_heads,
                     const int* slots, int history_stride, const int* slots_in,
                     int tstride = 1,
                     const float* state_r = nullptr, float* state_w = nullptr)
{
    constexpr int HEAD_DIM = 128;
    constexpr int BTS = HEAD_DIM / GDN_SUBK;             // k channels per k-slice (bt)
    constexpr size_t HEAD_STATE_SIZE = HEAD_DIM * HEAD_DIM;
    const int group = num_v_heads / num_k_heads;
    const int v_chunk_dim = HEAD_DIM / V_SPLIT;
    const float scale = 1.0f / sycl::sqrt((float) HEAD_DIM);
    const size_t slot_size = (size_t) history_stride * group * num_k_heads * HEAD_STATE_SIZE;
    // mixed_qkv token row: [q (k_dim), k (k_dim), v (v_dim)]
    const size_t qrow = (size_t) 2 * HEAD_DIM * num_k_heads + (size_t) HEAD_DIM * num_v_heads;
    const size_t grow = (size_t) num_v_heads * (CHANNELWISE ? HEAD_DIM : 1);
    const size_t beta_row = (size_t) num_v_heads;
    const size_t out_row = (size_t) num_v_heads * HEAD_DIM;

    // The x dimension must span the full k/q head (128) so the bt == 0 items load and L2-normalize
    // all 128 q/k channels; the v-chunk work is then guarded by `t < v_chunk_dim` below. (The CUDA
    // original launches threads.x = MAX(k_head_dim, v_head_dim / v_split); V_SPLIT=4 with 128 heads
    // makes that 128.) Using v_chunk_dim here left channels 32..127 uninitialized in shared memory.
    const int local_x = HEAD_DIM > v_chunk_dim ? HEAD_DIM : v_chunk_dim;
    const sycl::range<3> local((size_t) local_x, GDN_SUBK, 1);
    const sycl::range<3> global((size_t) local_x * bsz * num_v_heads * V_SPLIT, GDN_SUBK, 1);

    q.submit([&](sycl::handler& cgh)
    {
        sycl::local_accessor<float, 1> sh_q(sycl::range<1>(HEAD_DIM), cgh);
        sycl::local_accessor<float, 1> sh_k(sycl::range<1>(HEAD_DIM), cgh);
        sycl::local_accessor<float, 1> sh_scale(sycl::range<1>(2), cgh);
        sycl::local_accessor<float, 2> sh_g(sycl::range<2>(1, HEAD_DIM), cgh);
        sycl::local_accessor<float, 2> sh_dot1(sycl::range<2>(GDN_SUBK, HEAD_DIM), cgh);
        sycl::local_accessor<float, 2> sh_dot2(sycl::range<2>(GDN_SUBK, HEAD_DIM), cgh);

        cgh.parallel_for<GdnScanKernelName<V_SPLIT, CHANNELWISE>>(sycl::nd_range<3>(global, local),
            [=](sycl::nd_item<3> it)
        {
            const int t = (int) it.get_local_id(0);       // v channel within the chunk (k channel when bt == 0)
            const int bt = (int) it.get_local_id(1);      // k slice
            const size_t gid = it.get_group_linear_id();
            const int v_chunk = (int) (gid % V_SPLIT);
            const int head = (int) ((gid / V_SPLIT) % num_v_heads);
            const int bi = (int) (gid / ((size_t) V_SPLIT * num_v_heads));
            const int k_head = head / group;
            const int v_start = v_chunk * v_chunk_dim;

            const int state_slot = slots ? slots[bi] : bi;
            const int state_slot_in = (slots_in && history_stride) ? slots_in[bi] : state_slot;

        bf16* o_row = core_attn_out ? core_attn_out + (size_t) bi * seqlen * out_row : nullptr;

        // Replay mode: explicit read/write state rows (base/scratch) and staged token rows
        // tstride apart; a normal pass keeps state_r/state_w null and walks tokens with stride 1
        float* base_w_fixed = state_w ? state_w : nullptr;
        const float* base_r0 = state_r ? state_r : nullptr;

        for (int s = 0; s < seqlen; ++s)
        {
            const size_t tok = (tstride == 1) ? ((size_t) bi * seqlen + s) : (size_t) s * tstride;
            const bf16* qkv_tok = mixed_qkv + tok * qrow;
            const bf16* gl_q = qkv_tok + (size_t) k_head * HEAD_DIM;
            const bf16* gl_k = qkv_tok + (size_t) (num_k_heads + k_head) * HEAD_DIM;
            const bf16* gl_v = qkv_tok + (size_t) (2 * num_k_heads * HEAD_DIM) + (size_t) head * HEAD_DIM + v_start;

            // The first token reads its initial state from slots_in[bi] (a speculative pass);
            // every token reads and writes the scratch row slots[bi]. In replay mode the rows
            // are the job's absolute base/scratch pointers instead
            float* base_w = base_w_fixed ? base_w_fixed : recurrent_state + (size_t) state_slot * slot_size;
            const float* base_r = (s == 0)
                ? (base_r0 ? base_r0 : recurrent_state + (size_t) state_slot_in * slot_size)
                : base_w;
            const float* gl_rs_r = base_r + (size_t) head * HEAD_STATE_SIZE;
            float* gl_rs_w = base_w + (size_t) head * HEAD_STATE_SIZE;

            const float* g_tok = g + tok * grow;
            const bf16* beta_tok = beta + tok * beta_row;

            // Load q/k heads and L2-normalize (items with bt == 0 hold the k channels)
            if (bt == 0)
            {
                sh_q[t] = (float) gl_q[t];
                sh_k[t] = (float) gl_k[t];
            }
            if (t == 0 && bt == 0)
            {
                float sumq = 0.0f, sumk = 0.0f;
                for (int i = 0; i < HEAD_DIM; ++i)
                {
                    sumq += sh_q[i] * sh_q[i];
                    sumk += sh_k[i] * sh_k[i];
                }
                sh_scale[0] = sycl::rsqrt(sumq + 1e-6f);
                sh_scale[1] = sycl::rsqrt(sumk + 1e-6f);
                if constexpr (CHANNELWISE)
                {
                    for (int i = 0; i < HEAD_DIM; ++i)
                        sh_g[0][i] = sycl::exp(g_tok[(size_t) head * HEAD_DIM + i]);
                }
            }
            sycl::group_barrier(it.get_group());

            if (bt == 0)
            {
                sh_q[t] = sh_q[t] * sh_scale[0];
                sh_k[t] = sh_k[t] * sh_scale[1];
            }
            sycl::group_barrier(it.get_group());

            const float beta_h = (float) beta_tok[head];
            const float g_h = CHANNELWISE ? 1.0f : sycl::exp(g_tok[head]);

            float sreg[BTS];   // the item's state slice, read once (see the dot1 loop below)
            if (t < v_chunk_dim)
            {
                // Dot products with the last state, one partial per k-slice. CHANNELWISE decays
                // per k-channel, so the kv_mem readback reads the decayed state. The slice's
                // state values are kept in registers: the update loop below re-reads exactly
                // these elements (each (bt, t) item's range is disjoint across items and nothing
                // writes the state row between the two loops), so the re-read is redundant
                float sum = 0.0f;
                for (int i = 0; i < BTS; ++i)
                {
                    const int kk = bt * BTS + i;
                    const float gk = CHANNELWISE ? sh_g[0][kk] : 1.0f;
                    const float sv = gl_rs_r[(size_t) kk * HEAD_DIM + v_start + t];
                    sreg[i] = sv;
                    sum += sh_k[kk] * gk * sv;
                }
                sh_dot1[bt][t] = sum;
            }
            sycl::group_barrier(it.get_group());

            if (t < v_chunk_dim)
            {
                float dot1 = 0.0f;
                for (int sl = 0; sl < GDN_SUBK; ++sl) dot1 += sh_dot1[sl][t];
                const float v = (float) gl_v[t] - dot1 * g_h;

                float v_out = 0.0f;
                for (int i = 0; i < BTS; ++i)
                {
                    const int kk = bt * BTS + i;
                    const float gk = CHANNELWISE ? sh_g[0][kk] : g_h;
                    float state = sreg[i];
                    state = state * gk + sh_k[kk] * v * beta_h;
                    gl_rs_w[(size_t) kk * HEAD_DIM + v_start + t] = state;
                    v_out += sh_q[kk] * state;
                }
                sh_dot2[bt][t] = v_out;
            }
            sycl::group_barrier(it.get_group());

            if (t < v_chunk_dim && bt == 0 && o_row)
            {
                float v_out = 0.0f;
                for (int sl = 0; sl < GDN_SUBK; ++sl) v_out += sh_dot2[sl][t];
                o_row[(size_t) s * out_row + (size_t) head * HEAD_DIM + v_start + t] = bf16(v_out * scale);
            }
        }
        });
    });
}

} // namespace

// --- host entry points ----------------------------------------------------------------------------

void sycl_gated_delta_net_fused_op(
    const at::Tensor& mixed_qkvz, const at::Tensor& mixed_ba,
    const at::Tensor& dt_bias, const at::Tensor& a_log,
    at::Tensor& mixed_qkv, at::Tensor& z, at::Tensor& beta, at::Tensor& g,
    int64_t num_k_heads, int64_t num_v_heads, int64_t k_head_dim, int64_t v_head_dim,
    double beta_scale)
{
    const auto B = mixed_qkvz.size(0);
    const auto S = mixed_qkvz.size(1);
    const auto Nk = num_k_heads;
    const auto Hk = k_head_dim;
    const auto Hv = v_head_dim;
    const auto Nv = num_v_heads;

    TORCH_CHECK(Nk > 0 && Nv > 0 && Hk > 0 && Hv > 0, "gdn_fused_op: head counts must be positive");
    TORCH_CHECK(Nv % Nk == 0, "gdn_fused_op: num_v_heads must be divisible by num_k_heads");
    const auto Ng = Nv / Nk;
    const auto Fseg = 2 * Hk + 2 * Ng * Hv;
    TORCH_CHECK(mixed_qkvz.size(2) == Nk * Fseg, "gdn_fused_op: mixed_qkvz last dim mismatch");
    TORCH_CHECK(mixed_ba.dim() == 3 && mixed_ba.size(0) == B && mixed_ba.size(1) == S &&
                mixed_ba.size(2) == Nk * 2 * Ng, "gdn_fused_op: mixed_ba shape mismatch");
    TORCH_CHECK(mixed_qkv.size(0) == B && mixed_qkv.size(1) == 2 * Nk * Hk + Nv * Hv &&
                mixed_qkv.size(2) == S, "gdn_fused_op: mixed_qkv shape mismatch");
    TORCH_CHECK(z.numel() == B * S * Nv * Hv, "gdn_fused_op: z size mismatch");
    TORCH_CHECK(beta.numel() == B * S * Nv && g.numel() == B * S * Nv, "gdn_fused_op: beta/g size mismatch");
    TORCH_CHECK(dt_bias.numel() == Nv && a_log.numel() == Nv, "gdn_fused_op: dt_bias/a_log size mismatch");
    TORCH_CHECK(mixed_qkvz.is_contiguous() && mixed_ba.is_contiguous() && mixed_qkv.is_contiguous() &&
                z.is_contiguous() && beta.is_contiguous() && g.is_contiguous(),
                "gdn_fused_op: tensors must be contiguous");

    if (B * S == 0) return;

    const int threads = (int) std::max(Hk, Hv);
    TORCH_CHECK(threads <= 256, "gdn_fused_op: max head dim exceeded");

    sycl::queue& q = exl3_sycl_queue();
    if (threads <= 128)
        launch_gdn_fused_op<128>(q,
            (const float*) mixed_qkvz.data_ptr(), (const float*) mixed_ba.data_ptr(),
            (const bf16*) dt_bias.data_ptr(), (const bf16*) a_log.data_ptr(),
            (bf16*) mixed_qkv.data_ptr(), (bf16*) z.data_ptr(), (bf16*) beta.data_ptr(),
            (float*) g.data_ptr(), B, S, Nk, Ng, Hk, Hv, (float) beta_scale);
    else
        launch_gdn_fused_op<256>(q,
            (const float*) mixed_qkvz.data_ptr(), (const float*) mixed_ba.data_ptr(),
            (const bf16*) dt_bias.data_ptr(), (const bf16*) a_log.data_ptr(),
            (bf16*) mixed_qkv.data_ptr(), (bf16*) z.data_ptr(), (bf16*) beta.data_ptr(),
            (float*) g.data_ptr(), B, S, Nk, Ng, Hk, Hv, (float) beta_scale);
}

void sycl_gated_delta_net_fused_op_2(
    const at::Tensor& b, const at::Tensor& a, const at::Tensor& dt_bias, const at::Tensor& a_log,
    at::Tensor& beta, at::Tensor& g, double beta_scale)
{
    TORCH_CHECK(b.scalar_type() == at::kFloat && a.scalar_type() == at::kFloat, "gdn_fused_op_2: b/a must be fp32");
    TORCH_CHECK(dt_bias.scalar_type() == at::kBFloat16, "gdn_fused_op_2: dt_bias must be bf16");
    TORCH_CHECK(beta.scalar_type() == at::kBFloat16 && g.scalar_type() == at::kFloat, "gdn_fused_op_2: beta/g dtype");
    TORCH_CHECK(b.sizes() == a.sizes() && b.sizes() == beta.sizes() && b.sizes() == g.sizes(),
                "gdn_fused_op_2: shape mismatch");
    TORCH_CHECK(b.size(2) == dt_bias.numel() && b.size(2) == a_log.numel(), "gdn_fused_op_2: head count mismatch");

    const size_t B = b.size(0), S = b.size(1), H = b.size(2);
    if (b.numel() == 0) return;

    sycl::queue& q = exl3_sycl_queue();
    if (a_log.scalar_type() == at::kFloat)
        launch_gdn_fused_op_2<float>(q, (const float*) b.data_ptr(), (const float*) a.data_ptr(),
            (const bf16*) dt_bias.data_ptr(), (const float*) a_log.data_ptr(),
            (bf16*) beta.data_ptr(), (float*) g.data_ptr(), B * S, H, (float) beta_scale);
    else if (a_log.scalar_type() == at::kBFloat16)
        launch_gdn_fused_op_2<bf16>(q, (const float*) b.data_ptr(), (const float*) a.data_ptr(),
            (const bf16*) dt_bias.data_ptr(), (const bf16*) a_log.data_ptr(),
            (bf16*) beta.data_ptr(), (float*) g.data_ptr(), B * S, H, (float) beta_scale);
    else
        TORCH_CHECK(false, "gdn_fused_op_2: unsupported a_log dtype");
}


// -------------------------------------------------------------------------------------------------
// causal_conv1d_update: the decode-step short conv over the recurrent window (port of
// conv1d_update_kernel in gdn.cu; the CUDA kernel is unavailable here and the Triton path costs
// two launches per layer). One work item per (b, channel): read the K-entry window from the
// conv_state ring, append the incoming token(s), apply silu and write the new window back.
// seqlen <= 32, K <= 16 (the caller guarantees those).
// -------------------------------------------------------------------------------------------------

constexpr int CONV1D_MAX_K = 16;

template <bool ACT, bool HISTORY, bool INCREMENTAL>
class GdnConv1dKernelName;

template <bool ACT, bool HISTORY, bool INCREMENTAL>
static void launch_conv1d_update(sycl::queue& q, const bf16* x, bf16* state, const int* slots,
                                 const bf16* weight, const bf16* bias, bf16* out,
                                 int bsz, int dim, int seqlen, int state_size, int K)
{
    constexpr int NT = 128;
    const size_t n = (size_t) bsz * dim;
    const size_t gsz = ((n + NT - 1) / NT) * NT;
    q.submit([&](sycl::handler& h)
    {
        h.parallel_for<GdnConv1dKernelName<ACT, HISTORY, INCREMENTAL>>(
            sycl::nd_range<1>(sycl::range<1>(gsz), sycl::range<1>(NT)),
            [=](sycl::nd_item<1> it)
        {
            const size_t gid = it.get_global_linear_id();
            if (gid >= n) return;
            const int b = (int) (gid / dim);
            const int d = (int) (gid - (size_t) b * dim);
            const int slot = slots ? slots[b] : b;
            const bf16* x_d = x + ((size_t) b * dim + d) * seqlen;
            bf16* state_d = state + ((size_t) slot * dim + d) * state_size;

            float w[CONV1D_MAX_K];
            for (int k = 0; k < K; ++k) w[k] = (float) weight[(size_t) d * K + k];
            float bias_d = bias ? (float) bias[d] : 0.0f;

            const int win_off = INCREMENTAL ? state_size - K : 0;
            float old_state[CONV1D_MAX_K];
            float win[CONV1D_MAX_K];
            for (int k = 0; k < K; ++k) old_state[k] = (float) state_d[win_off + k];
            for (int k = 0; k < K - 1; ++k) win[k] = old_state[k + 1];
            const int last = K > 0 ? K - 1 : 0;

            for (int s = 0; s < seqlen; ++s)
            {
                win[last] = (float) x_d[s];
                float acc = bias_d;
                for (int k = 0; k < K; ++k) acc = sycl::fma(w[k], win[k], acc);
                if constexpr (ACT) acc *= 1.0f / (1.0f + sycl::exp(-acc));
                out[((size_t) b * seqlen + s) * dim + d] = (bf16) acc;
                for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
            }

            if constexpr (!HISTORY)
            {
                for (int k = 0; k < K; ++k)
                {
                    int src_t = seqlen + k;
                    float v = (src_t < K) ? old_state[src_t] : (float) x_d[src_t - K];
                    state_d[k] = (bf16) v;
                }
            }
            else if constexpr (INCREMENTAL)
            {
                int m = seqlen < state_size ? seqlen : state_size;
                for (int j = 0; j < state_size - m; ++j) state_d[j] = state_d[j + m];
                for (int j = 0; j < m; ++j) state_d[state_size - m + j] = (bf16) (float) x_d[seqlen - m + j];
            }
            else
            {
                int total = K + seqlen;
                int write_size = state_size < total ? state_size : total;
                int dst_start = state_size - write_size;
                int src_start = total - write_size;
                for (int j = 0; j < write_size; ++j)
                {
                    int src_t = src_start + j;
                    float v = (src_t < K) ? old_state[src_t] : (float) x_d[src_t - K];
                    state_d[dst_start + j] = (bf16) v;
                }
            }
        });
    });
}

void sycl_causal_conv1d_update(
    const at::Tensor& x, at::Tensor& conv_state,
    const c10::optional<at::Tensor>& slots,
    const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::Tensor& out, bool activation, bool history, bool incremental)
{
    TORCH_CHECK(!incremental || history, "causal_conv1d_update: incremental steps are history steps");
    const int bsz = (int) x.size(0);
    const int dim = (int) x.size(1);
    const int seqlen = (int) x.size(2);
    const int state_size = (int) conv_state.size(2);
    const int K = (int) weight.size(1);

    TORCH_CHECK(K <= CONV1D_MAX_K, "causal_conv1d_update: conv kernel size exceeds ", CONV1D_MAX_K);
    TORCH_CHECK(state_size >= K, "causal_conv1d_update: conv_state must have at least K entries");
    TORCH_CHECK(x.is_contiguous() && conv_state.is_contiguous() && weight.is_contiguous() && out.is_contiguous(),
                "causal_conv1d_update: x, conv_state, weight and out must be contiguous");
    TORCH_CHECK(conv_state.dim() == 3 && conv_state.size(1) == dim,
                "causal_conv1d_update: conv_state must be (num_slots, dim, state_size)");
    TORCH_CHECK(out.dim() == 3 && out.size(0) == bsz && out.size(1) == seqlen && out.size(2) == dim,
                "causal_conv1d_update: out must be (bsz, seqlen, dim)");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && conv_state.scalar_type() == at::kBFloat16 &&
                weight.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
                "causal_conv1d_update: x/state/weight/out must be bf16");
    TORCH_CHECK(!bias.has_value() || (bias->scalar_type() == at::kBFloat16 && bias->numel() == dim),
                "causal_conv1d_update: bias must be a bf16 (dim) tensor");
    if (!bsz || !dim || !seqlen) return;

    const int* slots_ptr = slots.has_value() ? (const int*) slots->data_ptr() : nullptr;
    const bf16* bias_ptr = bias.has_value() ? (const bf16*) bias->data_ptr() : nullptr;

    sycl::queue& q = exl3_sycl_queue();
    #define CONV_LAUNCH(ACT_, HIST_, INC_)                                                   \
        launch_conv1d_update<ACT_, HIST_, INC_>(q, (const bf16*) x.data_ptr(),                \
            (bf16*) conv_state.data_ptr(), slots_ptr, (const bf16*) weight.data_ptr(),       \
            bias_ptr, (bf16*) out.data_ptr(), bsz, dim, seqlen, state_size, K)

    if (activation)
    {
        if (incremental)     CONV_LAUNCH(true, true, true);
        else if (history)    CONV_LAUNCH(true, true, false);
        else                 CONV_LAUNCH(true, false, false);
    }
    else
    {
        if (incremental)     CONV_LAUNCH(false, true, true);
        else if (history)    CONV_LAUNCH(false, true, false);
        else                 CONV_LAUNCH(false, false, false);
    }
    #undef CONV_LAUNCH
}

void sycl_recurrent_gated_delta_rule(
    const at::Tensor& mixed_qkv, const at::Tensor& g, const at::Tensor& beta,
    at::Tensor& recurrent_state, at::Tensor& core_attn_out,
    int64_t num_k_heads, int64_t num_v_heads, int64_t k_head_dim, int64_t v_head_dim,
    const c10::optional<at::Tensor>& slots, bool history,
    const c10::optional<at::Tensor>& slots_in)
{
    const int bsz = (int) mixed_qkv.size(0);
    const int seqlen = (int) mixed_qkv.size(1);

    TORCH_CHECK(num_k_heads > 0, "gdn scan: num_k_heads must be positive");
    TORCH_CHECK(num_v_heads % num_k_heads == 0, "gdn scan: num_v_heads must be divisible by num_k_heads");
    TORCH_CHECK(k_head_dim == 128 && v_head_dim == 128,
                "gdn scan (SYCL): only 128x128 head dims are supported");
    TORCH_CHECK(mixed_qkv.scalar_type() == at::kBFloat16, "gdn scan: mixed_qkv must be bf16");
    TORCH_CHECK(beta.scalar_type() == at::kBFloat16, "gdn scan: beta must be bf16");
    TORCH_CHECK(recurrent_state.scalar_type() == at::kFloat, "gdn scan: state must be fp32");
    TORCH_CHECK(core_attn_out.scalar_type() == at::kBFloat16, "gdn scan: core_attn_out must be bf16");

    const bool channelwise = g.dim() == 4;
    if (channelwise)
    {
        TORCH_CHECK(g.size(0) == bsz && g.size(1) == seqlen && g.size(2) == num_v_heads &&
                    g.size(3) == k_head_dim, "gdn scan: channelwise g must be [b, s, heads, 128]");
    }
    else
    {
        TORCH_CHECK(g.dim() == 3 && g.size(0) == bsz && g.size(1) == seqlen && g.size(2) == num_v_heads,
                    "gdn scan: g must be [b, s, heads]");
    }

    TORCH_CHECK(mixed_qkv.is_contiguous() && g.is_contiguous() && beta.is_contiguous() &&
                recurrent_state.is_contiguous() && core_attn_out.is_contiguous(),
                "gdn scan: tensors must be contiguous");

    // EXL3_GDN_VSPLIT overrides the split for A/B testing (1/2/4). The default is 1: with the
    // full v head in one group every item does state work (V_SPLIT=4 leaves 128 of 512 items
    // active), measured 1.16x on the decode scan (79.8 vs 92.8 us, 48x128x128 state) and
    // bit-exact (scan_ab.py)
    const char* vs_env = getenv("EXL3_GDN_VSPLIT");
    const int v_split = vs_env ? atoi(vs_env) : 1;
    TORCH_CHECK(v_split == 1 || v_split == 2 || v_split == 4,
                "gdn scan: EXL3_GDN_VSPLIT must be 1, 2 or 4");
    TORCH_CHECK(v_head_dim % v_split == 0, "gdn scan: v_head_dim must be divisible by v_split");

    if (bsz == 0 || seqlen == 0 || num_v_heads == 0) return;

    sycl::queue& q = exl3_sycl_queue();
    const int history_stride = (int) recurrent_state.size(1);

    const int* slots_ptr = slots.has_value() ? (const int*) slots->data_ptr() : nullptr;
    const int* slots_in_ptr = history && slots_in.has_value() ? (const int*) slots_in->data_ptr() : nullptr;

    #define GDN_LAUNCH(VS, CW) \
        launch_gdn_scan<VS, CW>(q, \
            (const bf16*) mixed_qkv.data_ptr(), (const float*) g.data_ptr(), \
            (const bf16*) beta.data_ptr(), (float*) recurrent_state.data_ptr(), \
            (bf16*) core_attn_out.data_ptr(), bsz, seqlen, (int) num_k_heads, (int) num_v_heads, \
            slots_ptr, history_stride, slots_in_ptr)

    if (channelwise)
    {
        if (v_split == 4) GDN_LAUNCH(4, true); else if (v_split == 2) GDN_LAUNCH(2, true); else GDN_LAUNCH(1, true);
    }
    else
    {
        if (v_split == 4) GDN_LAUNCH(4, false); else if (v_split == 2) GDN_LAUNCH(2, false); else GDN_LAUNCH(1, false);
    }
    #undef GDN_LAUNCH
}

// -------------------------------------------------------------------------------------------------
// Speculative-decoding rewind, ports of gdn.cu's batched rewind kernels.
//
// batched_conv_rewind: conv_state[slot][:, :cdim] <- conv_state[slot][:, p-cdim:p] for every job,
// one work-item per channel, reading its (up to CONV1D_MAX_K) elements into registers before
// writing any of them back so overlapping windows (num_tokens < conv_kernel_size) are safe
// without synchronization. All jobs go through a single launch via a device-side job table.
//
// batched_scan_replay: rerun the sequential scan over the first `prefix` staged tokens of one
// batch row of the last speculative pass, from the base state row into the scratch row — the
// same kernel the pass ran (launch_gdn_scan) over the same inputs and initial state, so the
// result is the state that pass had after `prefix` tokens. One launch per job (a rewind is one
// call carrying every layer's job; the scans are independent — disjoint state rows and inputs).
// -------------------------------------------------------------------------------------------------

#define GDN_CONV_MAX_K 16
#define GDN_REWIND_CONV_THREADS 256

class GdnConvRewindKernelName;

void sycl_batched_conv_rewind(std::vector<ConvRewindJob> const& jobs, int64_t device_index)
{
    if (jobs.empty()) return;
    (void) device_index;

    sycl::queue& q = exl3_sycl_queue();

    // Device-side job table, grown on demand (a rewind touches every GDN layer of the model,
    // so the same buffer is reused on every call; in-order queue orders copy -> kernel)
    static ConvRewindJob* tbl = nullptr;
    static size_t cap = 0;
    if (jobs.size() > cap)
    {
        if (tbl) { q.wait(); sycl::free(tbl, q); }
        cap = jobs.size();
        tbl = sycl::malloc_device<ConvRewindJob>(cap, q);
    }
    q.memcpy(tbl, jobs.data(), jobs.size() * sizeof(ConvRewindJob)).wait();

    size_t max_dim = 0;
    for (auto& j : jobs)
    {
        TORCH_CHECK(j.cdim <= GDN_CONV_MAX_K, "batched_conv_rewind: cdim exceeds GDN_CONV_MAX_K");
        max_dim = std::max(max_dim, (size_t) j.dim);
    }
    const size_t nblk = (max_dim + GDN_REWIND_CONV_THREADS - 1) / GDN_REWIND_CONV_THREADS;
    ConvRewindJob* tbl_dev = tbl;   // static data can't be captured into a SYCL kernel

    q.submit([&](sycl::handler& h)
    {
        h.parallel_for<GdnConvRewindKernelName>(
            sycl::nd_range<2>(sycl::range<2>(nblk * GDN_REWIND_CONV_THREADS, jobs.size()),
                              sycl::range<2>(GDN_REWIND_CONV_THREADS, 1)),
            [=](sycl::nd_item<2> it)
        {
            const int d = (int) it.get_local_id(0) + (int) it.get_group(0) * GDN_REWIND_CONV_THREADS;
            const int ji = (int) it.get_group(1);
            if (d >= tbl_dev[ji].dim) return;

            const bf16* s = (const bf16*) tbl_dev[ji].src + (size_t) d * tbl_dev[ji].stride;
            bf16* t = (bf16*) tbl_dev[ji].dst + (size_t) d * tbl_dev[ji].stride;

            bf16 reg[GDN_CONV_MAX_K];
            #pragma unroll
            for (int k = 0; k < GDN_CONV_MAX_K; ++k)
                if (k < tbl_dev[ji].cdim) reg[k] = s[k];
            #pragma unroll
            for (int k = 0; k < GDN_CONV_MAX_K; ++k)
                if (k < tbl_dev[ji].cdim) t[k] = reg[k];
        });
    });
}

void sycl_batched_scan_replay(std::vector<ScanReplayJob> const& jobs, int64_t device_index,
                              int64_t kind, int64_t num_k_heads, int64_t num_v_heads,
                              int64_t k_head_dim, int64_t v_head_dim)
{
    if (jobs.empty()) return;
    (void) device_index;

    TORCH_CHECK(kind != 1, "batched_scan_replay (SYCL): Mamba2 scan not ported");
    TORCH_CHECK(num_k_heads > 0 && num_v_heads % num_k_heads == 0, "batched_scan_replay: bad head counts");
    TORCH_CHECK(k_head_dim == 128 && v_head_dim == 128, "batched_scan_replay (SYCL): only 128x128 head dims");

    sycl::queue& q = exl3_sycl_queue();
    const bool channelwise = (kind == 2);
    const int v_split = (num_v_heads <= 64) ? 4 : 1;

    for (auto const& j : jobs)
    {
        TORCH_CHECK(j.prefix >= 1, "batched_scan_replay: prefix must be at least 1");
        TORCH_CHECK(j.tstride >= 1, "batched_scan_replay: tstride must be at least 1");
        TORCH_CHECK((const void*) j.base != (const void*) j.scratch,
                    "batched_scan_replay: base and scratch rows must differ");

        // bsz = 1 (one job = one batch row), seqlen = prefix, explicit base/scratch state rows,
        // no attention output. The recurrent_state / history_stride args are unused in replay
        // mode (the state rows are the job's absolute pointers)
        #define GDN_REPLAY_LAUNCH(VS, CW) \
            launch_gdn_scan<VS, CW>(q, \
                (const bf16*) j.qkv, (const float*) j.g, (const bf16*) j.beta, \
                (float*) j.base, nullptr, 1, j.prefix, (int) num_k_heads, (int) num_v_heads, \
                nullptr, 0, nullptr, j.tstride, (const float*) j.base, (float*) j.scratch)

        if (channelwise)
        {
            if (v_split == 4) GDN_REPLAY_LAUNCH(4, true); else GDN_REPLAY_LAUNCH(1, true);
        }
        else
        {
            if (v_split == 4) GDN_REPLAY_LAUNCH(4, false); else GDN_REPLAY_LAUNCH(1, false);
        }
        #undef GDN_REPLAY_LAUNCH
    }
}
