// Fused decode attention (flash-decode) for the SYCL backend: one kernel per layer call replacing
// the eager GQA decode path's ~8 torch kernel launches per layer per token (batched matmul -> fp32
// cast -> scale -> softmax -> cast -> matmul, all on tiny tensors whose cost is launch overhead).
//
// Online softmax over kv chunks; scores and accumulators stay in fp32 (the eager path rounds the
// score matmul to fp16 first — a precision-only difference, verified against it).
//
// Layout: q (1, Hq, hd) fp16, k/v (kv, Hkv, hd) fp16 contiguous window (the caller guarantees the
// rows are the sequence's dense window with the new tokens already appended), out (1, Hq, hd) fp16.
// GQA: q-head h reads kv head h / (Hq / Hkv). One work-group of Lanes lanes per q-head; each lane
// scores one kv position per chunk (hd-dot serially) and owns hd/Lanes output dims (hd == 256).

#include <sycl/sycl.hpp>
#include <ATen/Tensor.h>
#include <cmath>

#include "exl3_compat.h"
#include "sycl_kernels.h"

namespace {

template <int Lanes>
class FlashDecodeKernelName;

template <int Lanes>
void flash_decode_impl(sycl::queue& q, const sycl::half* qp, const sycl::half* kp,
                       const sycl::half* vp, sycl::half* op,
                       int Hq, int Hkv, int hd, int kv_len, float scale)
{
    const int grp = Hq / Hkv;
    const int stride_kv = Hkv * hd;   // elements per kv position
    constexpr int DPL = 256 / Lanes;  // output dims per lane (hd is guarded to 256 by the caller)

    q.submit([&](sycl::handler& h)
    {
        sycl::local_accessor<float, 1> lsc(sycl::range<1>(Lanes), h);   // chunk scores
        sycl::local_accessor<float, 1> les(sycl::range<1>(Lanes), h);   // chunk exp(score - m)
        sycl::local_accessor<float, 1> lred(sycl::range<1>(Lanes), h);  // reduction scratch
        h.parallel_for<FlashDecodeKernelName<Lanes>>(
            sycl::nd_range<1>(sycl::range<1>((size_t) Hq * Lanes), sycl::range<1>(Lanes)),
            [=](sycl::nd_item<1> it)
        {
            const int head = (int) it.get_group_linear_id();
            const int lane = (int) it.get_local_id(0);
            const int hk = head / grp;
            const sycl::half* qh = qp + (size_t) head * hd;
            const sycl::half* kbase = kp + (size_t) hk * hd;
            const sycl::half* vbase = vp + (size_t) hk * hd;

            float acc[DPL];
            #pragma unroll
            for (int j = 0; j < DPL; ++j) acc[j] = 0.0f;
            float m = -INFINITY;
            float l = 0.0f;

            for (int c0 = 0; c0 < kv_len; c0 += Lanes)
            {
                // this lane's position score (full-headdim dot, serial over d)
                float score = -INFINITY;
                const int p = c0 + lane;
                if (p < kv_len)
                {
                    const sycl::half* krow = kbase + (size_t) p * stride_kv;
                    float dot = 0.0f;
                    for (int d = 0; d < hd; ++d)
                        dot = fmaf((float) qh[d], (float) krow[d], dot);
                    score = dot * scale;
                }
                lsc[lane] = score;
                // the tree's init write must be fenced by the SAME barrier that precedes the
                // first cross-lane read (a race here intermittently corrupts the chunk max)
                lred[lane] = score;
                it.barrier(sycl::access::fence_space::local_space);

                // chunk max (tree over lred)
                #pragma unroll
                for (int off = Lanes / 2; off > 0; off >>= 1)
                {
                    if (lane < off) lred[lane] = sycl::fmax(lred[lane], lred[lane + off]);
                    it.barrier(sycl::access::fence_space::local_space);
                }
                const float mc = lred[0];
                const float mn = sycl::fmax(m, mc);
                // first chunk: m = -inf, correction is exactly 0 (and -inf - -inf would be NaN)
                const float corr = (m == -INFINITY) ? 0.0f : sycl::exp(m - mn);

                // es per lane: kept in les for the accumulation pass, copied into lred for the sum
                const float es = (p < kv_len) ? sycl::exp(lsc[lane] - mn) : 0.0f;
                les[lane] = es;
                lred[lane] = es;
                it.barrier(sycl::access::fence_space::local_space);

                // chunk sum (tree over lred)
                #pragma unroll
                for (int off = Lanes / 2; off > 0; off >>= 1)
                {
                    if (lane < off) lred[lane] += lred[lane + off];
                    it.barrier(sycl::access::fence_space::local_space);
                }
                const float se = lred[0];

                l = l * corr + se;
                #pragma unroll
                for (int j = 0; j < DPL; ++j) acc[j] *= corr;

                // accumulate this chunk's contribution to this lane's output dims
                const int npos = sycl::min((int) Lanes, kv_len - c0);
                for (int j = 0; j < npos; ++j)
                {
                    const float pj = les[j];
                    const sycl::half* vrow = vbase + (size_t) (c0 + j) * stride_kv;
                    #pragma unroll
                    for (int d = 0; d < DPL; ++d)
                        acc[d] = fmaf(pj, (float) vrow[lane * DPL + d], acc[d]);
                }
                it.barrier(sycl::access::fence_space::local_space);
                m = mn;
            }

            const float inv_l = 1.0f / l;  // l >= 1 for kv_len >= 1 (the max lane contributes exp(0))
            sycl::half* orow = op + (size_t) head * hd;
            #pragma unroll
            for (int d = 0; d < DPL; ++d)
                orow[lane * DPL + d] = sycl::half(acc[d] * inv_l);
        });
    });
}

} // namespace

void sycl_flash_decode(const at::Tensor& q_, const at::Tensor& k_buf,
                       const at::Tensor& v_buf, int64_t kv_len, double scale, at::Tensor& out)
{
    TORCH_CHECK(q_.dim() == 3 && q_.size(0) == 1, "flash_decode: q must be (1, Hq, hd)");
    TORCH_CHECK(k_buf.dim() == 3 && v_buf.dim() == 3, "flash_decode: k/v must be (kv, Hkv, hd)");
    TORCH_CHECK(k_buf.size(-1) == q_.size(-1) && v_buf.size(-1) == q_.size(-1),
                "flash_decode: headdim mismatch");
    TORCH_CHECK(k_buf.size(1) == v_buf.size(1), "flash_decode: k/v head mismatch");
    TORCH_CHECK(q_.size(-1) == 256, "flash_decode: headdim 256 only");
    TORCH_CHECK(q_.scalar_type() == at::kHalf && k_buf.scalar_type() == at::kHalf &&
                v_buf.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
                "flash_decode: fp16 only");
    TORCH_CHECK(kv_len >= 1 && kv_len <= k_buf.size(0), "flash_decode: bad kv_len");

    const int Hq = (int) q_.size(1);
    const int Hkv = (int) k_buf.size(1);
    TORCH_CHECK(Hq % Hkv == 0, "flash_decode: Hq not divisible by Hkv");
    const int hd = (int) q_.size(-1);

    flash_decode_impl<128>(exl3_sycl_queue(),
        (const sycl::half*) q_.data_ptr(), (const sycl::half*) k_buf.data_ptr(),
        (const sycl::half*) v_buf.data_ptr(), (sycl::half*) out.data_ptr(),
        Hq, Hkv, hd, (int) kv_len, (float) scale);
}
