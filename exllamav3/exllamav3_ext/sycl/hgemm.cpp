// Plain fp16 GEMM via oneMKL, port of hgemm.cu's cuBLAS path.
//
// Row-major C = A @ B computed with the column-major trick: gemm(transpose::N, transpose::N,
// m = n_cols, n = m_rows, k = k) over the column-major reinterpretations. oneMKL requires a/b/c
// to share a dtype, so the fp32-output case runs the gemm into a half scratch and upcasts.

#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <algorithm>

#include "exl3_compat.h"
#include "sycl_kernels.h"

namespace {

// Upcast a contiguous half buffer into y, respecting y's row stride
class UpcastKernelName;

void upcast_rows(sycl::queue& q, const sycl::half* src, int src_stride, float* dst, int dst_stride,
                 int width, int rows)
{
    q.parallel_for<UpcastKernelName>(sycl::range<2>((size_t) rows, (size_t) width),
        [=](sycl::item<2> item)
    {
        const int r = (int) item.get_id(0);
        const int c = (int) item.get_id(1);
        dst[(size_t) r * dst_stride + c] = (float) src[(size_t) r * src_stride + c];
    });
}

// ---------------------------------------------------------------- custom split-K dense GEMV (m == 1)
// oneMKL's GEMV is latency-bound on this backend (~240 GB/s at best on the 4B's dense shapes,
// measured on cold production-pattern buffers). This kernel: one work-group per (128-column tile,
// k-chunk), lane = column, a coalesced 256-byte W read per k, disjoint fp32 partial slices in a
// scratch buffer (no atomics, no memset), and a tiny reduce kernel summing the slices into C —
// safe because the production queue is in-order (partial -> reduce and consecutive calls). The
// chunk target is ~320 k-iterations (bench_dgemv.cpp, exact 4B shapes, cold round-robin:
// k=9216 n=2560 1.39x over oneMKL at 249 vs 179 GB/s, k=2560 n=9216 ~1.05x at 254 vs 241).
// m == 1 only (the callers gate on it); C's row-0 start is written, so C's row stride is irrelevant
template <int KS> class DgemvSkK;
template <int KS> class DgemvSkR;

static float* dgemv_scratch(sycl::queue& q, size_t n)
{
    static float* buf = nullptr;
    static size_t cap = 0;
    if (n > cap)
    {
        // in-flight kernels may still read the old buffer: drain before freeing
        if (buf) { q.wait(); sycl::free(buf, q); }
        cap = n;
        buf = sycl::malloc_device<float>(cap * 32, q);   // KS_MAX = 32 slices
    }
    return buf;
}

template <int KS>
static void dense_gemv_ks(sycl::queue& q, const sycl::half* x, const sycl::half* W, int w_ld,
                          void* C, int size_k, int size_n, bool out_fp32, float* scratch)
{
    const int ntiles = size_n / 128;
    const int chunk = (size_k + KS - 1) / KS;

    q.submit([&](sycl::handler& h)
    {
        h.parallel_for<DgemvSkK<KS>>(sycl::nd_range<1>(sycl::range<1>((size_t) ntiles * KS * 128),
                                                       sycl::range<1>(128)),
            [=](sycl::nd_item<1> it)
        {
            const int gid = (int) it.get_group_linear_id();
            const int tile = gid / KS;
            const int kc = gid % KS;
            const int lane = (int) it.get_local_id(0);
            const int col = tile * 128 + lane;
            const int k0 = kc * chunk;
            const int k1 = sycl::min(k0 + chunk, size_k);
            float acc = 0.0f;
            for (int k = k0; k < k1; ++k)
                acc = fmaf((float) x[k], (float) W[(size_t) k * w_ld + col], acc);
            scratch[(size_t) kc * size_n + col] = acc;
        });
    });
    q.submit([&](sycl::handler& h)
    {
        h.parallel_for<DgemvSkR<KS>>(sycl::range<1>((size_t) size_n), [=](sycl::item<1> item)
        {
            const int i = (int) item.get_linear_id();
            float s = 0.0f;
            #pragma unroll
            for (int c = 0; c < KS; ++c) s += scratch[(size_t) c * size_n + i];
            if (out_fp32) ((float*) C)[i] = s;
            else          ((sycl::half*) C)[i] = sycl::half(s);
        });
    });
}

void hgemm_impl(at::Tensor a, at::Tensor b, at::Tensor c)
{
    TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf,
                "hgemm: a and b must be float16");
    TORCH_CHECK(b.dim() == 2, "hgemm: b must be 2D");
    TORCH_CHECK(c.dim() >= 2, "hgemm: c must have at least 2 dimensions");
    TORCH_CHECK(a.size(-1) == b.size(0), "hgemm: a's last dim must match b's first dim");
    TORCH_CHECK(b.size(1) == c.size(-1), "hgemm: b's last dim must match c's last dim");
    TORCH_CHECK(c.stride(-1) == 1, "hgemm: c must have contiguous columns");

    const bool output_fp32 = c.scalar_type() == at::kFloat;
    TORCH_CHECK(output_fp32 || c.scalar_type() == at::kHalf, "hgemm: c must be float32 or float16");

    const int size_k = (int) a.size(-1);
    int64_t size_m_ = 1;
    for (int d = 0; d < a.dim() - 1; ++d) size_m_ *= a.size(d);
    const int size_m = (int) size_m_;
    const int size_n = (int) b.size(1);
    const int64_t c_stride_m = c.stride(-2);
    TORCH_CHECK(c_stride_m >= size_n, "hgemm: c row stride is too small");

    // No rows or no output columns: nothing to compute. Empty reduction (k = 0): the product is zero
    if (!size_m || !size_n) return;
    if (!size_k)
    {
        c.zero_();
        return;
    }

    sycl::queue& q = exl3_sycl_queue();

    const sycl::half* a_ptr = (const sycl::half*) a.data_ptr();
    const sycl::half* b_ptr = (const sycl::half*) b.data_ptr();

    // m == 1 decode: the custom split-K dense GEMV, but ONLY for long-k shapes — measured on
    // cold production-pattern buffers (bench_dgemv.cpp), it beats oneMKL 1.39x on the down_proj
    // shape (k=9216 n=2560) but its 128-column-tile layout runs out of parallelism on small-n
    // shapes (n=1024 -> 8 tiles x KS groups), where it measured much slower than oneMKL and
    // regressed the 4B end to end. oneMKL is at ~240 GB/s on those — keep it there
    if (size_m == 1 && size_n % 128 == 0 && size_k >= 6144)
    {
        void* c0 = (void*) ((char*) c.data_ptr());
        sycl_dense_gemv_raw(q, a_ptr, b_ptr, size_n, c0, size_k, size_n, output_fp32);
        return;
    }

    // C^T = B^T . A^T via the column-major reinterpretation, matching the cuBLAS call:
    // C_cm (n x m) = B_cm (n x k) @ A_cm (k x m), all leading dims as in the row-major layouts
    const oneapi::mkl::transpose N = oneapi::mkl::transpose::N;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (output_fp32)
    {
        at::Tensor scratch = at::empty({size_m, size_n}, a.options().dtype(at::kHalf));
        SYCL_CHECK(oneapi::mkl::blas::column_major::gemm(
            q, N, N,
            size_n, size_m, size_k,
            alpha, b_ptr, size_n,
                   a_ptr, size_k,
            beta,  (sycl::half*) scratch.data_ptr(), size_n));
        upcast_rows(q, (const sycl::half*) scratch.data_ptr(), size_n,
                    (float*) c.data_ptr(), (int) c_stride_m, size_n, size_m);
    }
    else
    {
        SYCL_CHECK(oneapi::mkl::blas::column_major::gemm(
            q, N, N,
            size_n, size_m, size_k,
            alpha, b_ptr, size_n,
                   a_ptr, size_k,
            beta,  (sycl::half*) c.data_ptr(), (int) c_stride_m));
    }
}

} // namespace

void sycl_dense_gemv_raw(sycl::queue& q, const sycl::half* x, const sycl::half* W, int w_ld,
                         void* C, int size_k, int size_n, bool out_fp32)
{
    int ks = (size_k + 160) / 320;
    ks = std::max(1, std::min(32, ks));
    float* scratch = dgemv_scratch(q, (size_t) size_n);
    if      (ks <= 4)  dense_gemv_ks<4>(q, x, W, w_ld, C, size_k, size_n, out_fp32, scratch);
    else if (ks <= 8)  dense_gemv_ks<8>(q, x, W, w_ld, C, size_k, size_n, out_fp32, scratch);
    else if (ks <= 16) dense_gemv_ks<16>(q, x, W, w_ld, C, size_k, size_n, out_fp32, scratch);
    else               dense_gemv_ks<32>(q, x, W, w_ld, C, size_k, size_n, out_fp32, scratch);
}

void sycl_hgemm(at::Tensor a, at::Tensor b, at::Tensor c)
{
    hgemm_impl(a, b, c);
}

// Reconstruct-path GEMM (same call on SYCL)
void sycl_hgemm_recon(at::Tensor a, at::Tensor b, at::Tensor c)
{
    hgemm_impl(a, b, c);
}
