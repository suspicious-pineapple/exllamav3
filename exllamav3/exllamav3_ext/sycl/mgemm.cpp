// Multi-linear GEMM (exl3_mgemm), SYCL port of the fused mgemm path.
//
// The CUDA kernel reads per-tensor trellis/scale addresses from a device pointer table and
// computes every tensor's output in one fused kernel. On SYCL this becomes a loop over the
// table: for each tensor, the raw-pointer GEMV flow (input Hadamard -> GEMV -> output
// Hadamard) writes its slice of C / A_had. The indices/weights (MoE) variant is not supported
// and refuses on the host; the sliced mode (equal-width column slices of fewer source
// matrices, used by the qkv projection path) runs one input Hadamard per source and a
// strided GEMV + output Hadamard per slice into the caller's output pointers.

#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <ATen/Tensor.h>

#include <mutex>
#include <unordered_map>

#include "exl3_compat.h"
#include "exl3_decode.h"
#include "sycl_kernels.h"

// Host copies of the device pointer tables (trellis/suh/svh and the sliced-mode int lists). The
// tables are written once at model load and never change, so the D2H copy (a blocking
// wait_and_throw that drains the whole in-order queue) happens once per table, not once per call.
// Keyed by the table tensor's data_ptr and length: the address identifies the load-time
// allocation, and a different length cannot share an allocation.
namespace {

struct PtrTableCache
{
    std::mutex mu;
    std::unordered_map<uintptr_t, std::vector<uint64_t>> u64;
    std::unordered_map<uintptr_t, std::vector<int>> i32;
};

PtrTableCache& ptr_table_cache()
{
    static PtrTableCache c;
    return c;
}

} // namespace

void exl3_sycl_ptr_cache_clear()
{
    // Drop every cached host copy of a device pointer table. The entries are keyed by the
    // table tensor's allocation address, which the caching allocator recycles across model
    // loads: a table of the new model can land on an address a table of the unloaded model
    // used with the same length, and the stale copy would silently feed the old model's
    // pointers (freed or reused memory) into the kernel. Called when a model is unloaded or
    // before a new one loads
    PtrTableCache& c = ptr_table_cache();
    std::lock_guard<std::mutex> g(c.mu);
    c.u64.clear();
    c.i32.clear();
}

void sycl_exl3_mgemm(const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
                     const at::Tensor& suh, const at::Tensor& A_had, const at::Tensor& svh,
                     const c10::optional<at::Tensor>& indices,
                     const c10::optional<at::Tensor>& weights,
                     float K, int64_t force_shape_idx, bool mcg, bool mul1,
                     int64_t min_index, int64_t max_index, int64_t force_num_sms,
                     int64_t num_tokens,
                     const c10::optional<at::Tensor>& size_n_list,
                     const c10::optional<at::Tensor>& c_ptrs,
                     const c10::optional<at::Tensor>& n_stride_list,
                     const c10::optional<at::Tensor>& had_src_list,
                     int64_t num_had_src, bool dense)
{
    sycl_exl3_mgemm_impl(A, B, C, suh, A_had, svh, indices, weights, K, force_shape_idx, mcg, mul1,
                         min_index, max_index, force_num_sms, num_tokens, size_n_list, c_ptrs,
                         n_stride_list, had_src_list, num_had_src, dense);
}

void sycl_exl3_mgemm_impl(const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
                      const at::Tensor& suh, const at::Tensor& A_had, const at::Tensor& svh,
                      const c10::optional<at::Tensor>& indices,
                      const c10::optional<at::Tensor>& weights,
                      float K, int64_t force_shape_idx, bool mcg, bool mul1,
                      int64_t min_index, int64_t max_index, int64_t force_num_sms,
                      int64_t num_tokens,
                      const c10::optional<at::Tensor>& size_n_list,
                      const c10::optional<at::Tensor>& c_ptrs,
                      const c10::optional<at::Tensor>& n_stride_list,
                      const c10::optional<at::Tensor>& had_src_list,
                      int64_t num_had_src, bool dense)
{
    (void) force_shape_idx;
    (void) force_num_sms;
    (void) num_tokens;
    TORCH_CHECK(!indices.has_value() && !weights.has_value(),
                "exl3_mgemm (SYCL): the indices/weights (MoE) variant is not supported");
    TORCH_CHECK(min_index < 0 && max_index < 0,
                "exl3_mgemm (SYCL): min_index/max_index require the MoE variant");

    const bool sliced = had_src_list.has_value();
    TORCH_CHECK(sliced == size_n_list.has_value() && sliced == c_ptrs.has_value() &&
                sliced == n_stride_list.has_value(),
                "exl3_mgemm: size_n_list/c_ptrs/n_stride_list require had_src_list and vice versa");

    // B/suh/svh are device pointer tables (int64 tensor of raw data_ptr()s, two's-complement
    // for XPU pointers above 2^63; the bit pattern is preserved by the uint64 reads below)
    TORCH_CHECK(B.dim() == 1 && B.scalar_type() == at::kLong, "exl3_mgemm: B must be a 1D pointer table");
    TORCH_CHECK(suh.dim() == 1 && suh.scalar_type() == at::kLong, "exl3_mgemm: suh must be a 1D pointer table");
    TORCH_CHECK(svh.dim() == 1 && svh.scalar_type() == at::kLong, "exl3_mgemm: svh must be a 1D pointer table");
    TORCH_CHECK(B.size(0) == svh.size(0), "exl3_mgemm: table length mismatch");
    TORCH_CHECK(sliced ? suh.size(0) >= num_had_src : suh.size(0) == B.size(0),
                "exl3_mgemm: suh table length mismatch");

    const int64_t num_tensors = B.size(0);
    TORCH_CHECK(C.dim() == 3 && C.size(0) == num_tensors, "exl3_mgemm: C must be [num_tensors, m, n]");
    TORCH_CHECK(A.scalar_type() == at::kHalf, "exl3_mgemm: A must be float16");

    const int size_k = (int) A.size(-1);
    const int size_n = (int) C.size(-1);
    int size_m = 1;
    for (int d = 1; d < A.dim() - 1; ++d) size_m *= (int) A.size(d);
    TORCH_CHECK(size_m == (int) C.size(1), "exl3_mgemm: m mismatch between A and C");
    TORCH_CHECK(size_k == (int) A_had.size(-1), "exl3_mgemm: k mismatch between A and A_had");
    TORCH_CHECK(size_k % 128 == 0, "exl3_mgemm: k must be divisible by 128");
    if (sliced)
    {
        TORCH_CHECK(A.dim() == 3 && A.size(0) == 1, "exl3_mgemm: sliced mode is single-input");
        TORCH_CHECK(num_had_src > 0 && A_had.size(0) >= num_had_src,
                    "exl3_mgemm: sliced mode needs one A_had slab per source");
        TORCH_CHECK(size_n % 128 == 0, "exl3_mgemm: slice width must be divisible by 128");
    }
    else
    {
        TORCH_CHECK(A_had.dim() == 3 && A_had.size(0) == num_tensors,
                    "exl3_mgemm: A_had must be [num_tensors, m, k]");
        TORCH_CHECK(size_n % 128 == 0, "exl3_mgemm: dims must be divisible by 128");
    }

    const BitsK bk = bits_from_K(K);
    TORCH_CHECK(!bk.half || mul1, "exl3_mgemm: half-integer bitrates require the mul1 codebook");

    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    if (!num_tensors || !size_m || !size_n || !size_k) return;

    // Read the pointer tables to the host. They are written once at model load and never change,
    // so each table is copied (a blocking D2H wait that drains the queue) only on first use and
    // served from the static cache afterwards
    sycl::queue& q = exl3_sycl_queue();
    auto copy_ptrs = [&](const at::Tensor& t) -> const std::vector<uint64_t>&
    {
        const uintptr_t key = (uintptr_t) t.data_ptr();
        PtrTableCache& c = ptr_table_cache();
        std::lock_guard<std::mutex> g(c.mu);
        auto it = c.u64.find(key);
        if (it == c.u64.end() || it->second.size() != (size_t) t.size(0))
        {
            std::vector<uint64_t> host((size_t) t.size(0));
            q.memcpy(host.data(), t.data_ptr(), t.size(0) * sizeof(uint64_t)).wait_and_throw();
            return c.u64[key] = std::move(host);
        }
        return it->second;
    };
    auto copy_ints = [&](const at::Tensor& t) -> const std::vector<int>&
    {
        const uintptr_t key = (uintptr_t) t.data_ptr();
        PtrTableCache& c = ptr_table_cache();
        std::lock_guard<std::mutex> g(c.mu);
        auto it = c.i32.find(key);
        if (it == c.i32.end() || it->second.size() != (size_t) t.numel())
        {
            std::vector<int> host((size_t) t.numel());
            q.memcpy(host.data(), t.data_ptr(), t.numel() * sizeof(int)).wait_and_throw();
            return c.i32[key] = std::move(host);
        }
        return it->second;
    };
    const std::vector<uint64_t>& trellis_ptrs = copy_ptrs(B);
    const std::vector<uint64_t>& suh_ptrs = copy_ptrs(suh);
    const std::vector<uint64_t>& svh_ptrs = copy_ptrs(svh);

    const sycl::half* A_ptr = (const sycl::half*) A.data_ptr();
    const int m_stride = size_m * size_n;
    const int k_stride = size_m * size_k;
    const bool c_fp32 = C.scalar_type() == at::kFloat;
    if (!c_fp32) TORCH_CHECK(C.scalar_type() == at::kHalf, "exl3_mgemm: C must be float16 or float32");

    // Dense mode: the pointer tables hold dense fp16 weights ([k, n] row-major per source,
    // pre-dequantized from the trellis at load) instead of packed trellis. The decode GEMV
    // kernel is compute/latency-bound on this backend (~1.1 ms per 2048x2048), while oneMKL
    // runs the same GEMM in ~0.05 ms, so the flow matches the packed GEMV's: input Hadamard
    // per source (pre = suh) -> oneMKL gemm per tensor/slice -> output Hadamard in place
    // (post = svh). The dense weights are the rotated-basis (raw) V, so both-side Hadamards
    // yield the same result the packed path computes.
    if (dense)
    {
        const oneapi::mkl::transpose N = oneapi::mkl::transpose::N;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // fp32 out is only served by the non-sliced branch (hadamard in fp32, as the packed
        // path does); the sliced fp32-out bundle stays on the packed path (gated at the call site)
        TORCH_CHECK(!dense || !c_fp32 || !sliced, "exl3_mgemm: dense sliced mode requires fp16 output");

        if (sliced)
        {
            const std::vector<uint64_t>& out_ptrs = copy_ptrs(*c_ptrs);
            const std::vector<int>& n_list = copy_ints(*size_n_list);
            const std::vector<int>& stride_list = copy_ints(*n_stride_list);
            const std::vector<int>& src_list = copy_ints(*had_src_list);
            TORCH_CHECK(n_list.size() >= (size_t) num_tensors && stride_list.size() >= (size_t) num_tensors &&
                        src_list.size() >= (size_t) num_tensors && out_ptrs.size() >= (size_t) num_tensors,
                        "exl3_mgemm: sliced mode needs one entry per slice");

            for (int64_t s = 0; s < num_had_src; ++s)
            {
                TORCH_CHECK(src_list.size() > 0, "exl3_mgemm: empty sliced dispatch");
                sycl_had_h_raw(q, A_ptr, (sycl::half*) A_had.data_ptr() + (size_t) s * k_stride,
                               (const sycl::half*) (uintptr_t) suh_ptrs[s], nullptr, 1.0f,
                               size_m, size_k / 128);
            }
            for (int64_t j = 0; j < num_tensors; ++j)
            {
                const int src = src_list[j];
                const int n_j = n_list[j];
                const int stride_j = stride_list[j];
                TORCH_CHECK(src >= 0 && src < num_had_src, "exl3_mgemm: had_src_list out of range");
                TORCH_CHECK(n_j > 0 && n_j % 128 == 0 && stride_j >= n_j,
                            "exl3_mgemm: invalid slice width/stride");
                // C = A_had_src @ W_j. The dense W slice is a strided submatrix of the source's
                // [in, out] tensor, so its row stride is the source's full width. The custom
                // dense GEMV only for long-k shapes (see hgemm.cpp: it runs out of parallelism
                // on small-n shapes and regressed end-to-end there)
                if (size_m == 1 && size_k >= 6144)
                    sycl_dense_gemv_raw(q, (const sycl::half*) A_had.data_ptr() + (size_t) src * k_stride,
                                        (const sycl::half*) (uintptr_t) trellis_ptrs[j], stride_j,
                                        (sycl::half*) (uintptr_t) out_ptrs[j], size_k, n_j, false);
                else
                    SYCL_CHECK(oneapi::mkl::blas::column_major::gemm(
                        q, N, N,
                        n_j, size_m, size_k,
                        alpha, (const sycl::half*) (uintptr_t) trellis_ptrs[j], stride_j,
                               (const sycl::half*) A_had.data_ptr() + (size_t) src * k_stride, size_k,
                        beta,  (sycl::half*) (uintptr_t) out_ptrs[j], stride_j));
                // Output Hadamard in place on the slice (post = svh at the slice's column offset)
                sycl_had_h_raw(q, (sycl::half*) (uintptr_t) out_ptrs[j],
                               (sycl::half*) (uintptr_t) out_ptrs[j], nullptr,
                               (const sycl::half*) (uintptr_t) svh_ptrs[j], 1.0f,
                               size_m, n_j / 128, stride_j);
            }
            return;
        }

        for (int64_t i = 0; i < num_tensors; ++i)
        {
            // Input Hadamard into this tensor's A_had slab (pre = suh_i)
            sycl_had_h_raw(q, A_ptr, (sycl::half*) A_had.data_ptr() + (size_t) i * k_stride,
                           (const sycl::half*) (uintptr_t) suh_ptrs[i], nullptr, 1.0f,
                           size_m, size_k / 128);
            void* c_i = (void*) ((char*) C.data_ptr() + (size_t) i * m_stride * (c_fp32 ? 4 : 2));
            if (size_m == 1 && size_k >= 6144)
                sycl_dense_gemv_raw(q, (const sycl::half*) A_had.data_ptr() + (size_t) i * k_stride,
                                    (const sycl::half*) (uintptr_t) trellis_ptrs[i], size_n,
                                    c_i, size_k, size_n, c_fp32);
            else
                SYCL_CHECK(oneapi::mkl::blas::column_major::gemm(
                    q, N, N,
                    size_n, size_m, size_k,
                    alpha, (const sycl::half*) (uintptr_t) trellis_ptrs[i], size_n,
                           (const sycl::half*) A_had.data_ptr() + (size_t) i * k_stride, size_k,
                    beta,  (sycl::half*) c_i, size_n));
            // Output Hadamard in place on C_i (post = svh_i)
            if (c_fp32)
                sycl_had_f_raw(q, (float*) c_i, (float*) c_i, nullptr,
                               (const sycl::half*) (uintptr_t) svh_ptrs[i], 1.0f,
                               size_m, size_n / 128);
            else
                sycl_had_h_raw(q, (sycl::half*) c_i, (sycl::half*) c_i, nullptr,
                               (const sycl::half*) (uintptr_t) svh_ptrs[i], 1.0f,
                               size_m, size_n / 128);
        }
        return;
    }

    if (sliced)
    {
        const std::vector<uint64_t>& out_ptrs = copy_ptrs(*c_ptrs);
        const std::vector<int>& n_list = copy_ints(*size_n_list);
        const std::vector<int>& stride_list = copy_ints(*n_stride_list);
        const std::vector<int>& src_list = copy_ints(*had_src_list);
        TORCH_CHECK(n_list.size() >= (size_t) num_tensors && stride_list.size() >= (size_t) num_tensors &&
                    src_list.size() >= (size_t) num_tensors && out_ptrs.size() >= (size_t) num_tensors,
                    "exl3_mgemm: sliced mode needs one entry per slice");

        // Input Hadamard, once per source, into its A_had slab (suh is per source here)
        for (int64_t s = 0; s < num_had_src; ++s)
        {
            TORCH_CHECK(src_list.size() > 0, "exl3_mgemm: empty sliced dispatch");
            sycl_had_h_raw(q, A_ptr, (sycl::half*) A_had.data_ptr() + (size_t) s * k_stride,
                           (const sycl::half*) (uintptr_t) suh_ptrs[s], nullptr, 1.0f,
                           size_m, size_k / 128);
        }

        // Per slice: strided GEMV into the caller's output pointer, then the output Hadamard
        for (int64_t j = 0; j < num_tensors; ++j)
        {
            const int src = src_list[j];
            const int n_j = n_list[j];
            const int stride_j = stride_list[j];
            TORCH_CHECK(src >= 0 && src < num_had_src, "exl3_mgemm: had_src_list out of range");
            TORCH_CHECK(n_j > 0 && n_j % 128 == 0 && stride_j >= n_j,
                        "exl3_mgemm: invalid slice width/stride");
            sycl_exl3_gemm_raw2(q,
                (const sycl::half*) A_had.data_ptr() + (size_t) src * k_stride,
                (const uint16_t*) (uintptr_t) trellis_ptrs[j],
                (void*) (uintptr_t) out_ptrs[j],
                (const sycl::half*) (uintptr_t) svh_ptrs[j],
                size_m, size_k, n_j, stride_j, c_fp32,
                bk.bits, cb, bk.half);
        }
        return;
    }

    for (int64_t i = 0; i < num_tensors; ++i)
    {
        sycl_exl3_gemm_raw(q,
            A_ptr,
            (sycl::half*) A_had.data_ptr() + (size_t) i * k_stride,
            (const uint16_t*) (uintptr_t) trellis_ptrs[i],
            (char*) C.data_ptr() + (size_t) i * m_stride * (c_fp32 ? 4 : 2),
            (const sycl::half*) (uintptr_t) suh_ptrs[i],
            (const sycl::half*) (uintptr_t) svh_ptrs[i],
            size_m, size_k, size_n, c_fp32,
            bk.bits, cb, bk.half);
    }
}
