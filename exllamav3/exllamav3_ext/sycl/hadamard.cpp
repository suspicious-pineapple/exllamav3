// had_r_128, port of quant/hadamard.cu + hadamard_inner.cuh into SYCL.
//
// For each 128-chunk of the last dim: y = ((x . pre_scale) . H / sqrt(128)) . post_scale, the
// pre-scale applied before H and the post-scale after. One work-group of 128 items per chunk, the
// butterfly running in fp32 through a small local buffer. The CUDA kernel converts to fp32 after
// the (fp16) pre-scale and rounds once at the end; this port applies the pre/post scales in fp32,
// which differs only in that rounding (noted in README.md).

#include <sycl/sycl.hpp>
#include <ATen/Tensor.h>

#include "exl3_compat.h"

namespace {

constexpr float HAD_R_SCALE = 0.08838834764831845f;   // 1/sqrt(128)

template <bool PRE, bool POST>
class HadKernelName;

template <bool PRE, bool POST>
void launch_had(sycl::queue& q, const sycl::half* input, sycl::half* output,
                const sycl::half* scale, float r_scale, int rows, int chunks, int stride = 0)
{
    const int row_stride = stride > 0 ? stride : chunks * 128;
    // One work-group of 128 items per (row, 128-chunk); the butterfly is a local-memory buffer
    q.submit([&](sycl::handler& cgh)
    {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(128), cgh);
        cgh.parallel_for<HadKernelName<PRE, POST>>(
            sycl::nd_range<1>(sycl::range<1>((size_t) rows * chunks * 128), sycl::range<1>(128)),
            [=](sycl::nd_item<1> it)
        {
            const int chunk = (int) (it.get_group_linear_id() % chunks);
            const int row = (int) (it.get_group_linear_id() / chunks);
            const int p = (int) it.get_local_id(0);
            const size_t base = (size_t) row * row_stride + (size_t) chunk * 128;

            float v = (float) input[base + p];
            if constexpr (PRE)
                v = v * (float) scale[chunk * 128 + p];
            buf[p] = v;
            sycl::group_barrier(it.get_group());

            // 7-stage natural-order Sylvester butterfly over the 128 positions. Sign convention
            // follows the CUDA shuffle_had: out[set-bit lane] = partner - self
            for (int s = 0; s < 7; ++s)
            {
                float pv = buf[p ^ (1 << s)];
                sycl::group_barrier(it.get_group());
                buf[p] = (((p >> s) & 1) ? (pv - buf[p]) : (buf[p] + pv));
                sycl::group_barrier(it.get_group());
            }

            float out = buf[p] * r_scale;
            if constexpr (POST)
                out = out * (float) scale[chunk * 128 + p];
            output[base + p] = sycl::half(out);
        });
    });
}

} // namespace

void sycl_had_r_128_h(const at::Tensor& input, const at::Tensor& output,
                      const c10::optional<at::Tensor>& pre_scale,
                      const c10::optional<at::Tensor>& post_scale, float scale)
{
    TORCH_CHECK(input.dim() == 2, "had_r_128: input must be 2D");
    TORCH_CHECK(input.sizes() == output.sizes(), "had_r_128: input/output shape mismatch");
    TORCH_CHECK(input.size(1) % 128 == 0, "had_r_128: last dim must be divisible by 128");
    TORCH_CHECK(input.scalar_type() == at::kHalf && output.scalar_type() == at::kHalf,
                "had_r_128: half in/out supported");

    const int rows = (int) input.size(0);
    const int chunks = (int) input.size(1) / 128;
    const float r_scale = scale * HAD_R_SCALE;

    if (!rows || !chunks) return;

    sycl::queue& q = exl3_sycl_queue();
    const sycl::half* in_p = (const sycl::half*) input.data_ptr();
    sycl::half* out_p = (sycl::half*) output.data_ptr();
    const sycl::half* scale_p = nullptr;
    if (pre_scale.has_value()) scale_p = (const sycl::half*) pre_scale->data_ptr();
    if (post_scale.has_value()) scale_p = (const sycl::half*) post_scale->data_ptr();

    if (pre_scale.has_value())
        launch_had<true, false>(q, in_p, out_p, scale_p, r_scale, rows, chunks);
    else if (post_scale.has_value())
        launch_had<false, true>(q, in_p, out_p, scale_p, r_scale, rows, chunks);
    else
        launch_had<false, false>(q, in_p, out_p, scale_p, r_scale, rows, chunks);
}

// fp32 in/out variant (used for the fp32-output GEMV decode path's output Hadamard)
template <bool PRE, bool POST>
class HadFFKernelName;

template <bool PRE, bool POST>
static void launch_had_ff(sycl::queue& q, const float* input, float* output,
                          const sycl::half* scale, float r_scale, int rows, int chunks, int stride = 0)
{
    const int row_stride = stride > 0 ? stride : chunks * 128;
    q.submit([&](sycl::handler& cgh)
    {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(128), cgh);
        cgh.parallel_for<HadFFKernelName<PRE, POST>>(
            sycl::nd_range<1>(sycl::range<1>((size_t) rows * chunks * 128), sycl::range<1>(128)),
            [=](sycl::nd_item<1> it)
        {
            const int chunk = (int) (it.get_group_linear_id() % chunks);
            const int row = (int) (it.get_group_linear_id() / chunks);
            const int p = (int) it.get_local_id(0);
            const size_t base = (size_t) row * row_stride + (size_t) chunk * 128;

            float v = input[base + p];
            if constexpr (PRE)
                v = v * (float) scale[chunk * 128 + p];
            buf[p] = v;
            sycl::group_barrier(it.get_group());

            for (int s = 0; s < 7; ++s)
            {
                float pv = buf[p ^ (1 << s)];
                sycl::group_barrier(it.get_group());
                buf[p] = (((p >> s) & 1) ? (pv - buf[p]) : (buf[p] + pv));
                sycl::group_barrier(it.get_group());
            }

            float out = buf[p] * r_scale;
            if constexpr (POST)
                out = out * (float) scale[chunk * 128 + p];
            output[base + p] = out;
        });
    });
}

void sycl_had_r_128_f(const at::Tensor& input, const at::Tensor& output,
                      const c10::optional<at::Tensor>& pre_scale,
                      const c10::optional<at::Tensor>& post_scale, float scale)
{
    TORCH_CHECK(input.dim() == 2, "had_r_128: input must be 2D");
    TORCH_CHECK(input.sizes() == output.sizes(), "had_r_128: input/output shape mismatch");
    TORCH_CHECK(input.size(1) % 128 == 0, "had_r_128: last dim must be divisible by 128");
    TORCH_CHECK(input.scalar_type() == at::kFloat && output.scalar_type() == at::kFloat,
                "had_r_128: float in/out supported");

    const int rows = (int) input.size(0);
    const int chunks = (int) input.size(1) / 128;
    const float r_scale = scale * HAD_R_SCALE;

    if (!rows || !chunks) return;

    sycl::queue& q = exl3_sycl_queue();
    const float* in_p = (const float*) input.data_ptr();
    float* out_p = (float*) output.data_ptr();
    const sycl::half* scale_p = nullptr;
    if (pre_scale.has_value()) scale_p = (const sycl::half*) pre_scale->data_ptr();
    if (post_scale.has_value()) scale_p = (const sycl::half*) post_scale->data_ptr();

    if (pre_scale.has_value())
        launch_had_ff<true, false>(q, in_p, out_p, scale_p, r_scale, rows, chunks);
    else if (post_scale.has_value())
        launch_had_ff<false, true>(q, in_p, out_p, scale_p, r_scale, rows, chunks);
    else
        launch_had_ff<false, false>(q, in_p, out_p, scale_p, r_scale, rows, chunks);
}

// Raw-pointer entries for mgemm.cpp's pointer-table loop (pre and post are never both set in
// the GEMV flow: one call pre-scales the input, the other post-scales the output)
void sycl_had_h_raw(sycl::queue& q, const sycl::half* input, sycl::half* output,
                    const sycl::half* pre_scale, const sycl::half* post_scale, float scale,
                    int rows, int chunks, int stride = 0)
{
    const float r_scale = scale * HAD_R_SCALE;
    if (!rows || !chunks) return;
    if (pre_scale)
        launch_had<true, false>(q, input, output, pre_scale, r_scale, rows, chunks, stride);
    else if (post_scale)
        launch_had<false, true>(q, input, output, post_scale, r_scale, rows, chunks, stride);
    else
        launch_had<false, false>(q, input, output, nullptr, r_scale, rows, chunks, stride);
}

void sycl_had_f_raw(sycl::queue& q, const float* input, float* output,
                    const sycl::half* pre_scale, const sycl::half* post_scale, float scale,
                    int rows, int chunks, int stride = 0)
{
    const float r_scale = scale * HAD_R_SCALE;
    if (!rows || !chunks) return;
    if (pre_scale)
        launch_had_ff<true, false>(q, input, output, pre_scale, r_scale, rows, chunks, stride);
    else if (post_scale)
        launch_had_ff<false, true>(q, input, output, post_scale, r_scale, rows, chunks, stride);
    else
        launch_had_ff<false, false>(q, input, output, nullptr, r_scale, rows, chunks, stride);
}

// Dispatch on dtype, mirroring the CUDA had_r_128 surface
void sycl_had_r_128(const at::Tensor& input, const at::Tensor& output,
                    const c10::optional<at::Tensor>& pre_scale,
                    const c10::optional<at::Tensor>& post_scale, float scale)
{
    if (input.scalar_type() == at::kHalf)
        sycl_had_r_128_h(input, output, pre_scale, post_scale, scale);
    else if (input.scalar_type() == at::kFloat)
        sycl_had_r_128_f(input, output, pre_scale, post_scale, scale);
    else
        TORCH_CHECK(false, "had_r_128: unsupported datatype");
}
