// Elementwise kernels, ports of norm.cu, add.cu, softcap.cu, activation.cu, plus the GDN fused
// ops from gdn.cu (see gdn.cpp for the recurrent scan). The fp32 math follows the CUDA kernels;
// where the CUDA chain computes in fp16, the SYCL port computes in fp32 and rounds once, which is
// bit-identical for the values involved (see exl3_compat.h).

#include <sycl/sycl.hpp>
#include <ATen/Tensor.h>
#include <cmath>
#include <type_traits>

#include "exl3_compat.h"
#include "sycl_kernels.h"

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

// -------------------------------------------------------------------------------------------------
// add: z = x + y, elementwise, any matching float dtypes
// -------------------------------------------------------------------------------------------------

template <typename T>
class AddKernelName;

template <typename T>
void launch_add(sycl::queue& q, const T* x, const T* y, T* z, size_t numel)
{
    q.parallel_for<AddKernelName<T>>(sycl::range<1>(numel), [=](sycl::item<1> item)
    {
        const size_t i = item.get_linear_id();
        z[i] = (T) ((float) x[i] + (float) y[i]);
    });
}

// -------------------------------------------------------------------------------------------------
// softcap: y = tanh(x / scale) * scale
// -------------------------------------------------------------------------------------------------

template <typename T>
class SoftcapKernelName;

template <typename T>
void launch_softcap(sycl::queue& q, const T* x, T* y, size_t numel, float scale)
{
    q.parallel_for<SoftcapKernelName<T>>(sycl::range<1>(numel), [=](sycl::item<1> item)
    {
        const size_t i = item.get_linear_id();
        float v = (float) x[i];
        v = sycl::tanh(v / scale) * scale;
        y[i] = (T) v;
    });
}

// -------------------------------------------------------------------------------------------------
// rms_norm: y = x * (w + constant_bias) / sqrt(row_mean(x^2) + epsilon) * constant_scale
// res modes: 0 = none, 1 = residual in (r += x rounded to r's dtype, feeds the norm), 2 = residual
// post (y = norm(x) + y). One work-group of 128 items per row; W = int means no weight.
// -------------------------------------------------------------------------------------------------

template <typename IN, typename OUT, typename W, typename RD, int LS>
class RmsNormKernelName;

// One work-group per row. The group size is picked by row width: a decode row is a single group on
// one EU, so a narrow group leaves the row latency-bound (40 sequential elements per item at
// LS=128) -- 512 items cut that to ~10 (measured 29us -> ~11us per 5120-wide decode norm).
template <typename IN, typename OUT, typename W, typename RD, int LS>
void launch_rms_norm(sycl::queue& q, const IN* x, const W* w, OUT* y, RD* r,
                     float epsilon, int rows, int dim, float constant_bias, float constant_scale,
                     int res)
{
    q.submit([&](sycl::handler& cgh)
    {
        sycl::local_accessor<float, 1> red(sycl::range<1>(LS), cgh);
        cgh.parallel_for<RmsNormKernelName<IN, OUT, W, RD, LS>>(
            sycl::nd_range<1>(sycl::range<1>((size_t) rows * LS), sycl::range<1>(LS)),
            [=](sycl::nd_item<1> it)
        {
            const int row = (int) it.get_group_linear_id();
            const int t = (int) it.get_local_id(0);

            const IN* xrow = x + (size_t) row * dim;
            OUT* yrow = y + (size_t) row * dim;
            RD* rrow = (res == 1) ? r + (size_t) row * dim : nullptr;

            float partial = 0.0f;
            for (int c = t; c < dim; c += LS)
            {
                float v;
                if (res == 1)
                {
                    // r += x, rounded to the residual dtype so the result matches an unfused add
                    float rv = (float) rrow[c] + (float) xrow[c];
                    rrow[c] = (RD) rv;
                    v = (float) rrow[c];
                }
                else
                    v = (float) xrow[c];
                partial += v * v;
            }
            red[t] = partial;
            sycl::group_barrier(it.get_group());
            // tree reduction over the partials (the serial thread-0 sum cost ~LS cycles)
            for (int st = LS / 2; st > 0; st >>= 1)
            {
                if (t < st) red[t] += red[t + st];
                sycl::group_barrier(it.get_group());
            }

            const float rmf = sycl::rsqrt(red[0] / (float) dim + epsilon) * constant_scale;

            for (int c = t; c < dim; c += LS)
            {
                float v;
                if (res == 1)
                    v = (float) rrow[c];
                else
                    v = (float) xrow[c];
                if constexpr (!std::is_same_v<W, int>)
                {
                    float wv = (float) w[c];
                    if (constant_bias != 0.0f) wv += constant_bias;
                    v = v * wv * rmf;
                }
                else
                    v = v * rmf;
                if (res == 2)
                    v = v + (float) yrow[c];
                yrow[c] = (OUT) v;
            }
        });
    });
}

// Weight-dtype dispatch shared by all rms_norm res modes (0 = plain, 1 = residual-in, 2 = residual
// out). RD is only read for res == 1; the other modes pass sycl::half as a placeholder
template <typename IN, typename OUT>
void rms_norm_dispatch_w(sycl::queue& q, const void* x_ptr, const void* w_ptr, bool w_is_bf16,
                         void* y_ptr, void* r_ptr, float epsilon, int rows, int dim,
                         float constant_bias, float constant_scale, int res_mode,
                         at::ScalarType rdt)
{
    using bf = sycl::ext::oneapi::bfloat16;
    #define RN_CALL_WR(W_T, RD_T) \
        do { \
            if (dim > 1024) \
                launch_rms_norm<IN, OUT, W_T, RD_T, 512>(q, (const IN*) x_ptr, (const W_T*) w_ptr, \
                    (OUT*) y_ptr, (RD_T*) r_ptr, epsilon, rows, dim, constant_bias, constant_scale, res_mode); \
            else \
                launch_rms_norm<IN, OUT, W_T, RD_T, 128>(q, (const IN*) x_ptr, (const W_T*) w_ptr, \
                    (OUT*) y_ptr, (RD_T*) r_ptr, epsilon, rows, dim, constant_bias, constant_scale, res_mode); \
        } while (0)
    #define RN_CALL_W(W_T) \
        do { \
            if (res_mode == 1 && rdt == at::kBFloat16) RN_CALL_WR(W_T, bf); \
            else if (res_mode == 1 && rdt == at::kFloat) RN_CALL_WR(W_T, float); \
            else RN_CALL_WR(W_T, sycl::half); \
        } while (0)
    if (!w_ptr)          RN_CALL_W(int);
    else if (w_is_bf16)  RN_CALL_W(bf);
    else                 RN_CALL_W(sycl::half);
    #undef RN_CALL_WR
    #undef RN_CALL_W
}

// -------------------------------------------------------------------------------------------------
// gated_rms_norm: y = norm(x) * (w + bias) * act(g) (gate_last) or norm(x * act(g)) * w (gate_first)
// x is bf16, gate fp32/bf16, weight half/bf16, output half/float
// -------------------------------------------------------------------------------------------------

inline float gate_fn(float v, int gate_act)
{
    if (gate_act == 1)
        return 1.0f / (1.0f + sycl::exp(-v));                       // sigmoid
    return v / (1.0f + sycl::exp(-v));                              // silu
}

template <typename W, typename OUT, typename G>
class GatedRmsNormKernelName;

template <typename W, typename OUT, typename G>
void launch_gated_rms_norm(sycl::queue& q, const bf16* x, const W* w, OUT* y, const G* g,
                           float epsilon, int rows, int dim, float constant_bias,
                           int w_groups, bool gate_first, int gate_act)
{
    constexpr int LS = 128;
    q.submit([&](sycl::handler& cgh)
    {
        sycl::local_accessor<float, 1> red(sycl::range<1>(LS), cgh);
        cgh.parallel_for<GatedRmsNormKernelName<W, OUT, G>>(
            sycl::nd_range<1>(sycl::range<1>((size_t) rows * LS), sycl::range<1>(LS)),
            [=](sycl::nd_item<1> it)
        {
            const int row = (int) it.get_group_linear_id();
            const int t = (int) it.get_local_id(0);

            const bf16* xrow = x + (size_t) row * dim;
            OUT* yrow = y + (size_t) row * dim;
            const G* grow = g + (size_t) row * dim;
            const W* wrow = w + (size_t) (row % w_groups) * dim;

            float partial = 0.0f;
            for (int c = t; c < dim; c += LS)
            {
                float v = (float) xrow[c];
                if (gate_first) v = v * gate_fn((float) grow[c], gate_act);
                partial += v * v;
            }
            red[t] = partial;
            sycl::group_barrier(it.get_group());
            // tree reduction over the partials (the serial thread-0 sum cost ~LS cycles)
            for (int st = LS / 2; st > 0; st >>= 1)
            {
                if (t < st) red[t] += red[t + st];
                sycl::group_barrier(it.get_group());
            }

            const float rmf = sycl::rsqrt(red[0] / (float) dim + epsilon);

            for (int c = t; c < dim; c += LS)
            {
                float v = (float) xrow[c];
                float wv = (float) wrow[c];
                if (constant_bias != 0.0f) wv += constant_bias;
                float gv = gate_fn((float) grow[c], gate_act);
                if (gate_first)
                    v = (v * gv) * wv * rmf;
                else
                    v = (v * wv * rmf) * gv;
                yrow[c] = (OUT) v;
            }
        });
    });
}

// -------------------------------------------------------------------------------------------------
// act_mul: z = act(x) * y; act = silu / gelu(tanh) / relu2 / relu; act_limit != 0 clamps y to
// +/-limit and x to +limit
// -------------------------------------------------------------------------------------------------

template <int ACT>
inline float apply_act(float x)
{
    if constexpr (ACT == 0)   // silu
        return x / (1.0f + sycl::exp(-x));
    if constexpr (ACT == 1)   // gelu (tanh approximation)
    {
        const float c = 0.797884560803f;  // sqrt(2/Pi)
        float tanh_arg = c * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + sycl::tanh(tanh_arg));
    }
    if constexpr (ACT == 2)   // relu2
    {
        x = sycl::fmax(0.0f, x);
        return x * x;
    }
    if constexpr (ACT == 4)   // relu
        return sycl::fmax(0.0f, x);
}

template <int ACT, typename T>
class ActMulKernelName;

template <int ACT, typename T>
void launch_act_mul(sycl::queue& q, const T* x, const T* y, sycl::half* z, float act_limit,
                    size_t numel)
{
    q.parallel_for<ActMulKernelName<ACT, T>>(sycl::range<1>(numel), [=](sycl::item<1> item)
    {
        const size_t i = item.get_linear_id();
        float xv = (float) x[i];
        float yv = (float) y[i];
        xv = apply_act<ACT>(xv);
        if (act_limit != 0.0f)
        {
            yv = sycl::fmin(sycl::fmax(yv, -act_limit), act_limit);
            xv = sycl::fmin(xv, act_limit);
        }
        z[i] = sycl::half(xv * yv);
    });
}

} // namespace

// --- host entry points ----------------------------------------------------------------------------

void sycl_add(const at::Tensor& x, const at::Tensor& y, at::Tensor& z)
{
    TORCH_CHECK(x.sizes() == y.sizes() && x.sizes() == z.sizes(), "add: shape mismatch");
    const auto dt = x.scalar_type();
    TORCH_CHECK(y.scalar_type() == dt && z.scalar_type() == dt, "add: dtype mismatch");
    const size_t numel = x.numel();
    if (!numel) return;
    sycl::queue& q = exl3_sycl_queue();

    if (dt == at::kHalf)
        launch_add<sycl::half>(q, (const sycl::half*) x.data_ptr(), (const sycl::half*) y.data_ptr(),
                               (sycl::half*) z.data_ptr(), numel);
    else if (dt == at::kFloat)
        launch_add<float>(q, (const float*) x.data_ptr(), (const float*) y.data_ptr(),
                          (float*) z.data_ptr(), numel);
    else if (dt == at::kBFloat16)
        launch_add<bf16>(q, (const bf16*) x.data_ptr(), (const bf16*) y.data_ptr(),
                         (bf16*) z.data_ptr(), numel);
    else
        TORCH_CHECK(false, "add: unsupported dtype");
}

void sycl_softcap(at::Tensor x, at::Tensor y, float scale)
{
    const size_t numel = x.numel();
    TORCH_CHECK(y.scalar_type() == x.scalar_type() && y.numel() == numel, "softcap: y must match x");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "softcap: x and y must be contiguous");
    if (!numel) return;
    sycl::queue& q = exl3_sycl_queue();

    if (x.scalar_type() == at::kFloat)
        launch_softcap<float>(q, (const float*) x.data_ptr(), (float*) y.data_ptr(), numel, scale);
    else if (x.scalar_type() == at::kHalf)
        launch_softcap<sycl::half>(q, (const sycl::half*) x.data_ptr(), (sycl::half*) y.data_ptr(), numel, scale);
    else
        TORCH_CHECK(false, "softcap: unsupported dtype");
}

// Weight dtype gate: half / bf16 / float, or none
static bool wdt_ok(const c10::optional<at::Tensor>& w)
{
    if (!w.has_value()) return true;
    const auto t = w->scalar_type();
    return t == at::kHalf || t == at::kBFloat16 || t == at::kFloat;
}

// Flatten to rows x dim and run the requested res mode
static void rms_norm_run(at::Tensor& x, const c10::optional<at::Tensor>& w, at::Tensor& y,
                         const c10::optional<at::Tensor>& r, float epsilon, float constant_bias,
                         float constant_scale, int res_mode)
{
    const int rows = (int) (x.numel() / x.size(-1));
    const int dim = (int) x.size(-1);
    if (!rows || !dim) return;
    sycl::queue& q = exl3_sycl_queue();

    const void* x_ptr = x.data_ptr();
    void* y_ptr = y.data_ptr();
    const void* w_ptr = w.has_value() ? w->data_ptr() : nullptr;
    const bool w_is_bf16 = w.has_value() && w->scalar_type() == at::kBFloat16;
    const void* r_ptr = r.has_value() ? r->data_ptr() : nullptr;
    const auto rdt = r.has_value() ? r->scalar_type() : at::kHalf;

    const auto xdt = x.scalar_type();
    const auto ydt = y.scalar_type();

    // RD (residual dtype) only matters for res == 1 (residual-in): the dispatch threads r's
    // actual dtype so the folded r += x rounds like an unfused add on the residual's storage
    #define RN_CASE(IN_T, OUT_T) \
        rms_norm_dispatch_w<IN_T, OUT_T>(q, x_ptr, w_ptr, w_is_bf16, y_ptr, const_cast<void*>(r_ptr), \
            epsilon, rows, dim, constant_bias, constant_scale, res_mode, rdt)

    using bf = sycl::ext::oneapi::bfloat16;

    if (xdt == at::kHalf && ydt == at::kHalf)              RN_CASE(sycl::half, sycl::half);
    else if (xdt == at::kHalf && ydt == at::kFloat)        RN_CASE(sycl::half, float);
    else if (xdt == at::kHalf && ydt == at::kBFloat16)     RN_CASE(sycl::half, bf);
    else if (xdt == at::kFloat && ydt == at::kHalf)        RN_CASE(float, sycl::half);
    else if (xdt == at::kFloat && ydt == at::kFloat)       RN_CASE(float, float);
    else if (xdt == at::kFloat && ydt == at::kBFloat16)    RN_CASE(float, bf);
    else if (xdt == at::kBFloat16 && ydt == at::kHalf)     RN_CASE(bf, sycl::half);
    else if (xdt == at::kBFloat16 && ydt == at::kFloat)    RN_CASE(bf, float);
    else if (xdt == at::kBFloat16 && ydt == at::kBFloat16) RN_CASE(bf, bf);
    else TORCH_CHECK(false, "rms_norm: unsupported dtype combination");
    #undef RN_CASE
    #undef RN_CASE_R
}

void sycl_rms_norm(at::Tensor x, const c10::optional<at::Tensor> w, at::Tensor y,
                   float epsilon, float constant_bias, float constant_scale, bool add_residual)
{
    TORCH_CHECK(x.sizes() == y.sizes(), "rms_norm: x/y shape mismatch");
    TORCH_CHECK(wdt_ok(w), "rms_norm: unsupported weight dtype");
    rms_norm_run(x, w, y, c10::nullopt, epsilon, constant_bias, constant_scale, add_residual ? 2 : 0);
}

void sycl_rms_norm_res_in(at::Tensor x, c10::optional<at::Tensor> w, at::Tensor y, at::Tensor r,
                          float epsilon, float constant_bias, float constant_scale)
{
    TORCH_CHECK(x.sizes() == y.sizes() && x.sizes() == r.sizes(), "rms_norm_res_in: shape mismatch");
    TORCH_CHECK(wdt_ok(w), "rms_norm_res_in: unsupported weight dtype");
    rms_norm_run(x, w, y, r, epsilon, constant_bias, constant_scale, 1);
}

void sycl_gated_rms_norm(at::Tensor x, at::Tensor w, at::Tensor y, at::Tensor g,
                         float epsilon, float constant_bias, int w_groups, bool gate_first,
                         int gate_act)
{
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "gated_rms_norm: x must be bf16");
    TORCH_CHECK(x.sizes() == y.sizes(), "gated_rms_norm: x/y shape mismatch");
    TORCH_CHECK(x.sizes() == g.sizes(), "gated_rms_norm: x/g shape mismatch");
    if (x.numel() == 0) return;

    const int rows = (int) (x.numel() / x.size(-1));
    const int dim = (int) x.size(-1);
    sycl::queue& q = exl3_sycl_queue();

    const bf16* x_ptr = (const bf16*) x.data_ptr();
    const bool w_is_bf16 = w.scalar_type() == at::kBFloat16;
    const bool w_is_f32 = w.scalar_type() == at::kFloat;

    #define GRN_CASE(OUT_T) \
        do { \
            if (g.scalar_type() == at::kFloat) \
            { \
                if (w_is_bf16) launch_gated_rms_norm<bf16, OUT_T, float>(q, x_ptr, (const bf16*) w.data_ptr(), (OUT_T*) y.data_ptr(), \
                    (const float*) g.data_ptr(), epsilon, rows, dim, constant_bias, w_groups, gate_first, gate_act); \
                else if (w_is_f32) launch_gated_rms_norm<float, OUT_T, float>(q, x_ptr, (const float*) w.data_ptr(), \
                    (OUT_T*) y.data_ptr(), (const float*) g.data_ptr(), epsilon, rows, dim, constant_bias, w_groups, gate_first, gate_act); \
                else launch_gated_rms_norm<sycl::half, OUT_T, float>(q, x_ptr, (const sycl::half*) w.data_ptr(), \
                    (OUT_T*) y.data_ptr(), (const float*) g.data_ptr(), epsilon, rows, dim, constant_bias, w_groups, gate_first, gate_act); \
            } \
            else \
            { \
                if (w_is_bf16) launch_gated_rms_norm<bf16, OUT_T, bf16>(q, x_ptr, (const bf16*) w.data_ptr(), (OUT_T*) y.data_ptr(), \
                    (const bf16*) g.data_ptr(), epsilon, rows, dim, constant_bias, w_groups, gate_first, gate_act); \
                else if (w_is_f32) launch_gated_rms_norm<float, OUT_T, bf16>(q, x_ptr, (const float*) w.data_ptr(), \
                    (OUT_T*) y.data_ptr(), (const bf16*) g.data_ptr(), epsilon, rows, dim, constant_bias, w_groups, gate_first, gate_act); \
                else launch_gated_rms_norm<sycl::half, OUT_T, bf16>(q, x_ptr, (const sycl::half*) w.data_ptr(), \
                    (OUT_T*) y.data_ptr(), (const bf16*) g.data_ptr(), epsilon, rows, dim, constant_bias, w_groups, gate_first, gate_act); \
            } \
        } while (0)

    if (y.scalar_type() == at::kHalf) GRN_CASE(sycl::half);
    else if (y.scalar_type() == at::kFloat) GRN_CASE(float);
    else if (y.scalar_type() == at::kBFloat16) GRN_CASE(bf16);
    else TORCH_CHECK(false, "gated_rms_norm: unsupported output dtype");
    #undef GRN_CASE
}

void sycl_act_mul(at::Tensor x, at::Tensor y, at::Tensor z, int act, float act_limit)
{
    const size_t numel = x.numel();
    TORCH_CHECK(y.numel() == numel && z.numel() == numel, "act_mul: numel mismatch");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && z.is_contiguous(), "act_mul: must be contiguous");
    TORCH_CHECK(z.scalar_type() == at::kHalf, "act_mul: z must be float16");
    if (!numel) return;
    sycl::queue& q = exl3_sycl_queue();

    sycl::half* z_ptr = (sycl::half*) z.data_ptr();

    #define AM_CASE(ACT) \
        do { \
            if (x.scalar_type() == at::kHalf) \
                launch_act_mul<ACT, sycl::half>(q, (const sycl::half*) x.data_ptr(), \
                    (const sycl::half*) y.data_ptr(), z_ptr, act_limit, numel); \
            else if (x.scalar_type() == at::kFloat) \
                launch_act_mul<ACT, float>(q, (const float*) x.data_ptr(), \
                    (const float*) y.data_ptr(), z_ptr, act_limit, numel); \
            else TORCH_CHECK(false, "act_mul: unsupported dtype"); \
        } while (0)

    switch (act)
    {
        case 0: AM_CASE(0); break;   // silu
        case 1: AM_CASE(1); break;   // gelu
        case 2: AM_CASE(2); break;   // relu2
        case 4: AM_CASE(4); break;   // relu
        default: TORCH_CHECK(false, "act_mul: unsupported activation");
    }
    #undef AM_CASE
}

// -------------------------------------------------------------------------------------------------
// Attention gates: x *= sigmoid(y) (element-wise and broadcast), x *= softplus(y) (broadcast,
// computed in fp32 like the reference)
// -------------------------------------------------------------------------------------------------

template <bool BROADCAST, typename T>
class MulSigmoidKernelName;

inline float sigmoid_fast(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float softplus_fast(float x) { return sycl::fmax(x, 0.0f) + sycl::log1p(sycl::exp(-sycl::fabs(x))); }

// sigmoid gate, element-wise: x *= sigmoid(y) evaluated in fp16 like the CUDA kernel
template <typename T>
void launch_mul_sigmoid(sycl::queue& q, T* x, const T* y, size_t numel)
{
    constexpr int NT = 256;
    const size_t pairs = numel / 2;
    const size_t groups = (pairs + NT - 1) / NT;
    q.parallel_for<MulSigmoidKernelName<false, T>>(
        sycl::nd_range<1>(sycl::range<1>(groups * NT), sycl::range<1>(NT)),
        [=](sycl::nd_item<1> it)
    {
        const size_t i = it.get_global_id(0);
        if (i >= pairs) return;
        T a = x[i * 2], b = x[i * 2 + 1];
        T g0 = (T) sigmoid_fast((float) y[i * 2]);
        T g1 = (T) sigmoid_fast((float) y[i * 2 + 1]);
        x[i * 2] = (T) ((float) a * (float) g0);
        x[i * 2 + 1] = (T) ((float) b * (float) g1);
    });
}

// sigmoid gate, broadcast: y is [.., H], x [.., H, D], dim = D
template <typename T>
void launch_mul_sigmoid_broadcast(sycl::queue& q, T* x, const T* y, size_t numel, size_t dim)
{
    constexpr int NT = 256;
    const size_t pairs = numel / 2;
    const size_t groups = (pairs + NT - 1) / NT;
    q.parallel_for<MulSigmoidKernelName<true, T>>(
        sycl::nd_range<1>(sycl::range<1>(groups * NT), sycl::range<1>(NT)),
        [=](sycl::nd_item<1> it)
    {
        const size_t i = it.get_global_id(0);
        if (i >= pairs) return;
        T a = x[i * 2], b = x[i * 2 + 1];
        T g0 = (T) sigmoid_fast((float) y[(i * 2) / dim]);
        T g1 = (T) sigmoid_fast((float) y[(i * 2 + 1) / dim]);
        x[i * 2] = (T) ((float) a * (float) g0);
        x[i * 2 + 1] = (T) ((float) b * (float) g1);
    });
}

// softplus gate, broadcast, computed in fp32 (the reference evaluates softplus(g.float()))
template <typename T>
class MulSoftplusKernelName;

template <typename T>
void launch_mul_softplus_broadcast(sycl::queue& q, T* x, const T* y, size_t numel, size_t dim)
{
    constexpr int NT = 256;
    const size_t pairs = numel / 2;
    const size_t groups = (pairs + NT - 1) / NT;
    q.parallel_for<MulSoftplusKernelName<T>>(
        sycl::nd_range<1>(sycl::range<1>(groups * NT), sycl::range<1>(NT)),
        [=](sycl::nd_item<1> it)
    {
        const size_t i = it.get_global_id(0);
        if (i >= pairs) return;
        float a = (float) x[i * 2], b = (float) x[i * 2 + 1];
        float g0 = softplus_fast((float) y[(i * 2) / dim]);
        float g1 = softplus_fast((float) y[(i * 2 + 1) / dim]);
        x[i * 2] = (T) (a * g0);
        x[i * 2 + 1] = (T) (b * g1);
    });
}

void sycl_mul_sigmoid(at::Tensor x, at::Tensor y)
{
    TORCH_CHECK(x.scalar_type() == at::kHalf && y.scalar_type() == at::kHalf,
                "mul_sigmoid_: half tensors expected");
    TORCH_CHECK(x.numel() == y.numel(), "mul_sigmoid_: numel mismatch");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "mul_sigmoid_: contiguous expected");
    if (!x.numel()) return;
    sycl::queue& q = exl3_sycl_queue();
    launch_mul_sigmoid<sycl::half>(q, (sycl::half*) x.data_ptr(), (const sycl::half*) y.data_ptr(), x.numel());
}

void sycl_mul_sigmoid_broadcast(at::Tensor x, at::Tensor y)
{
    TORCH_CHECK(x.scalar_type() == at::kHalf && y.scalar_type() == at::kHalf,
                "mul_sigmoid_broadcast_: half tensors expected");
    TORCH_CHECK(x.dim() == 4 && y.dim() == 3, "mul_sigmoid_broadcast_: x [B,S,H,D], y [B,S,H]");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "mul_sigmoid_broadcast_: contiguous expected");
    if (!x.numel()) return;
    sycl::queue& q = exl3_sycl_queue();
    launch_mul_sigmoid_broadcast<sycl::half>(q, (sycl::half*) x.data_ptr(), (const sycl::half*) y.data_ptr(),
                                       x.numel(), (size_t) x.size(3));
}

void sycl_mul_softplus_broadcast(at::Tensor x, at::Tensor y)
{
    TORCH_CHECK(x.scalar_type() == at::kHalf && y.scalar_type() == at::kHalf,
                "mul_softplus_broadcast_: half tensors expected");
    TORCH_CHECK(x.dim() == 4 && y.dim() == 3, "mul_softplus_broadcast_: x [B,S,H,D], y [B,S,H]");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "mul_softplus_broadcast_: contiguous expected");
    if (!x.numel()) return;
    sycl::queue& q = exl3_sycl_queue();
    launch_mul_softplus_broadcast<sycl::half>(q, (sycl::half*) x.data_ptr(), (const sycl::half*) y.data_ptr(),
                                        x.numel(), (size_t) x.size(3));
}

// -------------------------------------------------------------------------------------------------
// deinterleave_qg: qg is (.., heads * 2 * head_dim) half with each head's segment laid out as
// [q (head_dim), g (head_dim)]; split into q and g
// -------------------------------------------------------------------------------------------------

class DeinterleaveQgKernelName;

void sycl_deinterleave_qg(const at::Tensor& qg, at::Tensor& q, at::Tensor& g, int head_dim)
{
    TORCH_CHECK(qg.scalar_type() == at::kHalf && q.scalar_type() == at::kHalf &&
                g.scalar_type() == at::kHalf, "deinterleave_qg: half tensors expected");
    TORCH_CHECK(head_dim > 0 && head_dim % 8 == 0, "deinterleave_qg: head_dim must be a multiple of 8");
    TORCH_CHECK(qg.is_contiguous() && q.is_contiguous() && g.is_contiguous(),
                "deinterleave_qg: contiguous expected");
    TORCH_CHECK(q.numel() == g.numel() && q.numel() * 2 == qg.numel(), "deinterleave_qg: size mismatch");
    if (!q.numel()) return;

    const int hd8 = head_dim / 8;
    const size_t n8 = q.numel() / 8;
    sycl::queue& q_queue = exl3_sycl_queue();
    constexpr int NT = 256;
    const size_t groups = (n8 + NT - 1) / NT;
    const sycl::half* src = (const sycl::half*) qg.data_ptr();
    sycl::half* qp = (sycl::half*) q.data_ptr();
    sycl::half* gp = (sycl::half*) g.data_ptr();

    q_queue.parallel_for<DeinterleaveQgKernelName>(
        sycl::nd_range<1>(sycl::range<1>(groups * NT), sycl::range<1>(NT)),
        [=](sycl::nd_item<1> it)
    {
        const size_t i = it.get_global_id(0);
        if (i >= n8) return;
        const size_t d = i % hd8;
        const size_t h = i / hd8;
        const size_t src8 = (h * 2 * hd8 + d) * 8;
        for (int j = 0; j < 8; ++j)
        {
            qp[i * 8 + j] = src[src8 + j];
            gp[i * 8 + j] = src[src8 + hd8 * 8 + j];
        }
    });
}

// -------------------------------------------------------------------------------------------------
// paged_kv_cache_update: scatter k/v into the paged fp16 cache. For token (b, s):
// page = block_table[b, (seqlen + s) / 256], pos = (seqlen + s) % 256.
// -------------------------------------------------------------------------------------------------

class PagedKvUpdateKernelName;

void sycl_paged_kv_cache_update(const at::Tensor& k, const at::Tensor& v,
                                at::Tensor& k_cache, at::Tensor& v_cache,
                                const at::Tensor& block_table, const at::Tensor& cache_seqlens)
{
    TORCH_CHECK(k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf &&
                k_cache.scalar_type() == at::kHalf && v_cache.scalar_type() == at::kHalf,
                "paged_kv_cache_update: half tensors expected");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "paged_kv_cache_update: k/v must be [B, S, H, D]");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
                "paged_kv_cache_update: cache must be [num_blocks, 256, H, D]");
    TORCH_CHECK(k_cache.size(1) == 256 && v_cache.size(1) == 256, "paged_kv_cache_update: page_size == 256");
    TORCH_CHECK(block_table.dim() == 2 && cache_seqlens.dim() == 1,
                "paged_kv_cache_update: block_table [B, max_blocks], cache_seqlens [B]");
    TORCH_CHECK(k.is_contiguous() && v.is_contiguous() && k_cache.is_contiguous() &&
                v_cache.is_contiguous() && block_table.is_contiguous() && cache_seqlens.is_contiguous(),
                "paged_kv_cache_update: tensors must be contiguous");
    TORCH_CHECK(k_cache.size(2) == k.size(2) && k_cache.size(3) == k.size(3),
                "paged_kv_cache_update: cache heads/dim do not match k");

    const int B = (int) k.size(0);
    const int S = (int) k.size(1);
    const int H = (int) k.size(2);
    const int D = (int) k.size(3);
    if (!B || !S || !H || !D) return;

    sycl::queue& q = exl3_sycl_queue();
    constexpr int NT = 256;
    const size_t tokens = (size_t) B * S;
    const size_t groups = (tokens + NT - 1) / NT;
    const sycl::half* kp = (const sycl::half*) k.data_ptr();
    const sycl::half* vp = (const sycl::half*) v.data_ptr();
    sycl::half* kcp = (sycl::half*) k_cache.data_ptr();
    sycl::half* vcp = (sycl::half*) v_cache.data_ptr();
    const int* btp = (const int*) block_table.data_ptr();
    const int* slp = (const int*) cache_seqlens.data_ptr();
    const int max_blocks = (int) block_table.size(1);

    q.parallel_for<PagedKvUpdateKernelName>(
        sycl::nd_range<1>(sycl::range<1>(groups * NT), sycl::range<1>(NT)),
        [=](sycl::nd_item<1> it)
    {
        const size_t tok = it.get_global_id(0);
        if (tok >= tokens) return;
        const int bi = (int) (tok / S);
        const int si = (int) (tok % S);
        const int t = slp[bi] + si;                 // absolute position in the sequence
        const int page = btp[bi * max_blocks + (t >> 8)];      // 256 tokens per page
        const int pos = t & 255;
        const size_t src = ((size_t) bi * S + si) * H * D;
        const size_t dst = ((size_t) page * 256 + pos) * H * D;
        for (int i = 0; i < H * D; ++i)
        {
            kcp[dst + i] = kp[src + i];
            vcp[dst + i] = vp[src + i];
        }
    });
}
