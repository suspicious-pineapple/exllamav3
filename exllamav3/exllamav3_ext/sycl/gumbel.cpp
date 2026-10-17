// Gumbel-noise sampling kernels, port of generator/gumbel.cu + the packed-logit-bitmask kernel of
// generator/sampling_fused.cu into SYCL.
//
// The RNG is a Philox4x32-10 counter-based generator, bit-identical to cuRAND's
// curand_init(random, idx, 0) + curand_uniform: the kernels read exactly the first output word of
// the counter (0, 0, idx_lo, idx_hi) with key (seed_lo, seed_hi), so sampling parity holds.

#include <sycl/sycl.hpp>
#include <ATen/Tensor.h>
#include <cmath>

#include "exl3_compat.h"

namespace {

// cuRAND Philox4x32-10 constants
constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;
constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;
constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;

struct PhiloxCtr { uint32_t x, y, z, w; };
struct PhiloxKey { uint32_t x, y; };

inline void philox_mulhilo(uint32_t a, uint32_t b, uint32_t* hi, uint32_t* lo)
{
    const uint64_t p = (uint64_t) a * (uint64_t) b;
    *hi = (uint32_t) (p >> 32);
    *lo = (uint32_t) p;
}

inline PhiloxCtr philox_round(PhiloxCtr c, PhiloxKey k)
{
    uint32_t hi0, lo0, hi1, lo1;
    philox_mulhilo(PHILOX_M0, c.x, &hi0, &lo0);
    philox_mulhilo(PHILOX_M1, c.z, &hi1, &lo1);
    return { hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0 };
}

inline PhiloxCtr philox4x32_10(PhiloxCtr c, PhiloxKey k)
{
    for (int i = 0; i < 10; ++i)
    {
        c = philox_round(c, k);
        if (i < 9)
        {
            k.x += PHILOX_W0;
            k.y += PHILOX_W1;
        }
    }
    return c;
}

// curand_init(random, idx, 0) + first curand_uniform: first output word of the counter
// (0, 0, idx_lo, idx_hi) with key (seed_lo, seed_hi)
inline float curand_uniform_first(uint32_t seed, uint32_t subseq)
{
    const PhiloxCtr c = { 0u, 0u, subseq, 0u };
    const PhiloxKey k = { seed, 0u };
    const PhiloxCtr out = philox4x32_10(c, k);
    constexpr float CURAND_2POW32_INV = 2.3283064e-10f;
    return out.x * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
}

inline float gumbel(float x)
{
    x = sycl::fmin(x, 0.99999994f);   // Largest float32 < 1.0
    return -sycl::log(sycl::fmax(-sycl::log(sycl::fmax(x, 1e-20f)), 1e-20f));
}

// -------------------------------------------------------------------------------------------------
// gumbel noise kernels: one work item per element
// -------------------------------------------------------------------------------------------------

template <typename T, bool LOG>
class GumbelKernelName;

template <typename T, bool LOG>
void launch_gumbel(sycl::queue& q, const T* in, float* logits, size_t size, uint32_t random)
{
    q.parallel_for<GumbelKernelName<T, LOG>>(sycl::range<1>(size), [=](sycl::item<1> item)
    {
        const size_t idx = item.get_linear_id();
        float x = (float) in[idx];
        if constexpr (LOG) x = sycl::log(x);
        const float u = curand_uniform_first(random, (uint32_t) (idx & 0xffffffffu));
        logits[idx] = x + gumbel(u);
    });
}

// fp16 output variant (SS_Sample's SS.INIT state adds gumbel noise in place on the half logits)
template <bool LOG>
class GumbelHKernelName;

template <bool LOG>
void launch_gumbel_h(sycl::queue& q, const sycl::half* in, sycl::half* logits, size_t size,
                     uint32_t random)
{
    q.parallel_for<GumbelHKernelName<LOG>>(sycl::range<1>(size), [=](sycl::item<1> item)
    {
        const size_t idx = item.get_linear_id();
        float x = (float) in[idx];
        if constexpr (LOG) x = sycl::log(x);
        const float u = curand_uniform_first(random, (uint32_t) (idx & 0xffffffffu));
        logits[idx] = sycl::half(x + gumbel(u));
    });
}

// -------------------------------------------------------------------------------------------------
// packed bitmask: out[i] = in[i] if bit i of the row word is set else -inf (bit clear = masked out)
// -------------------------------------------------------------------------------------------------

template <typename T>
class BitmaskKernelName;

template <typename T>
void launch_bitmask(sycl::queue& q, const T* in, T* out, const uint32_t* bits, int dim,
                    bool bits_per_row, int bits_words, int bsz)
{
    const int nbits = bits_words * 32;
    q.parallel_for<BitmaskKernelName<T>>(sycl::range<2>((size_t) bsz, (size_t) dim),
        [=](sycl::item<2> item)
    {
        const int row = (int) item.get_id(0);
        const int i = (int) item.get_id(1);
        const uint32_t* row_bits = bits + (bits_per_row ? (size_t) row * bits_words : 0);
        const bool keep = i < nbits && ((row_bits[i >> 5] >> (i & 31)) & 1u);
        out[(size_t) row * dim + i] = keep ? in[(size_t) row * dim + i]
                                           : (T) -std::numeric_limits<float>::infinity();
    });
}

} // namespace

// --- host entry points ----------------------------------------------------------------------------

void sycl_gumbel_noise_f16(const at::Tensor& logits_in, at::Tensor& logits, uint32_t random)
{
    TORCH_CHECK(logits_in.scalar_type() == at::kHalf && logits.scalar_type() == at::kHalf,
                "gumbel_noise_f16: half logits expected");
    TORCH_CHECK(logits_in.sizes() == logits.sizes(), "gumbel_noise_f16: shape mismatch");
    const size_t size = logits.numel();
    if (!size) return;
    sycl::queue& q = exl3_sycl_queue();
    launch_gumbel_h<false>(q, (const sycl::half*) logits_in.data_ptr(),
                           (sycl::half*) logits.data_ptr(), size, random);
}

void sycl_gumbel_noise_f32(const at::Tensor& logits_in, at::Tensor& logits, uint32_t random)
{
    TORCH_CHECK(logits_in.scalar_type() == at::kFloat && logits.scalar_type() == at::kFloat,
                "gumbel_noise_f32: float logits expected");
    TORCH_CHECK(logits_in.sizes() == logits.sizes(), "gumbel_noise_f32: shape mismatch");
    const size_t size = logits.numel();
    if (!size) return;
    sycl::queue& q = exl3_sycl_queue();
    launch_gumbel<float, false>(q, (const float*) logits_in.data_ptr(),
                                (float*) logits.data_ptr(), size, random);
}

void sycl_gumbel_noise_log(const at::Tensor& probs, at::Tensor& logits, uint32_t random)
{
    TORCH_CHECK(probs.scalar_type() == at::kFloat && logits.scalar_type() == at::kFloat,
                "gumbel_noise_log: float probs/logits expected");
    TORCH_CHECK(probs.sizes() == logits.sizes(), "gumbel_noise_log: shape mismatch");
    const size_t size = probs.numel();
    if (!size) return;
    sycl::queue& q = exl3_sycl_queue();
    launch_gumbel<float, true>(q, (const float*) probs.data_ptr(),
                               (float*) logits.data_ptr(), size, random);
}

void sycl_apply_logit_bitmask(const at::Tensor& logits_in, at::Tensor& logits_out,
                              const at::Tensor& bitmask)
{
    TORCH_CHECK(logits_in.dim() == 2, "apply_logit_bitmask: logits must be 2D");
    TORCH_CHECK(logits_in.is_contiguous() && logits_out.is_contiguous() && bitmask.is_contiguous(),
                "apply_logit_bitmask: tensors must be contiguous");
    TORCH_CHECK(logits_out.sizes() == logits_in.sizes() && logits_out.scalar_type() == logits_in.scalar_type(),
                "apply_logit_bitmask: output shape/dtype mismatch");
    TORCH_CHECK(bitmask.scalar_type() == at::kInt && bitmask.dim() == 2,
                "apply_logit_bitmask: bitmask must be 2D int32");
    TORCH_CHECK(bitmask.size(0) == 1 || bitmask.size(0) == logits_in.size(0),
                "apply_logit_bitmask: bad bitmask batch dim");

    const int bsz = (int) logits_in.size(0);
    const int dim = (int) logits_in.size(1);
    if (!bsz || !dim) return;

    sycl::queue& q = exl3_sycl_queue();
    const uint32_t* bits = (const uint32_t*) bitmask.data_ptr();
    const int bits_words = (int) bitmask.size(1);
    const bool bits_per_row = bitmask.size(0) == (size_t) bsz;

    if (logits_in.scalar_type() == at::kHalf)
        launch_bitmask<sycl::half>(q, (const sycl::half*) logits_in.data_ptr(),
            (sycl::half*) logits_out.data_ptr(), bits, dim, bits_per_row, bits_words, bsz);
    else if (logits_in.scalar_type() == at::kFloat)
        launch_bitmask<float>(q, (const float*) logits_in.data_ptr(),
            (float*) logits_out.data_ptr(), bits, dim, bits_per_row, bits_words, bsz);
    else
        TORCH_CHECK(false, "apply_logit_bitmask: logits must be half or float");
}
