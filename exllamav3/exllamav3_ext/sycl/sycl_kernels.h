#pragma once

// Declarations shared between the SYCL backend translation units

#include <sycl/sycl.hpp>
#include <ATen/Tensor.h>
#include <vector>
#include <cstdint>
#include <cstddef>

// hadamard.cpp
void sycl_had_r_128(const at::Tensor& input, const at::Tensor& output,
                    const c10::optional<at::Tensor>& pre_scale,
                    const c10::optional<at::Tensor>& post_scale, float scale);
void sycl_had_r_128_h(const at::Tensor& input, const at::Tensor& output,
                      const c10::optional<at::Tensor>& pre_scale,
                      const c10::optional<at::Tensor>& post_scale, float scale);
void sycl_had_r_128_f(const at::Tensor& input, const at::Tensor& output,
                      const c10::optional<at::Tensor>& pre_scale,
                      const c10::optional<at::Tensor>& post_scale, float scale);

// reconstruct.cpp
void sycl_reconstruct_impl(sycl::queue& q, at::Tensor unpacked, at::Tensor packed,
                           float K_, bool mcg, bool mul1, int64_t n_offset);
void sycl_reconstruct_had_impl(sycl::queue& q, at::Tensor unpacked, at::Tensor packed,
                               at::Tensor suh, at::Tensor svh, float K_, bool mcg, bool mul1,
                               int64_t n_offset);
void sycl_reconstruct(at::Tensor unpacked, at::Tensor packed, float K_, bool mcg, bool mul1);
void sycl_reconstruct_slice(at::Tensor unpacked, at::Tensor packed, float K_, bool mcg, bool mul1,
                            int64_t n_offset);
void sycl_reconstruct_had_slice(at::Tensor unpacked, at::Tensor packed, at::Tensor suh,
                                at::Tensor svh, float K_, bool mcg, bool mul1, int64_t n_offset);

// elementwise.cpp
void sycl_mul_sigmoid(at::Tensor x, at::Tensor y);
void sycl_mul_sigmoid_broadcast(at::Tensor x, at::Tensor y);
void sycl_mul_softplus_broadcast(at::Tensor x, at::Tensor y);
void sycl_deinterleave_qg(const at::Tensor& qg, at::Tensor& q, at::Tensor& g, int head_dim);
void sycl_paged_kv_cache_update(const at::Tensor& k, const at::Tensor& v,
                                at::Tensor& k_cache, at::Tensor& v_cache,
                                const at::Tensor& block_table, const at::Tensor& cache_seqlens);

// rope.cpp
void sycl_rope(const at::Tensor& q, at::Tensor& out_q,
               const c10::optional<at::Tensor>& k, c10::optional<at::Tensor>& out_k,
               const at::Tensor& inv_freq,
               int64_t position,
               const c10::optional<at::Tensor>& positions,
               const c10::optional<at::Tensor>& position_ids,
               int64_t rope_mode, float attn_factor,
               const c10::optional<at::Tensor>& q_norm, const c10::optional<at::Tensor>& k_norm,
               float norm_eps, float norm_constant_bias,
               float llama_4_scaling_beta, int64_t llama_4_scaling_original,
               int64_t rotate_dims, int64_t rotate_offset);

// gemv.cpp
void sycl_exl3_gemm_impl(sycl::queue& q, const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
                         const c10::optional<at::Tensor>& suh,
                         const c10::optional<at::Tensor>& A_had,
                         const c10::optional<at::Tensor>& svh,
                         int64_t force_shape_idx, bool mcg, bool mul1, int64_t force_num_sms);
void sycl_exl3_gemm(const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
                    const c10::optional<at::Tensor>& suh,
                    const c10::optional<at::Tensor>& A_had,
                    const c10::optional<at::Tensor>& svh,
                    int64_t force_shape_idx, bool mcg, bool mul1, int64_t force_num_sms);
void sycl_exl3_gemv(const at::Tensor& A, const at::Tensor& B, at::Tensor& C,
                    const c10::optional<at::Tensor>& suh,
                    const c10::optional<at::Tensor>& A_had,
                    const c10::optional<at::Tensor>& svh, bool mcg, bool mul1);

// gemv.cpp raw-pointer entries (used by mgemm.cpp's pointer-table loop):
//   had: rows x (chunks*128) in/out with optional pre/post scale vectors
//   gemv: had(pre=suh) -> GEMV -> had(post=svh) over one trellis
void sycl_had_h_raw(sycl::queue& q, const sycl::half* input, sycl::half* output,
                    const sycl::half* pre_scale, const sycl::half* post_scale, float scale,
                    int rows, int chunks, int stride = 0);
void sycl_had_f_raw(sycl::queue& q, const float* input, float* output,
                    const sycl::half* pre_scale, const sycl::half* post_scale, float scale,
                    int rows, int chunks, int stride = 0);
void sycl_exl3_gemm_raw(sycl::queue& q, const sycl::half* A, sycl::half* A_had,
                        const uint16_t* trellis, void* C, const sycl::half* suh,
                        const sycl::half* svh, int size_m, int size_k, int size_n, bool c_fp32,
                        int K, int cb, bool half_k);
void sycl_exl3_gemm_raw2(sycl::queue& q, const sycl::half* A_had,
                         const uint16_t* trellis, void* C, const sycl::half* svh,
                         int size_m, int size_k, int size_n, int stride, bool c_fp32,
                         int K, int cb, bool half_k);

// mgemm.cpp
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
                     int64_t num_had_src, bool dense);
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
                     int64_t num_had_src, bool dense);
void exl3_sycl_ptr_cache_clear();

// hgemm.cpp
void sycl_hgemm(at::Tensor a, at::Tensor b, at::Tensor c);
void sycl_hgemm_recon(at::Tensor a, at::Tensor b, at::Tensor c);
// Custom split-K dense GEMV (m == 1): C[i] = sum_k x[k] * W[k*w_ld + i]. The caller gates on
// size_n % 128 == 0 (the dense weights only apply at multiples of 128)
void sycl_dense_gemv_raw(sycl::queue& q, const sycl::half* x, const sycl::half* W, int w_ld,
                         void* C, int size_k, int size_n, bool out_fp32);

// gdn.cpp -- speculative-decoding rewind (GDN conv ring + recurrent-state replay). Job structs
// mirror gdn.cuh's CUDA definitions; addresses are raw device pointers computed in Python
struct ConvRewindJob
{
    uintptr_t src;
    uintptr_t dst;
    int dim;
    int cdim;
    int stride;

    ConvRewindJob() = default;
    ConvRewindJob(uintptr_t _src, uintptr_t _dst, int _dim, int _cdim, int _stride) :
        src(_src), dst(_dst), dim(_dim), cdim(_cdim), stride(_stride) {}
};

struct ScanReplayJob
{
    uintptr_t qkv;      // staged post-conv qkv rows, [prefix, qkv_dim] bf16
    uintptr_t g;        // staged log decay, [prefix, heads] (or [prefix, heads, dk] KDA) fp32
    uintptr_t beta;     // staged beta, [prefix, heads] bf16
    uintptr_t base;     // state row read for the first token, fp32
    uintptr_t scratch;  // state row written (first token from base, then in place)
    uintptr_t D;        // Mamba2 skip scales, else 0 (unused: Mamba2 scan not ported)
    int prefix;
    int tstride;        // staged rows between consecutive tokens

    ScanReplayJob() = default;
    ScanReplayJob(uintptr_t _qkv, uintptr_t _g, uintptr_t _beta, uintptr_t _base, uintptr_t _scratch,
                  uintptr_t _D, int _prefix, int _tstride = 1) :
        qkv(_qkv), g(_g), beta(_beta), base(_base), scratch(_scratch), D(_D), prefix(_prefix), tstride(_tstride) {}
};

void sycl_batched_conv_rewind(std::vector<ConvRewindJob> const& jobs, int64_t device_index);
void sycl_batched_scan_replay(std::vector<ScanReplayJob> const& jobs, int64_t device_index,
                              int64_t kind, int64_t num_k_heads, int64_t num_v_heads,
                              int64_t k_head_dim, int64_t v_head_dim);

// elementwise.cpp
void sycl_add(const at::Tensor& x, const at::Tensor& y, at::Tensor& z);
void sycl_softcap(at::Tensor x, at::Tensor y, float scale);
void sycl_rms_norm(at::Tensor x, const c10::optional<at::Tensor> w, at::Tensor y,
                   float epsilon, float constant_bias, float constant_scale, bool add_residual);
void sycl_rms_norm_res_in(at::Tensor x, c10::optional<at::Tensor> w, at::Tensor y, at::Tensor r,
                          float epsilon, float constant_bias, float constant_scale);
void sycl_gated_rms_norm(at::Tensor x, at::Tensor w, at::Tensor y, at::Tensor g,
                         float epsilon, float constant_bias, int w_groups, bool gate_first,
                         int gate_act);
void sycl_act_mul(at::Tensor x, at::Tensor y, at::Tensor z, int act, float act_limit);

// gdn.cpp
void sycl_gated_delta_net_fused_op(
    const at::Tensor& mixed_qkvz, const at::Tensor& mixed_ba,
    const at::Tensor& dt_bias, const at::Tensor& a_log,
    at::Tensor& mixed_qkv, at::Tensor& z, at::Tensor& beta, at::Tensor& g,
    int64_t num_k_heads, int64_t num_v_heads, int64_t k_head_dim, int64_t v_head_dim,
    double beta_scale);
void sycl_gated_delta_net_fused_op_2(
    const at::Tensor& b, const at::Tensor& a, const at::Tensor& dt_bias, const at::Tensor& a_log,
    at::Tensor& beta, at::Tensor& g, double beta_scale);
void sycl_causal_conv1d_update(
    const at::Tensor& x, at::Tensor& conv_state,
    const c10::optional<at::Tensor>& slots,
    const at::Tensor& weight, const c10::optional<at::Tensor>& bias,
    at::Tensor& out, bool activation, bool history, bool incremental);
void sycl_recurrent_gated_delta_rule(
    const at::Tensor& mixed_qkv, const at::Tensor& g, const at::Tensor& beta,
    at::Tensor& recurrent_state, at::Tensor& core_attn_out,
    int64_t num_k_heads, int64_t num_v_heads, int64_t k_head_dim, int64_t v_head_dim,
    const c10::optional<at::Tensor>& slots, bool history,
    const c10::optional<at::Tensor>& slots_in);

// gumbel.cpp
void sycl_gumbel_noise_f16(const at::Tensor& logits_in, at::Tensor& logits, uint32_t random);
void sycl_gumbel_noise_f32(const at::Tensor& logits_in, at::Tensor& logits, uint32_t random);
void sycl_gumbel_noise_log(const at::Tensor& probs, at::Tensor& logits, uint32_t random);
void sycl_apply_logit_bitmask(const at::Tensor& logits_in, at::Tensor& logits_out,
                              const at::Tensor& bitmask);

// flash_attn.cpp
void sycl_flash_decode(const at::Tensor& q, const at::Tensor& k_buf, const at::Tensor& v_buf,
                       int64_t kv_len, double scale, at::Tensor& out);
