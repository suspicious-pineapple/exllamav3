// pybind module for the SYCL backend (exllamav3_sycl_ext), mirroring the subset of the CUDA
// exllamav3_ext surface the model path uses. Functions the backend does not implement raise with
// a clear message instead of binding silently.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>
#include <c10/core/DeviceGuard.h>

#include "exl3_compat.h"
#include "sycl_kernels.h"

// -------------------------------------------------------------------------------------------------
// Queue: kernels run on the current torch XPU stream so they stay ordered with torch ops
// -------------------------------------------------------------------------------------------------

sycl::queue& exl3_sycl_queue()
{
    return c10::xpu::getCurrentXPUStream().queue();
}

// -------------------------------------------------------------------------------------------------
// Device memory: (free, total) bytes. free_memory comes from the Intel device extension over
// Level Zero; 0 when the query is unsupported (the python side falls back to the allocator)
// -------------------------------------------------------------------------------------------------

std::vector<int64_t> sycl_device_get_mem_info(int64_t device_index)
{
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    TORCH_CHECK(device_index >= 0 && (size_t) device_index < devices.size(),
                "device_get_mem_info: device index out of range");
    const sycl::device& dev = devices[device_index];
    size_t total = dev.get_info<sycl::info::device::global_mem_size>();
    size_t free = 0;
    try
    {
        free = dev.get_info<sycl::ext::intel::info::device::free_memory>();
    }
    catch (...) {}
    return {(int64_t) free, (int64_t) total};
}

// -------------------------------------------------------------------------------------------------
// BC handles (libtorch glue). Graph* is accepted for signature parity with the CUDA build and must
// always be null here (CUDA graphs are off on SYCL)
// -------------------------------------------------------------------------------------------------

namespace {

inline void check_no_graph(void* graph, const char* who)
{
    TORCH_CHECK(graph == nullptr, who, ": CUDA graphs are not supported on the SYCL backend");
}

}

struct BC_LinearFP16
{
    at::Tensor weight;
    c10::optional<at::Tensor> bias;

    BC_LinearFP16(at::Tensor _weight, c10::optional<at::Tensor> _bias)
        : weight(std::move(_weight)), bias(std::move(_bias)) {}

    void run_gr(const at::Tensor& x, at::Tensor& y, void* graph)
    {
        check_no_graph(graph, "BC_LinearFP16");
        run(x, y);
    }

    void run(const at::Tensor& x, at::Tensor& y)
    {
        if (x.scalar_type() == y.scalar_type())
            at::matmul_out(const_cast<at::Tensor&>(y), x, weight);
        else
            sycl_hgemm(const_cast<at::Tensor&>(x), const_cast<at::Tensor&>(weight),
                       const_cast<at::Tensor&>(y));
        if (bias)
            sycl_add(y, bias.value(), y);
    }
};

struct BC_LinearEXL3
{
    at::Tensor trellis;
    at::Tensor suh;
    at::Tensor svh;
    float K;                      // bitrate, integer or half-integer
    c10::optional<at::Tensor> bias;
    bool mcg;
    bool mul1;
    at::Tensor xh;                // bsz-1 input Hadamard scratch

    BC_LinearEXL3(at::Tensor _trellis, at::Tensor _suh, at::Tensor _svh, float _K,
                  c10::optional<at::Tensor> _bias, bool _mcg, bool _mul1, at::Tensor _xh)
        : trellis(std::move(_trellis)), suh(std::move(_suh)), svh(std::move(_svh)), K(_K),
          bias(std::move(_bias)), mcg(_mcg), mul1(_mul1), xh(std::move(_xh)) {}

    void run_gr(const at::Tensor& x, at::Tensor& y, void* graph)
    {
        check_no_graph(graph, "BC_LinearEXL3");
        run(x, y);
    }

    void run(const at::Tensor& x, at::Tensor& y)
    {
        if (x.numel() == x.size(-1))
        {
            sycl_exl3_gemm(x, trellis, const_cast<at::Tensor&>(y), suh, xh, svh, -1, mcg, mul1, 0);
        }
        else
        {
            at::Tensor xh_ = at::empty_like(x);
            sycl_exl3_gemm(x, trellis, const_cast<at::Tensor&>(y), suh, xh_, svh, -1, mcg, mul1, 0);
        }
        if (bias)
            sycl_add(y, bias.value(), y);
    }

    at::Tensor run_alloc(const at::Tensor& x, int64_t out_features, bool output_fp32)
    {
        std::vector<int64_t> out_shape = x.sizes().vec();
        out_shape.back() = out_features;

        at::Tensor y = at::empty(
            out_shape,
            x.options().dtype(output_fp32 ? at::kFloat : at::kHalf)
        );
        if (out_features == 0) return y;

        at::Tensor x_flat = x.view({-1, x.size(-1)});
        at::Tensor y_flat = y.view({-1, out_features});
        run(x_flat, y_flat);
        return y;
    }
};

// Stored but never run on SYCL: the gated norm goes through gated_rms_norm directly, and the
// graphed consumers that hold this handle are CUDA-only
struct BC_GatedRMSNorm
{
    at::Tensor weight;
    double epsilon;
    double constant_bias;
    int64_t groups;
    bool gate_first;
    int64_t gate_act;

    BC_GatedRMSNorm(at::Tensor _weight, double _epsilon, double _constant_bias, int64_t _groups,
                    bool _gate_first, int64_t _gate_act)
        : weight(std::move(_weight)), epsilon(_epsilon), constant_bias(_constant_bias),
          groups(_groups), gate_first(_gate_first), gate_act(_gate_act) {}

    void run(at::Tensor x, at::Tensor y, at::Tensor gate)
    {
        sycl_gated_rms_norm(x, weight, y, gate, (float) epsilon, (float) constant_bias,
                            (int) groups, gate_first, (int) gate_act);
    }
};

// --- unsupported entry points ---------------------------------------------------------------------

[[noreturn]] void unsupported(const char* name)
{
    throw std::runtime_error(std::string(name) + " is not supported on the SYCL backend");
}

void sycl_fused_sampler(const at::Tensor&, const c10::optional<at::Tensor>&,
                        const c10::optional<at::Tensor>&, const at::Tensor&, const at::Tensor&,
                        int64_t, double, double, uint32_t, int64_t, const at::Tensor&,
                        int64_t, double, const c10::optional<at::Tensor>&)
{ unsupported("fused_sampler"); }

void sycl_apply_rep_pens(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, const at::Tensor&, double, double, double, double, double,
                         int64_t, int64_t, int64_t, int64_t, int64_t, int64_t)
{ unsupported("apply_rep_pens"); }

void sycl_apply_pres_freq_pens(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, double, int64_t, int64_t, int64_t)
{ unsupported("apply_pres_freq_pens"); }

void sycl_dry_penalty(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                      int64_t, int64_t, double, double, double, double, int64_t, int64_t, int64_t, int64_t)
{ unsupported("dry_penalty"); }

void sycl_adaptivep_gumbel_noise(const at::Tensor&, const at::Tensor&, uint32_t, const at::Tensor&)
{ unsupported("adaptivep_gumbel_noise_f32"); }

void sycl_argmax_sample(const at::Tensor&, at::Tensor&, int64_t, const c10::optional<at::Tensor>&)
{ unsupported("argmax_sample"); }

void sycl_gumbel_sample(const at::Tensor&, const at::Tensor&, int64_t, uint32_t)
{ unsupported("gumbel_sample"); }

void sycl_batched_conv_rewind(std::vector<ConvRewindJob> const& jobs, int64_t device_index);

void sycl_batched_scan_replay(std::vector<ScanReplayJob> const& jobs, int64_t device_index,
                              int64_t kind, int64_t num_k_heads, int64_t num_v_heads,
                              int64_t k_head_dim, int64_t v_head_dim);

at::Tensor sycl_pinned_cuda_view(const at::Tensor&, int64_t)
{ unsupported("pinned_cuda_view"); }

int64_t sycl_gemm_num_kernel_shapes()
{ unsupported("exl3_gemm_num_kernel_shapes"); }

int64_t sycl_gemv_int8_max_k(int64_t)
{ unsupported("exl3_gemv_int8_max_k"); }

// Job descriptors live in sycl_kernels.h (gdn.cpp implements the rewind kernels over them)

// -------------------------------------------------------------------------------------------------
// Module
// -------------------------------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("had_r_128", &sycl_had_r_128, "had_r_128");
    m.def("reconstruct", [](at::Tensor unpacked, at::Tensor packed, float K, bool mcg, bool mul1)
    {
        sycl_reconstruct_impl(exl3_sycl_queue(), unpacked, packed, K, mcg, mul1, 0);
    }, "reconstruct");
    m.def("reconstruct_slice", [](at::Tensor unpacked, at::Tensor packed, float K, bool mcg,
                                  bool mul1, int64_t n_start)
    {
        sycl_reconstruct_impl(exl3_sycl_queue(), unpacked, packed, K, mcg, mul1, n_start);
    }, "reconstruct_slice");
    m.def("reconstruct_had_slice", &sycl_reconstruct_had_slice, "reconstruct_had_slice");

    m.def("hgemm", &sycl_hgemm, "hgemm");
    m.def("hgemm_recon", &sycl_hgemm_recon, "hgemm_recon");

    m.def("add", &sycl_add, "add");
    m.def("softcap", &sycl_softcap, "softcap");
    m.def("rms_norm",
          [](at::Tensor x, const c10::optional<at::Tensor> w, at::Tensor y, float epsilon,
             float constant_bias, float constant_scale, bool span_heads, bool add_residual,
             int64_t w_groups)
    {
        TORCH_CHECK(!span_heads, "rms_norm (SYCL): span_heads is not supported");
        TORCH_CHECK(w_groups == 1, "rms_norm (SYCL): w_groups > 1 is not supported");
        sycl_rms_norm(x, w, y, epsilon, constant_bias, constant_scale, add_residual);
    }, "rms_norm");
    m.def("rms_norm_res_in", &sycl_rms_norm_res_in, "rms_norm_res_in");
    m.def("gated_rms_norm", &sycl_gated_rms_norm, "gated_rms_norm");

    // Activation aliases: the ext kernels treat x, y, z as flat buffers of x.numel() elements
    m.def("silu_mul", [](const at::Tensor& x, const at::Tensor& y, at::Tensor& z, float act_limit)
    { sycl_act_mul(const_cast<at::Tensor&>(x), const_cast<at::Tensor&>(y), z, 0, act_limit); }, "silu_mul");
    m.def("gelu_mul", [](const at::Tensor& x, const at::Tensor& y, at::Tensor& z, float act_limit)
    { sycl_act_mul(const_cast<at::Tensor&>(x), const_cast<at::Tensor&>(y), z, 1, act_limit); }, "gelu_mul");
    m.def("relu2_mul", [](const at::Tensor& x, const at::Tensor& y, at::Tensor& z, float act_limit)
    { sycl_act_mul(const_cast<at::Tensor&>(x), const_cast<at::Tensor&>(y), z, 2, act_limit); }, "relu2_mul");
    m.def("relu_mul", [](const at::Tensor& x, const at::Tensor& y, at::Tensor& z, float act_limit)
    { sycl_act_mul(const_cast<at::Tensor&>(x), const_cast<at::Tensor&>(y), z, 4, act_limit); }, "relu_mul");

    m.def("gated_delta_net_fused_op", &sycl_gated_delta_net_fused_op, "gated_delta_net_fused_op");
    m.def("gated_delta_net_fused_op_2", &sycl_gated_delta_net_fused_op_2, "gated_delta_net_fused_op_2");
    m.def("cuda_recurrent_gated_delta_rule", &sycl_recurrent_gated_delta_rule,
          "recurrent gated delta rule scan (SYCL)");

    m.def("exl3_gemm", &sycl_exl3_gemm, "exl3_gemm");
    m.def("exl3_gemv", &sycl_exl3_gemv, "exl3_gemv");
    m.def("exl3_mgemm", &sycl_exl3_mgemm, "exl3_mgemm",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("suh"), py::arg("A_had"), py::arg("svh"),
          py::arg("indices"), py::arg("weights"), py::arg("K"), py::arg("force_shape_idx"), py::arg("mcg"),
          py::arg("mul1"), py::arg("min_index"), py::arg("max_index"), py::arg("force_num_sms"),
          py::arg("num_tokens") = 1, py::arg("size_n_list") = py::none(), py::arg("c_ptrs") = py::none(),
          py::arg("n_stride_list") = py::none(), py::arg("had_src_list") = py::none(),
          py::arg("num_had_src") = 0, py::arg("dense") = false);

    m.def("exl3_sycl_ptr_cache_clear", &exl3_sycl_ptr_cache_clear,
          "drop cached host copies of the mgemm device pointer tables (call on model unload/load)");

    m.def("gumbel_noise_f16", &sycl_gumbel_noise_f16, "gumbel_noise_f16");
    m.def("gumbel_noise_f32", &sycl_gumbel_noise_f32, "gumbel_noise_f32");
    m.def("gumbel_noise_log", &sycl_gumbel_noise_log, "gumbel_noise_log");
    m.def("apply_logit_bitmask", &sycl_apply_logit_bitmask, "apply_logit_bitmask");
    m.def("flash_decode", &sycl_flash_decode, "flash_decode");

    // Unsupported entry points, raising with a clear message
    m.def("fused_sampler", &sycl_fused_sampler, "fused_sampler");
    m.def("apply_rep_pens", &sycl_apply_rep_pens, "apply_rep_pens");
    m.def("apply_pres_freq_pens", &sycl_apply_pres_freq_pens, "apply_pres_freq_pens");
    m.def("dry_penalty", &sycl_dry_penalty, "dry_penalty");
    m.def("adaptivep_gumbel_noise_f32", &sycl_adaptivep_gumbel_noise, "adaptivep_gumbel_noise_f32");
    m.def("batched_conv_rewind", &sycl_batched_conv_rewind, py::arg("jobs"), py::arg("device_index"));
    m.def("batched_scan_replay", &sycl_batched_scan_replay, py::arg("jobs"), py::arg("device_index"), py::arg("kind"),
          py::arg("num_k_heads"), py::arg("num_v_heads"), py::arg("k_head_dim"), py::arg("v_head_dim"));
    m.def("argmax_sample", &sycl_argmax_sample, "argmax_sample");
    m.def("gumbel_sample", &sycl_gumbel_sample, "gumbel_sample");
    m.def("pinned_cuda_view", &sycl_pinned_cuda_view, "pinned_cuda_view");
    m.def("exl3_gemm_num_kernel_shapes", &sycl_gemm_num_kernel_shapes, "exl3_gemm_num_kernel_shapes");
    m.def("exl3_gemv_int8_max_k", &sycl_gemv_int8_max_k, "exl3_gemv_int8_max_k");

    // attention gates, deinterleave and paged cache
    m.def("mul_sigmoid_", &sycl_mul_sigmoid, "mul_sigmoid_");
    m.def("mul_sigmoid_broadcast_", &sycl_mul_sigmoid_broadcast, "mul_sigmoid_broadcast_");
    m.def("mul_softplus_broadcast_", &sycl_mul_softplus_broadcast, "mul_softplus_broadcast_");
    m.def("deinterleave_qg", &sycl_deinterleave_qg, "deinterleave_qg");
    m.def("paged_kv_cache_update", &sycl_paged_kv_cache_update, "paged_kv_cache_update");
    m.def("device_get_mem_info", &sycl_device_get_mem_info, "device_get_mem_info");
    m.def("rope", &sycl_rope, "rope");

    // CUDA/TP/MoE/quantization surfaces this backend does not implement; the python side never
    // calls these on XPU (guards in the modules, or paths that require a CUDA/ROCm build)
    #define SYCL_STUB(name) m.def(name, []() -> void { \
        TORCH_CHECK(false, "exllamav3_sycl_ext: " name " is not implemented on the SYCL backend"); \
    }, name)
    SYCL_STUB("add_sigmoid_gate");
    SYCL_STUB("add_sigmoid_gate_proj");
    SYCL_STUB("bighead_attn_paged");
    SYCL_STUB("cache_rotate");
    SYCL_STUB("count_inf_nan");
    SYCL_STUB("count_match_tensor");
    m.def("cuda_causal_conv1d_update", &sycl_causal_conv1d_update, "cuda_causal_conv1d_update",
          py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("activation"),
          py::arg("history"), py::arg("incremental") = false);
    SYCL_STUB("cuda_device_get_attribute");
    SYCL_STUB("cuda_host_get_device_pointer");
    SYCL_STUB("cuda_host_register");
    SYCL_STUB("cuda_host_unregister");
    SYCL_STUB("cuda_recurrent_mamba2");
    SYCL_STUB("dequant_cache_cont");
    SYCL_STUB("dequant_cache_paged");
    SYCL_STUB("dequant_cache_paged_window");
    SYCL_STUB("det_quant_weight");
    SYCL_STUB("dflash2_dynconv");
    SYCL_STUB("dflash2_selector_walk");
    SYCL_STUB("dflash2_topk");
    SYCL_STUB("dsa_topk");
    SYCL_STUB("dsa_topk_merge_tiles");
    SYCL_STUB("dsa_topk_tile");
    SYCL_STUB("dspark_write_rows");
    SYCL_STUB("dsv4_compress");
    SYCL_STUB("dsv4_pool_quant_scatter");
    SYCL_STUB("dsv4_ring_append");
    SYCL_STUB("end_cpu_reduce_jobs");
    SYCL_STUB("exl3_moe");
    SYCL_STUB("exl3_moe_cpu_core_order");
    SYCL_STUB("exl3_moe_cpu_has_avx2");
    SYCL_STUB("exl3_moe_cpu_has_avx512_bw");
    SYCL_STUB("exl3_moe_cpu_has_avx512_vbmi");
    SYCL_STUB("exl3_moe_cpu_has_avx512_vnni");
    SYCL_STUB("exl3_moe_cpu_make_layer");
    SYCL_STUB("exl3_moe_cpu_planar_layout");
    SYCL_STUB("exl3_moe_cpu_set_memops");
    SYCL_STUB("exl3_moe_cpu_set_prof");
    SYCL_STUB("exl3_moe_cpu_swizzle_group");
    SYCL_STUB("exl3_moe_cpu_worker_run");
    SYCL_STUB("exl3_moe_flag_wait");
    SYCL_STUB("exl3_moe_flag_write");
    SYCL_STUB("exl3_moe_gather");
    SYCL_STUB("exl3_moe_max_concurrency");
    SYCL_STUB("gen_mrope_pos_ids");
    SYCL_STUB("gr_mix");
    SYCL_STUB("gr_mix_tiled");
    SYCL_STUB("gr_mix_tiled_slices");
    SYCL_STUB("had_paley");
    SYCL_STUB("had_paley2");
    SYCL_STUB("had_r_128_batch");
    SYCL_STUB("hc_apply");
    SYCL_STUB("hc_head");
    SYCL_STUB("hc_mix");
    SYCL_STUB("hc_mix_fused");
    SYCL_STUB("hc_mix_num_chunks");
    SYCL_STUB("hgemm_batched");
    SYCL_STUB("mamba2_dt_op");
    SYCL_STUB("moe_split_collect_add");
    SYCL_STUB("moe_split_issue");
    SYCL_STUB("moe_split_map");
    SYCL_STUB("moe_unswizzle_trellis");
    SYCL_STUB("ngram_dequant");
    SYCL_STUB("ngram_gather_cpu");
    SYCL_STUB("ngram_hash_cpu");
    SYCL_STUB("pack_signs");
    SYCL_STUB("pack_trellis");
    SYCL_STUB("pack_trellis_frac");
    SYCL_STUB("partial_strings_match");
    SYCL_STUB("pg_all_reduce");
    SYCL_STUB("pg_all_reduce_cpu");
    SYCL_STUB("pg_barrier");
    SYCL_STUB("pg_broadcast");
    SYCL_STUB("pg_broadcast_ll");
    SYCL_STUB("pg_gather");
    SYCL_STUB("pg_gather_small");
    SYCL_STUB("pg_init_context");
    SYCL_STUB("ple_forward_streams");
    SYCL_STUB("ple_gate");
    SYCL_STUB("quant_cache_cont");
    SYCL_STUB("quant_cache_paged");
    SYCL_STUB("quantize_tiles");
    SYCL_STUB("quantize_tiles_frac");
    SYCL_STUB("quantize_tiles_scratch");
    SYCL_STUB("reconstruct_batch");
    SYCL_STUB("reconstruct_had_batch");
    SYCL_STUB("routing_ds3_nogroup");
    SYCL_STUB("routing_gemm_det");
    SYCL_STUB("routing_sel_norm");
    SYCL_STUB("routing_std");
    SYCL_STUB("run_cpu_reduce_jobs");
    SYCL_STUB("silu_oai_mul");
    SYCL_STUB("stloader_close_file");
    SYCL_STUB("stloader_deferred_batch");
    SYCL_STUB("stloader_open_file");
    SYCL_STUB("stloader_read");
    SYCL_STUB("test_distribution");
    SYCL_STUB("unpack_trellis");
    SYCL_STUB("xielu");
    #undef SYCL_STUB


    // Attributes the python side checks
    m.attr("HAS_GR_MIX_TILED") = false;
    m.attr("HAS_DET_GEMM") = false;
    m.attr("HAS_HGEMM_F16ACC") = false;
    m.attr("HAS_DFLASH2") = false;
    m.attr("FUSED_SAMPLER_MAX_BLOCKS") = 0;
    m.attr("FUSED_SAMPLER_HIST_STRIDE") = 0;

    py::class_<BC_LinearFP16>(m, "BC_LinearFP16")
        .def(py::init<at::Tensor, c10::optional<at::Tensor>>())
        .def("run", &BC_LinearFP16::run)
        .def("run_gr", &BC_LinearFP16::run_gr);

    py::class_<BC_LinearEXL3>(m, "BC_LinearEXL3")
        .def(py::init<at::Tensor, at::Tensor, at::Tensor, float, c10::optional<at::Tensor>,
                      bool, bool, at::Tensor>(),
             py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("K"), py::arg("bias"),
             py::arg("mcg"), py::arg("mul1"), py::arg("xh"))
        .def("run", &BC_LinearEXL3::run)
        .def("run_gr", &BC_LinearEXL3::run_gr)
        .def("run_alloc", &BC_LinearEXL3::run_alloc);

    py::class_<BC_GatedRMSNorm>(m, "BC_GatedRMSNorm")
        .def(py::init<at::Tensor, double, double, int64_t, bool, int64_t>())
        .def("run", &BC_GatedRMSNorm::run);

    py::class_<ConvRewindJob>(m, "ConvRewindJob")
        .def(py::init<uintptr_t, uintptr_t, int, int, int>());
    py::class_<ScanReplayJob>(m, "ScanReplayJob")
        .def(py::init<uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, int, int>(),
             py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("prefix"), py::arg("tstride") = 1);

    m.def("batched_conv_rewind", &sycl_batched_conv_rewind, py::arg("jobs"), py::arg("device_index"));
    m.def("batched_scan_replay", &sycl_batched_scan_replay, py::arg("jobs"), py::arg("device_index"), py::arg("kind"),
          py::arg("num_k_heads"), py::arg("num_v_heads"), py::arg("k_head_dim"), py::arg("v_head_dim"));
}
