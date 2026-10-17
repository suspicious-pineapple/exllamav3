# SYCL backend (`exllamav3_sycl_ext`)

A hand-written SYCL 2020 port of the exllamav3 EXL3 decode path, targeting Intel GPUs through
DPC++ (`icpx -fsycl`) and the torch XPU runtime. It is a **separate pybind module**
(`exllamav3_sycl_ext`) built with a custom build path (not `torch.utils.cpp_extension`), and is
imported by `exllamav3/ext.py` under the `exllamav3_ext` name the python surface expects.

The kernels run on the current torch XPU stream (`c10::xpu::getCurrentXPUStream().queue()`, an
in-order queue), so they stay ordered with torch ops and the oneMKL GEMMs.

## Scope

Follows HANDOFF_SYCL.md §2. Quantization, the EXL3 tensor-core GEMM, CUDA graphs, tensor parallel
and MoE are out of scope. Inference = `reconstruct` + `hgemm` (prefill) and scalar-dot `GEMV`
(decode); attention is torch SDPA on XPU.

## Build

```bash
# via the normal import path (rebuilds when any sycl/ source changes):
import exllamav3   # -> util/sycl_build.py -> icpx -fsycl

# manually:
icpx -std=c++20 -fsycl -fPIC -O3 \
     -DTORCH_EXTENSION_NAME=exllamav3_sycl_ext -DTORCH_API_INCLUDE_EXTENSION_H \
     -D_GLIBCXX_USE_CXX11_ABI=1 \
     -I<torch>/include -I<torch>/include/torch/csrc/api/include \
     -I<oneapi>/compiler/<v>/include -I/usr/include/python3.12 \
     -I exllamav3_ext/sycl -fsycl-targets=spir64 \
     -c exllamav3_ext/sycl/*.cpp

# link (the -fsycl on the LINK line performs the device-link step; omitting it links but every
# kernel launch segfaults):
icpx -fsycl -fsycl-targets=spir64 -shared \
     -L<torch>/lib -ltorch -ltorch_cpu -ltorch_python -lc10 -lc10_xpu -ltorch_xpu \
     -L<oneapi>/compiler/<v>/lib -lsycl \
     -L<oneapi>/mkl/<v>/lib -lmkl_sycl -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential \
     *.o -o exllamav3_sycl_ext.so
```

Notes:
- **C++20** (torch 2.14 headers need `std::strong_ordering`).
- AOT device images (`spir64_gen` + the `-device` target) require `ocloc`; when absent the build
  falls back to `-fsycl-targets=spir64` and JIT-compiles on first launch (works, first launch is
  slow). See `util/sycl_build.py::_sycl_target_flags`.
- dGPUs are detected by name (Arc A/B series, datacenter Max); iGPUs (Arc 140T/130T/110T, Iris,
  UHD, bare "Arc Graphics") map to their own AOT target (`arl-h`/`mtl-h`/`lnl-m`/`ptl-*`) since a
  dg2/bmg/pvc image will not load on iGPU silicon. `SYCL_TARGET_DEVICE=<ocloc name>` (e.g.
  `arl-h`) overrides detection; without `ocloc` everything JIT-compiles via `spir64`.
- oneMKL is linked for `hgemm` (plain fp16 GEMM) only.

## Implemented

### EXL3 decode
| Function | File | Notes |
|---|---|---|
| `reconstruct` / `reconstruct_slice` / `reconstruct_had_slice` | `reconstruct.cpp` | per-weight scatter using the §4.2 mapping; fused 128×128 hadamard in local memory |
| `exl3_gemm` / `exl3_gemv` | `gemv.cpp` | three launches: input Hadamard → GEMV (decode-on-the-fly) → output Hadamard |
| `exl3_mgemm` | `mgemm.cpp` | pointer-table loop, incl. the **sliced** mode (`size_n_list`/`c_ptrs`/`n_stride_list`/`had_src_list`) used by the qkv projection path |
| `had_r_128` | `hadamard.cpp` | pre-scale / post-scale butterfly, local-memory Sylvester network, fp16 & fp32 |
| `hgemm` / `hgemm_recon` | `hgemm.cpp` | oneMKL `column_major::gemm`; fp32-out via a half scratch + upcast |

### Elementwise / norms / gates
`rms_norm` (plain + `res_in` + `res_out`), `gated_rms_norm`, `add`, `softcap`,
`silu_mul`/`gelu_mul`/`relu2_mul`/`relu_mul`, `mul_sigmoid_`, `mul_sigmoid_broadcast_`,
`mul_softplus_broadcast_`, `deinterleave_qg`, `paged_kv_cache_update`, `apply_logit_bitmask` —
`elementwise.cpp` (all dtypes dispatched on x/y/w; the residual dtype of `rms_norm_res_in` follows
the residual tensor's actual dtype).

### RoPE / GDN / misc
`rope` (NEOX + GPTJ, per-head q/k norm, scalar/positions/position_ids, Llama-4 scaling) —
`rope.cpp`; `gated_delta_net_fused_op(_2)` + `cuda_recurrent_gated_delta_rule` (the delta-rule scan,
head-dim 128) — `gdn.cpp`; `gumbel_noise_f16/f32/log` — `gumbel.cpp`; `device_get_mem_info`
(free/total via the Intel `free_memory` device extension, 0 = unsupported → python falls back).

### BC handles
`BC_LinearFP16`, `BC_LinearEXL3`, `BC_GatedRMSNorm` (plus `ConvRewindJob`/`ScanReplayJob`
placeholders). `run_alloc` mirrors the CUDA glue; the fp32-output BC scratch is treated as raw half
storage for the input Hadamard, matching the CUDA kernel.

## Stubbed (raise with a clear message)

The following are bound and **raise** rather than silently no-op, so a missed guard fails loudly.
The python side never calls them on XPU (guarded in the modules, or paths that require a CUDA/ROCm
build / CUDA graphs / TP / MoE / quantization).

- **Sampler accelerators** (torch fallback is used instead): `fused_sampler`, `argmax_sample`,
  `gumbel_sample`, `adaptivep_gumbel_noise_f32`, `apply_rep_pens`, `apply_pres_freq_pens`,
  `dry_penalty`, `batched_conv_rewind`, `batched_scan_replay`.
- **Quantization / encode side**: `quantize_tiles(_frac/_scratch)`, `pack_trellis(_frac)`,
  `pack_signs`, `unpack_trellis`, `det_quant_weight`, `dequant_cache_*`, `quant_cache_*`,
  `ngram_*`, `had_paley(_2)`, `had_r_128_batch`.
- **MoE**: `exl3_moe*`, `moe_split_*`, `moe_unswizzle_trellis`.
- **Tensor parallel**: `pg_*`, `run_cpu_reduce_jobs`, `end_cpu_reduce_jobs`, `gen_mrope_pos_ids`.
- **CUDA-only kernels** (other archs / dsv4 / dsa / hc / ple / routing / stloader / bighead):
  `bighead_attn_paged`, `cuda_causal_conv1d_update`, `cuda_recurrent_mamba2`, `cuda_host_*`,
  `cuda_device_get_attribute`, `dsa_*`, `dsv4_*`, `dspark_write_rows`, `hc_*`, `ple_*`,
  `routing_*`, `stloader_*`, `gr_mix*`, `hgemm_batched`, `mamba2_dt_op`, `cache_rotate`,
  `xielu`, `silu_oai_mul`, `add_sigmoid_gate(_proj)`, `fa75_fwd`, `dflash2_*`, `count_*`,
  `partial_strings_match`, `test_distribution`, `pinned_cuda_view`.

## Known limitations / deviations

- **GEMV decode redundancy**: one scalar-dot work-item per output, decoding the whole `k` dim; fine
  for the `m ≤ 144` decode regime, not for prefill (which uses `hgemm`).
- **Butterfly rounding**: the SYCL Hadamard runs the butterfly in fp32 (the CUDA half path rounds
  to fp16 between stages); a precision-only difference.
- **Sliced `exl3_mgemm`** does one input Hadamard per source and a strided GEMV + output Hadamard
  per slice (no cooperative-kernel fusion); semantically identical to the CUDA kernel.
- **Attention** is torch SDPA on XPU (`attention_fn/dispatch.py` filters the Triton/xformers/
  bighead backends when `XPU`); no ext attention kernel.
- **XPU device pointers** live above 2^63, so pointer tables are built two's-complement int64 via
  `util/tensor.py::ptr_tensor` and read back as `uint64_t` on the device (bit pattern preserved).
- **`device_get_mem_info` free query** returns 0 on this driver; `util/memory.py` then approximates
  free memory as `total - torch.xpu.memory_reserved`.
- The half-integer bitrate decode (`dq_half`, mul1 only) is implemented but lightly exercised.
