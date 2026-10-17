# Handoff: SYCL backend for exllamav3 (built, running, optimized)

*Rewritten Oct 16 2026, replacing the Oct 13 "dry implementation" handoff (that document described a
never-compiled port written without hardware; everything in it about mission, scope and next steps is
obsolete). Hand this document to a fresh model together with the repo at `/tabby_exl3`
(branch state: uncommitted files on disk — the repo is NOT a git repo). The full optimization history
lives in `/tabby_exl3/OPTIMIZATION_LOG.md`; this document is the orientation layer on top of it.*

---

## 1. Current state (one paragraph)

The SYCL backend (`exllamav3/exllamav3_ext/sycl/`, pybind module `exllamav3_sycl_ext`) is **compiled,
loaded and serving models through TabbyAPI** on an Intel Arc A770 16 GB. 29 optimization changes are
in (see `OPTIMIZATION_LOG.md`): a dense-W decode path, pointer-table caching, sync elimination, an
attention rework, and a KS-split tile-once packed GEMV. Current steady-state decode: **2B ~33 tok/s,
4B ~23 tok/s, 27B (2bpw, all-packed) ~6.5 tok/s** — all verified coherent through the TabbyAPI
endpoint. The mission is now **continued optimization under strict correctness and VRAM-safety
constraints**, not bring-up.

## 2. Environment

| Item | Value |
|---|---|
| GPU | Intel Arc A770 16GB (`level_zero:gpu:0`, `dg2`), 512 GB/s theoretical bandwidth |
| CPU / RAM | AMD Ryzen 5 PRO 4650G, 12 threads; **only 15 GB system RAM** (see gotchas) |
| oneAPI | 2026.1 (`/opt/intel/oneapi`): icpx, MKL, VTune, ocloc |
| torch | 2.14.1+xpu in `/tabby_exl3/venv` (venv python symlink repaired to `/usr/bin/python3.12`) |
| Repo | `/tabby_exl3` (TabbyAPI + the exllamav3 fork with the SYCL backend) |
| Models | `/models/Qwen3.5-2B-exl3`, `/models/Qwen3.5-4B-heretic-exl3-4.0bpw`, `/models/Qwen3.8-27B-exl3-SC_2.00bpw_H3_V3`, all symlinked into `models_exl3/` |
| Server | TabbyAPI on port 5000, `inline_model_loading: true`, `cache_size: 32768` (+ `use_as_default`) |

## 3. Hard constraints & user decisions (do not relitigate)

| Constraint | Detail |
|---|---|
| **VRAM safety first** | The 15 GiB card has no headroom for near-OOM experiments; driving free VRAM → 0 spills to shared memory and **hangs/crashes the whole host**. All long-context measurements use `cache_size 32768` and bounded token counts (~2 min runs) |
| **Host RAM is 15 GB** | `torch.profiler` with XPU activity OOM-killed the host at trace dump. For the 27B use **cProfile**, never the torch profiler |
| Verification | Through the actual TabbyAPI endpoint (`/v1/completions`, greedy, bounded runs). Test scripts: `/tmp/opencode/ep_test.py`, `seq_test.py`, `speed_test.py` (NOTE: `speed_test.py <ntok> <model>` — the model arg was added Oct 15; earlier versions hardcoded the 4B and silently unloaded whatever was loaded) |
| Keep tabby up | The server stays running and untouched unless the user grants a reload/restart. `pkill -f "[m]ain\.py"` is safe (bracket trick); `pkill -f "main\.py"` self-matches in compound commands |
| Server start | `cd /tabby_exl3 && nohup ./venv/bin/python -u main.py > /tmp/opencode/tabby.log 2>&1 &` — **keep this command in its own shell call** (a non-zero exit in the same command once killed the server) |
| Extension rebuilds | Automatic on `import exllamav3` (~140 s) when any `sycl/` source changes. **A running server keeps the old .so mapped — it must be restarted to pick up a rebuild.** (This bit us once: a "no improvement" measurement was the old kernel.) |
| `hf` CLI | stale shebang — run as `./venv/bin/python ./venv/bin/hf ...` |

## 4. Backend architecture (what exists now)

```
exllamav3/exllamav3/exllamav3_ext/sycl/     # the backend (built by util/sycl_build.py, icpx -fsycl)
  gemv.cpp        # packed decode GEMV: per-output kernel (m>1) + split-K kernel (m==1, change 20)
  mgemm.cpp       # exl3_mgemm pointer-table loop + dense mode + PtrTableCache (D2H once per table)
  hgemm.cpp       # oneMKL column_major::gemm (fp32-out via half scratch + upcast)
  reconstruct.cpp # per-weight scatter + fused 128x128 hadamard (local mem butterfly)
  hadamard.cpp    # had_r_128 fp16/fp32 (pre/post scale)
  gdn.cpp         # gated_delta_net_fused_op(_2), delta-rule scan (head-dim 128)
  elementwise.cpp # rms_norm(+res), gated_rms_norm, add, softcap, act_muls, paged KV update, ...
  rope.cpp, gumbel.cpp, bindings.cpp, exl3_decode.h, exl3_compat.h, sycl_kernels.h
```

`exllamav3/exllamav3/sycl/README.md` has the implemented-vs-stubbed table and exact build flags.

### 4.1 The decode paths (the load-bearing design)

- **Dense-W path (change 9, the whale)**: `LinearEXL3.ensure_dense()` dequantizes the trellis into a
  persistent fp16 W once; `dense_forward` = had_r_128 → `hgemm_recon` (oneMKL) → had_r_128.
  **Bit-identical to the packed GEMV** (max diff 0.000000). ~23x faster than the packed GEMV per
  2048×2048. `mgemm.cpp` has a dense mode (per-source had → oneMKL gemm per tensor/slice → had in
  place, pointer tables aimed at the dense W).
- **Packed GEMV (change 20)**: for models too big to densify (the 27B), decode uses
  `launch_gemv_splitk`: one work-group of 64 lanes per output element, lanes striding over **k-tiles**
  (consecutive lanes → consecutive 64-byte tiles = one coalesced pass, each tile read exactly once),
  16 weights decoded per tile per lane, group tree reduction. Dispatched for **m==1 only**; the old
  per-output kernel stays for m>1.
- **Dense budget (changes 10/12/15/16)**: global `_dense_budget` = free VRAM at first
  `ensure_dense` − context-aware safety (1.5 GiB + ¼ KV-cache bytes). Densification happens in a
  post-load walk (`Model.load_gen` → `try_dense()`), never during load. Reset on `Model.unload()`.
  `EXL3_SYCL_DENSE=0` disables.
- **PtrTableCache (changes 8/11)**: mgemm's blocking `wait_and_throw` memcpys happen once per pointer
  table (keyed data_ptr + length), cleared via `exl3_sycl_ptr_cache_clear()` on unload/load_gen.

### 4.2 Attention (changes 6/13/14/19)

Decode (q_len==1) runs a **plain GQA matmul attention** (grouped view → matmul → fp32 softmax →
matmul; `unsqueeze(2)` on BOTH operands or heads 1-3 of each group read the wrong KV head). Prefill
keeps SDPA with CausalBias. **Never use SDPA with a bool mask or maskless on the decode path** — the
Intel oneAPI SDPA is fragile at these shapes (intermittent `partition compile failed`, wrong results,
and the CausalBias path leaked ~21 MB/token of workspace → decode cliff at ~1.4k tokens). Host page/
length metadata is threaded through `AttnArgs` (`block_table_host`/`cache_seqlens_host`) so the
fallback never calls `.item()`.

### 4.3 Env flags set for XPU (`ext.py`, before module imports)

`EXL3_BC_GDN=0` (the graph-captured GDN decode is CUDA-only — **and** its gate accidentally protects
XPU from the slower qkv+z bundle, see change 17), `EXL3_FUSED_SAMPLER=0`. CUDA graphs off, TP off,
attention → torch/matmul dispatch.

## 5. Performance state (Oct 16)

**Decode-ALU bound (change 30, THE wall for the 27B)**: every packed GEMV shape runs ~60-70 GB/s
cold with the mul1 decode, and a decode-stub variant of the same kernel hits **430 GB/s** — the
GEMV is bound by the mul1 codebook's byte-sum emulation (~13 ops/weight, ~4.3T ops/s), NOT by DRAM
(coalesced peak 355 GB/s), NOT by layout/pipelining/occupancy (all benched as washes). The whole 27B
is `"codebook": "mul1"` (config.json). Measured dead ends: 8-op SWAR (Gen INT mul slow), 64K weight
LUT (scattered gathers 12 GB/s), staged decode, pow2-AND, n-major repack, manual pipelining. HW dp4a
exists on Gen but DPC++ only exposes it via ESIMD — **change 31 (Oct 17) decoded the real semantics**:
`dp4a<T1>(src0, src1, src2)` computes **src0 + dot(src1, src2) over UNSIGNED bytes**, so
`dp4a(0x6400, x*M, 0x01010101) = 0x6400 + byte_sum(x*M)` — the ~11-op byte-sum collapses to ONE
instruction, verified exact on ~140 cases (test_dp4a8..19; ESIMD-kernel scalar access needs the
accessor + `esimd::scalar_store` path; raw-pointer scalar writes are what segfaulted before). BUT:
(1) **ESIMD intrinsics + local_accessor (SLM) in the same regular kernel = broken codegen** (silent
all-zero output, no exception); (2) with the lout replaced by global-memory atomics the dp4a GEMV is
CORRECT but **27x slower than the vec kernel** (8.83 vs 0.330 ms; decisive control: the same kernel
with the scalar dq4 decode + the same atomic epilogue = 0.496 ms) — the ESIMD usage triggers
catastrophic codegen (spills or scalar fallback) in this register-heavy kernel; in simple kernels the
intrinsic itself is fast (0.122 ms/4M ops vs 0.079 shift/add). Bench variants: 39 = esimd (correct,
dead end), 40 = atomi (scalar + atomic epilogue, the lout-free control). **The dp4a byte-sum collapse
is REAL but only reachable via a full SYCL_ESIMD_KERNEL GEMV rewrite** (proper ESIMD SLM ops + native
dp4a) — that is the only kernel path past ~68 GB/s. Densification is VRAM-blocked (~2 GiB budget).

**Change 32 (Oct 17) — the rewrite LANDED in production.** `launch_gemv_esimd<K, KS>` in gemv.cpp,
a proper `SYCL_ESIMD_KERNEL` tile-once GEMV (mul1/cb=2 only, dispatched for the MT2 shapes at
KS=16): each work item = 8 consecutive virtual lanes (simd<8>), the native dp4a for the byte-sum,
ONE 32-bit funnel decomposition + constant shifts for the 4 windows, the cross-lane reduction in
registers, disjoint scratch writes + the split-K reduce. Four IGC landmines mapped empirically
(test_esimd20..26, bench variants 41-43): (1) **the SLM is broken in nd_range ESIMD kernels** (works
in single_task only) → register-only epilogue; (2) **strided selects return wrong values and
`acc += select` silently kills the rest of the kernel** → scalar element access only;
(3) **`sycl::bit_cast` does not compile on `simd`** → the half bit-pattern 0x6400+bs handled
arithmetically (half_val(0x6400+bs) = 1024+bs exactly, so `float(dp4a_sum) − 24576` IS the half's
value, then round-to-half + promote — bit-matches the scalar decode); (4) the 4 dq4 windows are one
64-bit value shifted by consecutive amounts → `(a << (31−s2)) << 1 | (b >> s2)` + constant shifts
(0/128 vs the scalar fshift). KSPLIT: 16 > 32 > 64. Dispatch gate: k ≥ 4096 AND n ≥ 4096 (the 2B's
k=2048 shapes measured ~3% slower ungated; cb 0/1 keep the vec — the lop3 decode needs a uint16→half
bit-reinterpret on simd, which miscompiles). Bench27b cold production shapes, exact correctness:
down 0.181 → **0.151 ms (1.20x)**, gate/up 0.329 → **0.285 (1.15x)**, qkv 0.089 → 0.083, GDN 0.141 →
0.130.

| Model | Decode | Path |
|---|---|---|
| 2B (4bpw) | **39.5 tok/s** (was ~33) | densified, dense-W + oneMKL + custom dense GEMV (down_proj k=6144) |
| 4B heretic (4bpw, cache 32k) | **23.7 tok/s** | ~fully densified (200/201 linears, 7.13 GB dense; only the K=8 lm_head packed), oneMKL GEMVs + custom dense GEMV (down_proj k=9216), flash-decode attention |
| 27B (2bpw SC, cache 32k) | **8.7–8.8 tok/s** (was 0.9–1.0 → 2.0 → 4.2 → 6.5 → 7.4 → 7.8 → 8.3 → 8.8) | all-packed, ESIMD tile-once GEMV (8 lanes/work item, native dp4a decode, KS=16, gated k,n ≥ 4096; also the lm_head n=248320 via the enlarged scratch) + fused split-K reduce+H128 (1 launch/linear) + MT2+vec for cb 0/1 and small shapes + GDN scan V_SPLIT=1 + register-cached state slice + flash-decode attention |

## MTP (speculative decoding) status (change 27)

Both the 2B and 27B checkpoints carry a complete MTP head (39 `mtp.*` keys); the 4B has none. The
SYCL backend's GDN rewind machinery (draft rejection) is now ported (`batched_conv_rewind`,
`batched_scan_replay`, real job structs in `sycl/gdn.cpp` + `sycl/bindings.cpp`), so MTP runs.

| Model | MTP (draft_num_tokens=2) | Verdict |
|---|---|---|
| 2B | **53.4 vs 37.0 tok/s (1.44x)**, 70% acceptance | viable — enable via `draft_model: {"draft_mode": "mtp", "draft_num_tokens": 2}` (dt=4 drops to 46.5/42%) |
| 27B | 2.3 vs 6.5 tok/s (3x slower), 59% acceptance | NOT viable — confirmed twice (changes 29 + 30). The m3v kernel (change 30: the change-29 vectorized-A mapping for m=3, 2 vector A-loads/token vs the old m3's 12 scalar gathers) is **correct and 1.29x faster than the old m3 kernel** (0.62 vs 0.80 ms on the down shape, 35.8 vs 27.8 GB/s) — but even at 1.9x the m=1 GEMV per round, the verify ≈ 300-380 ms/round for 2.18 tokens ≈ 6-7.3 tok/s vs the 7.7 decode: the non-GEMV per-round overhead (draft pass, SDPA-at-m>1, the rest) dominates. The m3v kernel is in bench27b.cpp (variant 38) alongside the old m3, not ported. With the decode-ALU bound (change 30) the packed m=3 GEMM cannot amortize enough — MTP would close only if the decode itself got ~2-3x faster (hardware dp4a / ESIMD rewrite). The 2B wins only because it is fully dense — its verify runs the oneMKL GEMM at ~470 GB/s |
| 4B | — | no MTP head in the checkpoint |

### 5.1 The 27B after changes 23+25 (tile-once + KS-split GEMV)

- The 27B (hidden 5120, 64 layers, GQA 24/4, GDN 16k×48v×128, bf16 embed 2.5 GB, quantized lm_head
  5120→248320) must run the packed path. Free VRAM after load + 32k cache: ~4.1 GiB; the dense W
  would be ~54 GB.
- **The earlier "dequant-ALU-bound" analysis measured the wrong bound.** Microbenchmarking
  (bench27.cpp, exact shapes K=2/mul1 k=5120) showed the mul/dp4a/hfma decode chain is ~8% of the
  kernel time (dropping it changed nothing); the bound was the split-K kernel's work-group layout:
  one work-group per output COLUMN re-read every subtile's tile stream 16x through L2. The
  tile-once kernel (change 23) reads each tile once per n-subtile work-group and scatter-accumulates
  a local 16×16 tile: **3.7x cold** on the lm_head shape (127 → 469 GB/s unique), 1.4x on the
  hidden-layer shape with 128 lanes (87 GB/s). End to end: 2.0 → 4.17–4.21 tok/s.
- **Change 25: the layer shapes are parallelism-starved, not traffic-bound.** A production-pattern
  bench (bench27b.cpp: round-robin over 10 cold 22 MB trellis buffers, exactly like production where
  every layer's weights are evicted between tokens) reproduced the real ~40 GB/s: the
  one-group-per-subtile layout gives only 320 subtiles × 64 lanes = 20K threads on the
  down_proj/out_proj shapes. Fix: split the k-tiles across KS=8 work-groups per subtile, disjoint
  fp32 partial slices in a fixed 1.3 MB scratch (no atomics, no memset; the production queue is
  in-order so partial → reduce and consecutive calls are safe), plus a tiny reduce kernel. Measured
  on the exact cold production shapes: down_proj **2.2x** (0.989 → 0.454 ms), out_proj **2x**
  (0.354 → 0.179), mlp gate/up 1.4x (0.566 → 0.404), GDN bundle 1.25x. End to end: 4.2 → 6.5 tok/s.
- Dispatch: serial for the small short-k shapes (k=2560 n=1024 measured serial 2x faster — the
  tile-once layout doesn't fill the machine there); KS-split tile-once when `ngroups*64 < 163840`;
  unsplit tile-once otherwise (64 lanes when ntiles ≥ 4096 — the lm_head has 993K lanes and stays
  unsplit; 128 lanes below).
- Re-profile after the split-K: the 27B GPU wall dropped 196 → **104 ms/token** (6.5 tok/s wall,
  ~50 ms host overlap). KS=16/32 plateau at ~55 GB/s on the layer shapes (the split trades DRAM
  pattern locality for parallelism — the groups are no longer in k-lockstep, so the collective
  read is scattered; the lm_head's 470 GB/s keeps k-lockstep with 9.5k groups). The GEMV lever is
  at its ceiling at ~55 GB/s effective for the split shapes.
- Next levers, ranked: non-GEMV GPU ~25-30 ms/token (hads/GDN/norms/conv — note the 4B profile
  shows the GDN delta rule partly launching through the **triton Intel backend**, 24 launches/token,
  0.9 ms); host-side per-op overhead (the 4B: ~320 torch.empty + 224 had + 112 oneMKL gemm +
  ~40 mgemm calls per token; mgemm measures 139 µs/call host-side); fused had+gemv for had_in
  (gate and up share the same input — the had_in is computed twice per MLP layer).

### 5.2 4B remaining roadmap (from the log, ranked)

### 5.2 4B remaining roadmap (from the log, ranked)

1. Custom SYCL flash-decode kernel for head_dim 256 (decode attention is now ~3 launches/token/layer).
2. Fused/batched had+gemm+had for the dense path (~114 hgemm_recon + ~227 had launches/token).
3. `Linear.forward` fast path (163 calls/token through wrapper checks; ~2–4 ms of pure Python).
4. ~~GDN small-projection batching~~ — measured dead end on XPU (the qkv+z bundle is slower for
   gcd-width 4096; b+a merge in change 18 already took the ~2%).

## 6. Correctness invariants (each was a real bug — do not regress)

1. **Stale pointer tables**: any per-call-built c_ptrs table can hit `PtrTableCache` (keyed on
   data_ptr + length) when the allocator recycles the same block — layers 2+ get layer 1's stale
   output pointers → intermittent garbage that flips between requests. Prealloc outputs and build
   tables **once** (changes 11/17/19; change 19 restricted attention sliced prefill to m==1 for
   exactly this reason).
2. **Dense sliced mode requires fp16** outputs and an fp16 C carrier; fp32-out bundles stay packed.
3. **Model swaps**: clear `PtrTableCache` and reset `_dense_budget` on unload, or the model loaded
   after a swap runs on the previous model's pointers / stays all-packed.
4. **mul1 decode**: the uint16→half step is a **bit copy** (never a numeric conversion) and the
   dp4a byte-sum stays integer.
5. **oneMKL** wants A/B/C the same dtype; fp32-out goes through a half scratch + upcast.
6. **Verification gate**: after any change, `ep_test.py <model>` (×3 for attention-adjacent changes)
   + `seq_test.py`; zero `!!!!`/garbage expected. Known-benign: the 4B/27B are reasoning models —
   small `max_tokens` returns `content: null` with output in `reasoning_content`.

## 7. Gotchas (machine + harness, learned the hard way)

- **15 GB host RAM**: no torch.profiler with XPU activity on the 27B (OOM-killed the host). cProfile
  + manual per-stage sync timing instead.
- **Running a second model instance alongside the server** time-slices the A770 (33 → 18 tok/s on
  both sides) and VRAM oversubscription spills to shared memory (84–257 ms/tok) — never benchmark
  against a loaded server.
- Greedy "looping" is model behavior (temperature=0 repetition; raw completions skip the chat
  template) — EOS detection works. Mitigations: `loop_detect_window` → `stop_on_loop`, temp > 0,
  rep penalty.
- The first request after a server start pays SYCL JIT for every kernel specialization (up to
  ~340 s TTFT cold; AOT via ocloc is supported but this machine JIT-compiles).
- "No improvement" after a kernel change usually means **the server still has the old .so mapped** —
  restart it.
- Greedy benchmark traps: `speed_test.py`'s warmup request passes `"model": <name>` → an inline load
  that **unloads whatever the user had loaded**. Check `/tmp/opencode/tabby.log` for
  `Loading model` lines before trusting any endpoint measurement.

## 8. Where to work next

1. Read `OPTIMIZATION_LOG.md` (changes 1–20 + profiles) — it has the measurement history and the
   dead ends (SDPA maskless, qkvz bundle, bool-mask attention).
2. The 27B split-K GEMV is in `sycl/gemv.cpp` (`launch_gemv_splitk`, dispatched in
   `gemv_dispatch()` for m==1; the raw2/mgemm sites share it). A SIMD16/vectorized dequant rewrite
   is the highest-value open item.
3. Correctness screening: `/tmp/opencode/*.py` scripts (ep_test, seq_test, bisect_test, gdn_timing);
   they assume the server on port 5000 and take a model name where relevant.
