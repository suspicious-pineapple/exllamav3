from __future__ import annotations
import torch
from ...model.config import Config
from .exl3_lib.quantize import frac_k, preapply_had_l, preapply_had_r, had_k, had_n
from ...ext import exllamav3_ext as ext
from ...util.tensor import g_tensor_cache
import os
from ...util import profile_opt

AUTO_RECONSTRUCT_THRESHOLD = 144
MAX_RECONSTRUCT_SLICE_N = 32768
RECONSTRUCT_SLICE_GRANULARITY_N = 128

no_fused_reconstruct = os.environ.get("EXL3_NO_FUSED_RECONSTRUCT", "0") != "0"

# Pre-dequantize each EXL3 linear's trellis into a persistent dense fp16 weight and run every
# forward through had -> oneMKL gemm -> had instead of the decode-on-the-fly GEMV kernels. The
# SYCL GEMV is compute/latency-bound at decode shapes (~1.1 ms per 2048x2048) while oneMKL runs
# the same GEMM in ~0.05 ms, and the weights are static, so the dequant (one reconstruct launch
# per linear) pays for itself immediately. Memory cost: out_features x in_features x 2 bytes per
# linear (~4 GB for a 2B model). Engages on XPU only (EXL3_SYCL_DENSE=0 to disable)
sycl_dense = os.environ.get("EXL3_SYCL_DENSE", "1") != "0"

# Global budget for the dense weights: measured once (free VRAM at the first ensure_dense call,
# which happens during the load warmup) minus a safety margin for cache/activation transients.
# Each build subtracts its size, so the dense dequant never overcommits on models whose dense
# form would not fit (they fall back to the packed paths, per linear)
_dense_budget = None
_DENSE_SAFETY = 1536 << 20  # 1.5 GiB

def reset_dense_budget():
    """Forget the measured dense-weight budget. Called on model unload: the unloaded model's
    dense weights are freed, so a stale (exhausted) budget would keep every later model in
    the process all-packed instead of re-measuring free VRAM for the next load."""
    global _dense_budget
    _dense_budget = None

def _context_transient_bytes(config) -> int:
    """Runtime transients that grow with the KV context (attention intermediates, the allocator's
    working-set ratchet over the per-token buffers), estimated as a fraction of the KV cache at
    the model's max context. A dense weight that consumes the headroom these need is what turns
    a long generation into a shared-memory spill (a 5-10x decode cliff past ~1k tokens) once free
    VRAM reaches zero. 0 when the geometry can't be read (the dense budget then keeps its base
    safety margin only)."""
    try:
        d = getattr(config, "config_dict", None) or {}
        t = d.get("text_config") or d
        layers = t.get("num_hidden_layers")
        kv_heads = t.get("num_key_value_heads")
        head_dim = t.get("head_dim")
        if head_dim is None:
            head_dim = (t.get("hidden_size") or 0) // max(t.get("num_attention_heads") or 1, 1)
        max_pos = t.get("max_position_embeddings")
        if not (layers and kv_heads and head_dim and max_pos):
            return 0
        # Full-attention layers only; a hybrid model's recurrent layers keep no KV cache
        layer_types = t.get("layer_types")
        if layer_types:
            n_full = sum(1 for lt in layer_types if lt == "full_attention")
        else:
            n_full = layers
        kv_bytes = n_full * 2 * kv_heads * head_dim * 2 * max_pos
        return kv_bytes // 4
    except Exception:
        return 0

def _dense_safety(config) -> int:
    return _DENSE_SAFETY + _context_transient_bytes(config)

def _sycl_dense_ok(x: torch.device | None = None) -> bool:
    if not sycl_dense:
        return False
    try:
        if torch.version.cuda or torch.version.hip:
            return False
        if x is not None:
            return x.type == "xpu"
        return bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
    except Exception:
        return False

class LinearEXL3:

    quant_type: str = "exl3"

    def __init__(
        self,
        config: Config | None,
        in_features: int,
        out_features: int,
        scale: torch.Tensor | None = None,
        su: torch.Tensor | None = None,
        sv: torch.Tensor | None = None,
        suh: torch.Tensor | None = None,
        svh: torch.Tensor | None = None,
        trellis: torch.Tensor | None = None,
        mcg: torch.Tensor | None = None,
        mul1: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        transformers_fix: bool = False,
        key: str | None = None
    ):
        assert scale is None, "scale is no longer used"
        assert su is not None or suh is not None, "either su (packed) or suh (unpacked) is required"
        assert sv is not None or svh is not None, "either sv (packed) or svh (unpacked) is required"
        assert trellis is not None, "trellis is required"
        if su is not None: assert su.dtype == torch.int16, "su is wrong datatype"
        if sv is not None: assert sv.dtype == torch.int16, "sv is wrong datatype"
        if suh is not None: assert suh.dtype == torch.half, "suh is wrong datatype"
        if svh is not None: assert svh.dtype == torch.half, "svh is wrong datatype"
        assert trellis.dtype == torch.int16, "trellis is wrong datatype"
        assert len(trellis.shape) == 3, "trellis must have dim = 3"

        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)

        # Not a Module subclass, so the config-or-NullConfig default doesn't apply here; TP imports pass
        # config=None and forward() reads config.infer_params
        if config is None:
            from ...model.config import NullConfig
            config = NullConfig()
        self.config = config
        self.transformers_fix = transformers_fix
        self.key = key

        # self.scale = scale.item()
        self.su = None
        self.sv = None
        self.suh = suh if suh is not None else self.unpack_bf(su)
        self.svh = svh if svh is not None else self.unpack_bf(sv)
        self.trellis = trellis
        # Bitrate from the tile width: 16 * K uint16 per tile, 16 * K + 8 for half-integer K (mul1 only)
        K = trellis.shape[-1] / 16
        self.K = int(K) if float(K).is_integer() else K
        self.frac = frac_k(self.K)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.swap_device = None
        self.out_dtype = out_dtype
        self.default_out_dtype = out_dtype or torch.half

        self.mcg_tensor = mcg
        self.mul1_tensor = mul1
        self.mcg = self.mcg_tensor is not None
        self.mul1 = self.mul1_tensor is not None
        self._dense_w = None
        self._no_recon = None          # lazy: config.infer_params may not exist yet
        self._dense_static_ok = None   # lazy: backend/dims memo for ensure_dense
        self._dense_static_dev = None  # device the memo was computed for
        assert self.frac is None or self.mul1, f"{key}: half-integer bitrate {self.K} requires the mul1 codebook"

        self._fused_reconstruct = None
        self.bsz1_xh_args = (self.trellis.device, (1, self.in_features), self.out_dtype)
        # K is the bitrate (int, or float for the half-integer rates); the C++ side decomposes it
        self.bc = ext.BC_LinearEXL3(
            self.trellis,
            self.suh,
            self.svh,
            self.K,
            self.bias,
            self.mcg,
            self.mul1,
            g_tensor_cache.get(*self.bsz1_xh_args)
        )


    def unload(self):
        # Free the dense weight deterministically and forget the budget it consumed, so a model
        # swap re-measures free VRAM for the next load instead of inheriting an exhausted budget
        # NOTE: _dense_w is kept here on purpose — the driver-level free happens at model drop;
        # dropping it during unload interacted badly with the autosplit loader's OOM fallback
        reset_dense_budget()
        # g_tensor_cache.drop(*self.bsz1_xh_args)
        pass


    def get_tensors(self, key: str):
        return {
            f"{key}.{subkey}": tensor.contiguous()
            for subkey, tensor in [
                ("su", self.su),
                ("sv", self.sv),
                ("suh", self.suh),
                ("svh", self.svh),
                ("trellis", self.trellis),
                ("bias", self.bias),
                ("mcg", self.mcg_tensor),
                ("mul1", self.mul1_tensor),
            ] if tensor is not None
        }


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if "ovr" in params:
            ovr = params["ovr"]
            if self.key in ovr and ovr[self.key].inner is not self:
                return ovr[self.key].forward(x, params, out_dtype)

        # The EXL3 kernels read x as contiguous rows; a strided view (e.g. a head-group slice
        # of a wider tensor) would be silently misread as interleaved garbage. Producers are
        # responsible for contiguity (a silent copy here would hide a hot-path inefficiency
        # and break CUDA-graph address stability)
        assert x.is_contiguous(), f"LinearEXL3 {self.key}: non-contiguous input {tuple(x.shape)}"

        # Never build the dense weight during the autosplit loader's measuring forwards: the
        # loader reserves free VRAM for the weights it has yet to load, and a dense dequant
        # here would consume it and fail the load mid-way
        measure = params.get("autosplit_measure")

        if not params.get("reconstruct"):
            rows = x.numel() // x.shape[-1]
            no_recon = self._no_recon
            if no_recon is None:
                no_recon = self._no_recon = self.config.infer_params.no_reconstruct
            if rows <= AUTO_RECONSTRUCT_THRESHOLD or no_recon:
                dtype = out_dtype or self.default_out_dtype
                if not measure and self.ensure_dense(x.device):
                    return self.dense_forward(x, out_dtype)
                return self.bc.run_alloc(x, self.out_features, dtype == torch.float)

        if not measure and self.ensure_dense(x.device):
            return self.dense_forward(x, out_dtype)

        return self.reconstruct_hgemm(x, out_dtype)

    def ensure_dense(self, device: torch.device | None = None) -> bool:
        """Pre-dequantize the trellis into the persistent dense fp16 weight (XPU + EXL3_SYCL_DENSE
        only). Returns False where the dense path does not apply (other backends, unsuitable dims,
        or the global VRAM budget being exhausted, which falls back to the packed paths)."""
        global _dense_budget
        if self._dense_w is not None:
            return True
        # The backend/dims conditions never change per instance: memoized (this runs 373x/token
        # through the decode forwards), re-computed if the weight device changes (a CPU swap
        # would otherwise memoize False permanently). The VRAM budget stays dynamic.
        dev = self.trellis.device
        if self._dense_static_dev != dev:
            self._dense_static_dev = dev
            self._dense_static_ok = (
                _sycl_dense_ok(dev) and
                self.in_features % 128 == 0 and self.out_features % 128 == 0
            )
        if not self._dense_static_ok:
            return False
        nbytes = self.in_features * self.out_features * 2
        if _dense_budget is None:
            free, _ = torch.xpu.mem_get_info()
            _dense_budget = free - _dense_safety(self.config)
        if nbytes > _dense_budget:
            return False
        try:
            w = torch.empty(
                (self.in_features, self.out_features),
                dtype = torch.half,
                device = self.trellis.device,
            )
            ext.reconstruct(w, self.trellis, self.K, self.mcg, self.mul1)
            self._dense_w = w
        except torch.OutOfMemoryError:
            _dense_budget = 0
            return False
        _dense_budget -= nbytes
        return True

    def dense_forward(self, x: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """had -> oneMKL gemm -> had over the pre-dequantized (rotated-basis) dense weight. The
        same chain the packed GEMV and the non-fused prefill path implement."""
        shape = x.shape
        rows = x.numel() // shape[-1]
        out_shape = shape[:-1] + (self.out_features,)
        x2 = x.view(rows, self.in_features)
        y = torch.empty(out_shape, dtype = out_dtype or self.default_out_dtype, device = x.device)
        y_ = y.view(rows, self.out_features)

        xh = torch.empty_like(x2)
        ext.had_r_128(x2, xh, self.suh, None, 1.0)
        ext.hgemm_recon(xh, self._dense_w, y_)
        ext.had_r_128(y_, y_, None, self.svh, 1.0)

        if self.bias is not None:
            y += self.bias
        return y


    def unpack_bf(self, bitfield: torch.Tensor):
        # For some reason this operation causes a GPU assert on Transformers. Running on CPU seems to fix it
        device = bitfield.device
        if self.transformers_fix:
            bitfield = bitfield.cpu()

        # (Only used for full reconstruct and loading old models, not during inference)
        bitfield = bitfield.view(torch.uint16).to(torch.int)
        masks = (1 << torch.arange(16)).to(bitfield.device)
        expanded = (bitfield.unsqueeze(-1) & masks) > 0
        expanded = expanded.flatten()
        # NOT torch.where with CPU scalar tensors: that path misses the device guard when the
        # condition lives on a non-current device (observed on torch 2.11.0+cu130) — the kernel
        # launches on the current device's context, faults there, silently zero-fills the output
        # and leaves every other device in the process unusable. Map bool -> {-1, +1} arithmetically
        expanded = 1.0 - expanded.to(torch.float16) * 2.0
        return expanded.contiguous().to(device)


    def reconstruct_hgemm(self, x: torch.Tensor, out_dtype):

        shape = x.shape
        rows = x.numel() // shape[-1]
        out_shape = shape[:-1] + (self.out_features,)
        x = x.view(rows, self.in_features)
        y = torch.empty(out_shape, dtype = out_dtype or self.default_out_dtype, device = x.device)

        y_ = y.view(rows, self.out_features)

        # Fused path: reconstruct emits ORIGINAL-basis weights (both Hadamards + sign
        # vectors folded into the memory-bound reconstruct kernel), so the gemm runs on the
        # raw input and the standalone input/output had_r_128 launches disappear (~14% of
        # long-chunk prefill GPU time). Requires 128-divisible dims (always true for EXL3
        # tensors: both sides are had-transformed at quant time)
        if self._fused_reconstruct is None:
            self._fused_reconstruct = (
                self.in_features % 128 == 0 and self.out_features % 128 == 0
                and not no_fused_reconstruct
            )

        # The fused kernel costs ~4x plain reconstruct (k*n-proportional) while the saved
        # had launches scale with rows*(k+n); breakeven is rows ~400-900 across shapes
        use_fused = self._fused_reconstruct and rows >= 1024

        if use_fused:
            xh = x
        else:
            xh = torch.empty_like(x)
            ext.had_r_128(x, xh, self.suh, None, 1.0)

        if self.out_features <= MAX_RECONSTRUCT_SLICE_N:
            w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
            if use_fused:
                ext.reconstruct_had_slice(w, self.trellis, self.suh, self.svh, self.K, self.mcg, self.mul1, 0)
            else:
                self._reconstruct_inner(w, 0)
            ext.hgemm_recon(xh, w, y_)
        else:
            numel_ = self.in_features * MAX_RECONSTRUCT_SLICE_N
            w_ = torch.empty((numel_,), dtype = torch.half, device = self.trellis.device)
            for n_start in range(0, self.out_features, MAX_RECONSTRUCT_SLICE_N):
                n_end = min(n_start + MAX_RECONSTRUCT_SLICE_N, self.out_features)
                numel = self.in_features * (n_end - n_start)
                w = w_[:numel].view(self.in_features, n_end - n_start)
                if use_fused:
                    ext.reconstruct_had_slice(
                        w, self.trellis, self.suh, self.svh[n_start:], self.K, self.mcg, self.mul1, n_start)
                else:
                    self._reconstruct_inner(w, n_start)
                ext.hgemm_recon(xh, w, y_[:, n_start:n_end])

        if not use_fused:
            ext.had_r_128(y_, y_, None, self.svh, 1.0)

        if self.bias is not None:
            y += self.bias
        return y


    def _reconstruct_inner(self, w: torch.Tensor, n_start: int):
        """Rotated-basis weight slice (in_features, w.shape[1]) from packed columns n_start.."""
        if n_start == 0 and w.shape[1] == self.out_features:
            ext.reconstruct(w, self.trellis, self.K, self.mcg, self.mul1)
        else:
            ext.reconstruct_slice(w, self.trellis, self.K, self.mcg, self.mul1, n_start)

    def get_inner_weight_tensor(self, n_offset: int = 0, n_features: int | None = None):
        w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
        self._reconstruct_inner(w, 0)
        return w


    def get_weight_tensor(self):
        # suh = self.unpack_bf(self.su).unsqueeze(1)
        suh = self.unpack_bf(self.su).unsqueeze(1) if self.su else self.suh.unsqueeze(1)
        svh = self.unpack_bf(self.sv).unsqueeze(0) if self.sv else self.svh.unsqueeze(0)
        w = self.get_inner_weight_tensor()
        w = preapply_had_l(w, had_k)
        w *= suh
        w = preapply_had_r(w, had_n)
        w *= svh
        # w *= self.scale
        return w


    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias


    # Swap tensors to CPU (to free some space while quantizing)
    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.trellis.device
        if self.su is not None: self.su = self.su.cpu()
        if self.sv is not None: self.sv = self.sv.cpu()
        if self.suh is not None: self.suh = self.suh.cpu()
        if self.svh is not None: self.svh = self.svh.cpu()
        if self.trellis is not None: self.trellis = self.trellis.cpu()
        if self.bias is not None: self.bias = self.bias.cpu()


    def unswap_cpu(self):
        if self.swap_device is None:
            return
        if self.su is not None: self.su = self.su.to(self.swap_device)
        if self.sv is not None: self.sv = self.sv.to(self.swap_device)
        if self.suh is not None: self.suh = self.suh.to(self.swap_device)
        if self.svh is not None: self.svh = self.svh.to(self.swap_device)
        if self.trellis is not None: self.trellis = self.trellis.to(self.swap_device)
        if self.bias is not None: self.bias = self.bias.to(self.swap_device)
        self.swap_device = None


    def tp_export(self, plan, producer):
        return {
            "cls": LinearEXL3,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "suh": producer.send(self.suh),
            "svh": producer.send(self.svh),
            "trellis": producer.send(self.trellis),
            "bias": producer.send(self.bias),
            "mcg": producer.send(self.mcg_tensor),
            "mul1": producer.send(self.mul1_tensor),
            "out_dtype": self.out_dtype,
        }


    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)

        if split is not None:
            split_out, first, last = split
        else:
            split_out, first, last = True, 0, exported["out_features"]

        if split_out:
            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features = last - first
        else:
            suh = consumer.recv(id_suh, cuda = True, slice_dim = 0, first = first, last = last)
            svh = consumer.recv(id_svh, cuda = True)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 0, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True) if first == 0 else None
            in_features = last - first
            out_features = exported["out_features"]

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            bias = bias,
            out_dtype = exported["out_dtype"],
        )
        return module


    @staticmethod
    def tp_import_split_3(local_context, exported, plan, split_0, split_1, split_2, dbg = False):
        return LinearEXL3.tp_import_split_n(local_context, exported, plan, [split_0, split_1, split_2], dbg)


    @staticmethod
    def tp_import_split_n(local_context, exported, plan, splits, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)

        svh_ = []
        trellis_ = []
        bias_ = []
        in_features = 0
        out_features = 0

        for split in splits:
            assert split is not None
            split_out, first, last = split
            assert split_out

            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features += last - first
            svh_.append(svh)
            trellis_.append(trellis)
            bias_.append(bias)

        svh = torch.cat(svh_, dim = 0)
        trellis = torch.cat(trellis_, dim = 1)
        bias = torch.cat(bias_, dim = 0) if bias_[0] is not None else None

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            bias = bias,
            out_dtype = exported["out_dtype"],
        )
        return module
