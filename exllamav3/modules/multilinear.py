from __future__ import annotations
import torch
from . import Linear
from .quant.exl3 import _sycl_dense_ok
from ..util.tensor import ptr_tensor

class MultiLinear:
    def __init__(
        self,
        device: torch.Device,
        linears: list[Linear],
        allow_bias: bool = False,
    ):
        self.device = device
        self.linears = linears
        self.num_linears = len(linears)

        assert all(l.quant_type == "exl3" for l in linears)
        # The mgemm kernels never apply biases; callers setting allow_bias handle them
        # separately (BC_BlockSparseMLP bias kernels)
        assert allow_bias or all(l.inner.bias is None for l in linears)
        assert all(not l.softcap for l in linears)
        assert all(l.post_scale == 1.0 for l in linears)

        self.in_features = linears[0].in_features
        self.out_features = linears[0].out_features
        self.K = linears[0].inner.K
        assert all(l.inner.K == self.K for l in linears)
        assert all(l.in_features == self.in_features for l in linears)
        assert all(l.out_features == self.out_features for l in linears)

        self.ptrs_suh = ptr_tensor([l.inner.suh.data_ptr() for l in linears], device)
        self.ptrs_svh = ptr_tensor([l.inner.svh.data_ptr() for l in linears], device)

        # Dense mode (XPU + EXL3_SYCL_DENSE) is decided after the model has loaded (try_dense,
        # called once post-load): densifying here would consume the VRAM the loader reserved for
        # the not-yet-loaded weights, and the free-VRAM measurement is only honest once every
        # weight is in place
        self.dense = False
        self._dense_tried = False
        self.ptrs_trellis = ptr_tensor([l.inner.trellis.data_ptr() for l in linears], device)

        self.mcg = linears[0].inner.mcg
        assert all(l.inner.mcg == self.mcg for l in linears[1:])
        self.mul1 = linears[0].inner.mul1
        assert all(l.inner.mul1 == self.mul1 for l in linears[1:])

    def try_dense(self):
        """Build the dense weights and switch the pointer tables to them, once, after the model
        has loaded (densifying during the load would starve the loader's remaining weights).
        Returns True when the dense path is active."""
        if self._dense_tried:
            return self.dense
        self._dense_tried = True
        if not _sycl_dense_ok(self.device):
            return False
        if all(l.inner.ensure_dense(self.device) for l in self.linears):
            self.dense = True
            self.ptrs_trellis = ptr_tensor([l.inner._dense_w.data_ptr() for l in self.linears], self.device)
        return self.dense

    def q_cb(self):
        return self.mcg, self.mul1

    def unload(self):
        # Free the dense weights the pointer tables reference, so a model swap releases the
        # VRAM deterministically instead of waiting for the modules to be collected
        # (inner is None for linears already unloaded by Linear.unload)
        for l in self.linears:
            if l.inner is not None:
                l.inner._dense_w = None


class SlicedMultiLinear:
    """
    Bundle of linears that share an input but differ in width (Q/K/V, optionally a full gate),
    cut into equal-width column slices and run as ONE exl3_mgemm in sliced mode. The kernel
    schedules every slice as its own concurrent z-group, so a wide Q can't leave the K/V groups
    idle the way per-matrix scheduling does, and the input transform runs once per source.

    Slice pointers index the trellis / svh at the slice's first column; the kernel gets the
    source's full width as the row stride and writes each slice in place into the source's
    output. c_ptrs() builds the per-slice output pointer table for a set of output tensors.
    """

    def __init__(
        self,
        device: torch.Device,
        linears: list[Linear],
        min_width: int = 256,
    ):
        import math
        self.device = device
        self.linears = linears
        self.num_src = len(linears)

        assert all(l.quant_type == "exl3" for l in linears)
        assert all(l.inner.bias is None for l in linears), "SlicedMultiLinear: biases unsupported"
        assert all(not l.softcap for l in linears)
        assert all(l.post_scale == 1.0 for l in linears)

        self.in_features = linears[0].in_features
        self.K = linears[0].inner.K
        assert all(l.inner.K == self.K for l in linears)
        assert all(l.in_features == self.in_features for l in linears)
        self.mcg = linears[0].inner.mcg
        assert all(l.inner.mcg == self.mcg for l in linears[1:])
        self.mul1 = linears[0].inner.mul1
        assert all(l.inner.mul1 == self.mul1 for l in linears[1:])

        # Equal slice width: the gcd of the widths, which has to be a whole number of 128-column
        # blocks and not so narrow that a slice can't fill its group
        widths = [l.out_features for l in linears]
        self.width = math.gcd(*widths)
        if self.width % 128 != 0 or self.width < min_width:
            raise ValueError(f"SlicedMultiLinear: unsuitable slice width {self.width} for widths {widths}")

        # Per-slice layout (target, column offset, row stride, source) — independent of dense mode
        targets, offsets, strides, srcs = [], [], [], []
        for i, l in enumerate(linears):
            for n0 in range(0, l.out_features, self.width):
                targets.append(i)
                offsets.append(n0)
                strides.append(l.out_features)
                srcs.append(i)
        self.num_slices = len(targets)
        self.slice_targets = targets
        self.slice_offsets = offsets

        self.ptrs_suh = ptr_tensor([l.inner.suh.data_ptr() for l in linears], device)
        self.size_n_list = torch.full((self.num_slices,), self.width, dtype = torch.int32, device = device)
        self.n_stride_list = torch.tensor(strides, dtype = torch.int32, device = device)
        self.had_src_list = torch.tensor(srcs, dtype = torch.int32, device = device)
        # CPU descriptor for the C++ graph builders: target, column offset, width, row stride, source
        self.meta = torch.tensor([targets, offsets, [self.width] * self.num_slices, strides, srcs], dtype = torch.int32)

        # Dense mode is decided after the model has loaded (try_dense): densifying here would
        # consume the VRAM the loader reserved for the not-yet-loaded weights
        self.dense = False
        self._dense_tried = False
        self._build_tables()

    def _build_tables(self):
        """Weight and output-scale pointer tables for the current dense mode."""
        trellis_ptrs, svh_ptrs = [], []
        for i, l in enumerate(self.linears):
            if self.dense:
                trellis = l.inner._dense_w      # (in, out) fp16, pre-dequantized
                assert trellis.dim() == 2 and trellis.shape[-1] == l.out_features and trellis.dtype == torch.half
                slice_stride = trellis.element_size()
            else:
                trellis = l.inner.trellis     # (K_in / 16, N / 16, 16 * bits) int16
                assert trellis.dim() == 3 and trellis.shape[-1] == 16 * self.K and trellis.dtype == torch.int16
                slice_stride = trellis.shape[-1] * trellis.element_size()
            assert trellis.is_contiguous() and l.inner.svh.is_contiguous()
            for n0 in range(0, l.out_features, self.width):
                if self.dense:
                    # Column slice of the dense [in, out] weight: offset n0 elements, the kernel
                    # reads it with the full-width row stride (= n_stride_list)
                    trellis_ptrs.append(trellis.data_ptr() + n0 * slice_stride)
                else:
                    trellis_ptrs.append(trellis.data_ptr() + (n0 // 16) * slice_stride)
                svh_ptrs.append(l.inner.svh.data_ptr() + n0 * l.inner.svh.element_size())
        self.ptrs_trellis = ptr_tensor(trellis_ptrs, self.device)
        self.ptrs_svh = ptr_tensor(svh_ptrs, self.device)

    def try_dense(self):
        """Build the dense weights and switch the pointer tables to them, once, after the model
        has loaded. Returns True when the dense path is active."""
        if self._dense_tried:
            return self.dense
        self._dense_tried = True
        if not _sycl_dense_ok(self.device):
            return False
        if all(l.inner.ensure_dense(self.device) for l in self.linears):
            self.dense = True
            self._build_tables()
        return self.dense

    def c_ptrs(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        """Per-slice output pointers: outputs[i] is linear i's contiguous (rows, out_features) output"""
        assert len(outputs) == self.num_src
        ptrs = []
        for t, n0 in zip(self.slice_targets, self.slice_offsets):
            o = outputs[t]
            assert o.is_contiguous() and o.shape[-1] == self.linears[t].out_features
            ptrs.append(o.data_ptr() + n0 * o.element_size())
        return ptr_tensor(ptrs, self.device)

    def q_cb(self):
        return self.mcg, self.mul1

    def unload(self):
        # Free the dense weights the pointer tables reference, so a model swap releases the
        # VRAM deterministically instead of waiting for the modules to be collected
        # (inner is None for linears already unloaded by Linear.unload)
        for l in self.linears:
            if l.inner is not None:
                l.inner._dense_w = None
