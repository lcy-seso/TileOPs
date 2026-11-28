from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.attention.fp8_lightning_indexer import (
    FP8LightningIndexerCall,
    FP8LightningIndexerFwdInterface,
    FP8LightningIndexerKernel,
    FP8LightningIndexerWSKernel,
)
from tileops._kernels.constants import FP8_E4M3_MAX
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["FP8LightningIndexerFwdOp"]


class FP8LightningIndexerFwdOp(Op):
    """Lightning indexer logits over FP8 index keys.

    For query ``s`` and key ``t`` of group ``g``, the logit sums ``weights[s, h]`` times
    ``relu(q[s, h] . k[t, g])`` over the heads of group ``g``, for keys inside the query's
    window ``[starts[s], ends[s])``. A bf16 call is quantized to FP8 by the op. An FP8 call
    passes per-key scales in ``index_k_scale``; ``k[t, g]`` is then the FP8 key times its
    scale, which may be negative.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "fp8_lightning_indexer_kernel": FP8LightningIndexerKernel,
        "fp8_lightning_indexer_ws_kernel": FP8LightningIndexerWSKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "fp8_lightning_indexer": FP8LightningIndexerFwdInterface
    }

    def roofline_data_terms(self) -> "dict[str, int]":
        """The keys each batch row scores and the distinct keys its windows cover."""
        from tileops._perf.formulas import (
            lightning_indexer_read_keys,
            lightning_indexer_scored_keys,
        )

        call = self.last_call
        return {
            "scored_keys": lightning_indexer_scored_keys(call),
            "read_keys": lightning_indexer_read_keys(call),
        }

    def __init__(
        self,
        clean_logits: bool = True,
        *,
        assume_nonnegative_scale: bool = False,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            clean_logits: Set the logits of keys outside a query's window to ``-inf``;
                otherwise those slots hold unspecified values.
            assume_nonnegative_scale: Skip sign checks on externally supplied key scales.
                Set this only when every supplied scale is nonnegative. Defaults to ``False``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.clean_logits = clean_logits
        self.assume_nonnegative_scale = assume_nonnegative_scale

        super().__init__(target=target)

    def forward(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        index_k_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score every key in each query's window.

        Args:
            index_q: Index queries, ``(B, S, H, D)``, bfloat16 or float8_e4m3fn.
            index_k: Index keys, ``(B, S_kv, G, D)``, the dtype of ``index_q``; ``H`` is a
                multiple of ``G``.
            weights: Per-head weights, ``(S, H)``, float32.
            starts: First key of each query's window, ``(S,)``, int32 in ``[0, S_kv]``.
            ends: One past the last key of each query's window, ``(S,)``, int32 in
                ``[0, S_kv]``.
            index_k_scale: Per-key scales of FP8 keys, ``(B, S_kv, G)``, float32; only with
                float8_e4m3fn inputs, which then require it. ``None`` for a bf16 call.

        Returns:
            ``logits``, ``(B, S, S_kv, G)``, float32.
        """
        check_scale_sign = index_k_scale is not None and not self.assume_nonnegative_scale
        if index_k_scale is None:
            # A bf16 call is quantized here; the kernel indexes FP8 keys and their scales.
            index_q = index_q.to(torch.float8_e4m3fn)
            key_f32 = index_k.float()
            scale = key_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / FP8_E4M3_MAX
            index_k = (key_f32 * scale.reciprocal()).to(torch.float8_e4m3fn)
            index_k_scale = scale.squeeze(-1)
        batch, seq_len, heads, index_dim = index_q.shape
        _, seq_len_kv, kv_group, _ = index_k.shape
        inputs = tuple(
            t.contiguous() for t in (index_q, index_k, index_k_scale, weights, starts, ends)
        )
        kernel = self.kernel_for(
            "fp8_lightning_indexer",
            lambda: FP8LightningIndexerCall(
                batch=batch,
                seq_len=seq_len,
                heads=heads,
                index_dim=index_dim,
                seq_len_kv=seq_len_kv,
                kv_group=kv_group,
                clean_logits=self.clean_logits,
                check_scale_sign=check_scale_sign,
                device=index_q.device,
            ),
        )
        return kernel(*inputs)

    def roof_key(self) -> str:
        """Index scores contract at fp8 regardless of the input form."""
        return tensor_core_roof(torch.float8_e4m3fn)
