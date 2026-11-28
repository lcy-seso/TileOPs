from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.attention import (
    MLADecodeMMAKernel,
    MLADecodeSeesawKernel,
    MLADecodeWSKernel,
    MLADenseBwdKernel,
    MLADenseBwdWSKernel,
    MLAPagedFwdKernel,
    MLAPagedSeesawKernel,
    MLAVarlenBwdKernel,
    MLAVarlenBwdWSKernel,
    MLAVarlenPrefillFwdKernel,
    MLAVarlenPrefillWSFwdKernel,
)
from tileops._kernels.attention.call_spec import (
    MLABwdCall,
    MLADecodeCall,
    MLADecodeFwdInterface,
    MLADenseBwdInterface,
    MLAPagedCall,
    MLAPagedFwdInterface,
    MLAVarlenBwdInterface,
    MLAVarlenCall,
    MLAVarlenFwdInterface,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = [
    "MLADenseBwdOp",
    "MLADenseDecodeFwdOp",
    "MLAPagedFwdOp",
    "MLAVarlenBwdOp",
    "MLAVarlenFwdOp",
]


class MLADenseDecodeFwdOp(Op):
    """Multi-Head Latent Attention (MLA) decode against a per-request KV cache. Layout: BSHD.

    The op takes no ``sm_scale``; scores are scaled by ``(D + PE) ** -0.5``, with ``D`` and
    ``PE`` the last extents of ``q`` and ``q_pe``.
    """

    contraction_dtype_from = "q"

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mla_decode_kernel": MLADecodeWSKernel,
        "mla_decode_mma_kernel": MLADecodeMMAKernel,
        "mla_decode_seesaw_kernel": MLADecodeSeesawKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "mla_decode_kernel": MLADecodeFwdInterface
    }

    def forward(
        self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """Attend each request's one query token to its whole cache.

        Args:
            q: ``(B, H, D)`` latent queries, float16 or bfloat16.
            q_pe: ``(B, H, PE)`` rope queries in the query dtype.
            k: ``(B, N_kv, H_kv, D)`` latent keys, also the values, in the query dtype.
            k_pe: ``(B, N_kv, H_kv, PE)`` rope keys in the query dtype.

        Returns:
            ``o`` ``(B, H, D)`` in the query dtype.
        """
        batch, heads, dim = q.shape
        _, seqlen_kv, heads_kv, _ = k.shape
        return self.kernel_for(
            "mla_decode_kernel",
            lambda: MLADecodeCall(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                seqlen_kv=seqlen_kv,
                dim=dim,
                pe_dim=q_pe.shape[2],
                dtype=q.dtype,
                device=q.device,
            ),
        )(q, q_pe, k, k_pe)


class MLAPagedFwdOp(Op):
    """Multi-Head Latent Attention (MLA) decode over a paged latent cache, absorbed form.

    The cache holds one row per token, ``kv_c`` followed by ``k_pe``, shared by every query
    head: the key is the whole row and the value its first ``kv_lora_rank`` columns. The
    ``S_q`` query tokens of a request are its last ``S_q`` cache positions (one for decode,
    more for multi-token prediction drafts). ``lse`` is the float32 natural-log log-sum-exp
    of each row's scaled scores, ``(B, H, S_q)`` head-major as flash-attn returns it, for a
    caller merging partial results. With an FP8 cache
    the scores and the output are multiplied by ``kv_scale``; the cached values are used
    exactly. A cache row past a request's ``cache_seqlens`` never reaches a result,
    whatever it holds.
    """

    contraction_dtype_from = "q"

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mla_paged": MLAPagedFwdKernel,
        "mla_paged_seesaw": MLAPagedSeesawKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"mla_paged": MLAPagedFwdInterface}

    def __init__(
        self,
        kv_lora_rank: int,
        is_causal: bool = False,
        sm_scale: float | None = None,
        *,
        target: Target = None,
    ) -> None:
        """Configure the op. Tensor shapes and dtypes come from each call.

        Args:
            kv_lora_rank: Width of the value: the cache row's first columns.
            is_causal: Whether a query token sees only the keys at or before its position.
                Default ``False``, as in FlashInfer's MLA wrapper.
            sm_scale: Score scale, or ``None`` for ``DK ** -0.5``, ``DK`` the whole cache
                row width (latent plus rope part).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.kv_lora_rank = kv_lora_rank
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        super().__init__(target=target)

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        kv_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend each request's query tokens to its cache.

        Args:
            q: ``(B, S_q, H, DK)`` queries, latent part then rope part, float16 or bfloat16.
            kv_cache: ``(NP, PS, DK)`` latent cache in the query dtype or float8_e4m3fn.
            block_table: ``int32`` ``(B, W)``; key ``n`` of request ``b`` is row
                ``n % PS`` of page ``block_table[b, n // PS]``.
            cache_seqlens: ``int32`` ``(B,)``, the keys each request holds.
            kv_scale: ``float32`` ``(1,)`` dequantization scale, required exactly when
                the cache is float8_e4m3fn.

        Returns:
            ``o`` ``(B, S_q, H, kv_lora_rank)`` in the query dtype and ``lse``
            ``(B, H, S_q)`` float32.
        """
        q = q.contiguous()
        block_table = block_table.contiguous()
        cache_seqlens = cache_seqlens.contiguous()
        kv_scale = None if kv_scale is None else kv_scale.contiguous()
        return self.kernel_for(
            "mla_paged",
            lambda: MLAPagedCall(
                s_q=q.shape[1],
                heads=q.shape[2],
                dim=q.shape[3],
                dim_v=self.kv_lora_rank,
                page_size=kv_cache.shape[1],
                is_causal=self.is_causal,
                sm_scale=self.sm_scale,
                dtype=q.dtype,
                cache_dtype=kv_cache.dtype,
                device=q.device,
            ),
        )(q, kv_cache, block_table, cache_seqlens, kv_scale)


class MLAVarlenFwdOp(Op):
    """Multi-Head Latent Attention (MLA) prefill over packed requests, after the latent is
    decompressed. Layout: THD.

    The key of head ``h`` is ``k_nope[:, h]`` followed by ``k_pe``, which holds one row per
    token and is shared by every head. Queries and keys are the same tokens, so one
    ``cu_seqlens`` describes both and the causal mask sits on the diagonal. ``lse`` is
    the float32 natural-log log-sum-exp of each row's scaled scores, ``(H, T_q)``
    head-major as flash-attn returns it and as :class:`MLAVarlenBwdOp` takes it.

    The caller guarantees well-formed ``cu_seqlens``, or passes ``validate_inputs=True``,
    which checks them on the CPU on every call.
    """

    contraction_dtype_from = "q"

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mla_varlen_fwd": MLAVarlenPrefillFwdKernel,
        "mla_varlen_fwd_ws": MLAVarlenPrefillWSFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "mla_varlen_fwd": MLAVarlenFwdInterface
    }

    def __init__(
        self,
        is_causal: bool = False,
        sm_scale: float | None = None,
        *,
        validate_inputs: bool = False,
        target: Target = None,
    ) -> None:
        """Configure the op. Tensor shapes and input dtype come from each call.

        Args:
            is_causal: Whether a query row reads only keys at or before its own position.
                Default ``False``, as in ``flash_attn_varlen_func``.
            sm_scale: Score scale, or ``None`` for ``(DN + PE) ** -0.5``, the query-key
                head dimension including the rope part.
            validate_inputs: Check on the CPU that ``cu_seqlens`` starts at 0, never
                decreases and ends at ``T_q``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def forward(
        self,
        q: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend each packed request's queries to its own keys.

        Args:
            q: ``(T_q, H, DN + PE)`` queries, latent part then rope part, float16 or
                bfloat16.
            k_nope: ``(T_q, H, DN)`` non-rope keys in the query dtype.
            k_pe: ``(T_q, PE)`` rope keys, shared by every head, in the query dtype.
            v: ``(T_q, H, DV)`` values in the query dtype.
            cu_seqlens: ``int32`` ``(B + 1,)`` request offsets into the packed tokens,
                from 0 to ``T_q``, nondecreasing.

        Returns:
            ``o`` ``(T_q, H, DV)`` and ``lse`` ``(H, T_q)`` float32, ``-inf`` for a row
            that sees no key.
        """
        inputs = tuple(tensor.contiguous() for tensor in (q, k_nope, k_pe, v, cu_seqlens))
        return self.kernel_for(
            "mla_varlen_fwd",
            lambda: MLAVarlenCall(
                batch=cu_seqlens.shape[0] - 1,
                heads=q.shape[1],
                dim_nope=k_nope.shape[2],
                dim_pe=k_pe.shape[1],
                dim_v=v.shape[2],
                is_causal=self.is_causal,
                sm_scale=self.sm_scale,
                dtype=q.dtype,
                device=q.device,
            ),
        )(*inputs)


class MLADenseBwdOp(Op):
    """Multi-Head Latent Attention (MLA) prefill backward over fixed-length requests, after the
    latent is decompressed. Layout: BSHD.

    Every head has its own ``D_qk``-wide query and key and its own ``D_v``-wide value: the
    caller has expanded the shared rope part of the key to every head. ``o`` and ``lse`` are
    the forward's, ``lse`` the natural-log log-sum-exp of each query row's scaled scores. Query
    ``i`` sits at key position ``i + S_kv - S_q``; a query that sees no key has ``lse``
    ``-inf`` and a zero gradient.

    When ``S_q == S_kv``, :class:`MLAVarlenFwdOp` over ``B`` requests of length ``S`` feeds it:
    ``o.view(B, S, H, D_v)`` and ``lse.view(H, B, S).transpose(0, 1)``; a call with
    ``S_q != S_kv`` supplies its own ``o`` and ``lse``. The key is the joined
    ``k = cat(k_nope, k_pe broadcast over heads)``; the gradient of ``k_pe`` is
    ``dk[..., DN:]`` summed over heads.
    """

    contraction_dtype_from = "q"

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mla_dense_bwd": MLADenseBwdKernel,
        "mla_dense_bwd_ws": MLADenseBwdWSKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "mla_dense_bwd": MLADenseBwdInterface
    }

    def __init__(
        self,
        is_causal: bool = False,
        sm_scale: float | None = None,
        *,
        target: Target = None,
    ) -> None:
        """Configure the op. Tensor shapes and input dtype come from each call.

        Args:
            is_causal: Whether a query row reads only keys at or before its own position.
                Default ``False``, matching the forward.
            sm_scale: Score scale, or ``None`` for ``D_qk ** -0.5``, the query-key head
                dimension including the rope part.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        super().__init__(target=target)

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Back-propagate ``do`` through one dense attention call.

        Args:
            do: ``(B, S_q, H, D_v)`` gradient of the loss with respect to ``o``, in the
                query dtype.
            q: ``(B, S_q, H, D_qk)`` queries, float16 or bfloat16.
            k: ``(B, S_kv, H, D_qk)`` keys in the query dtype; ``S_kv > 0``.
            v: ``(B, S_kv, H, D_v)`` values in the query dtype.
            o: ``(B, S_q, H, D_v)`` forward output in the query dtype.
            lse: ``float32`` ``(B, H, S_q)`` natural-log log-sum-exp of the forward.

        Returns:
            ``dq`` ``(B, S_q, H, D_qk)``, ``dk`` ``(B, S_kv, H, D_qk)`` and ``dv``
            ``(B, S_kv, H, D_v)``, in the query dtype.
        """
        inputs = tuple(tensor.contiguous() for tensor in (q, k, v, o, do, lse))
        batch, seqlen_q, heads, dim_qk = q.shape
        return self.kernel_for(
            "mla_dense_bwd",
            lambda: MLABwdCall(
                batch=batch,
                heads=heads,
                dim_qk=dim_qk,
                dim_v=v.shape[3],
                seqlen_q=seqlen_q,
                seqlen_kv=k.shape[1],
                is_causal=self.is_causal,
                sm_scale=self.sm_scale,
                dtype=q.dtype,
                device=q.device,
            ),
        )(*inputs)


class MLAVarlenBwdOp(Op):
    """Multi-Head Latent Attention (MLA) prefill backward over packed requests, after the
    latent is decompressed. Layout: THD.

    Every head has its own ``D_qk``-wide query and key and its own ``D_v``-wide value: the
    caller has expanded the shared rope part of the key to every head. Each request attends
    only to its own keys. ``o`` and ``lse`` are the forward's, ``lse`` the natural-log
    log-sum-exp of each query row's scaled scores, head-major. Query ``i`` of a request sits
    at key position ``i + kv_len - q_len``; a query that sees no key has ``lse`` ``-inf`` and
    a zero gradient.

    When every request's query and key lengths match, :class:`MLAVarlenFwdOp` feeds it: its
    ``o`` and ``lse`` pass through unchanged; otherwise the caller supplies ``o`` and ``lse``.
    The key is the joined ``k = cat(k_nope, k_pe broadcast over heads)``; the gradient of
    ``k_nope`` is ``dk[..., :DN]`` and that of ``k_pe`` is ``dk[..., DN:]`` summed over heads.

    The caller guarantees well-formed offsets, or passes ``validate_inputs=True``, which
    checks them on the CPU on every call.
    """

    contraction_dtype_from = "q"

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mla_varlen_bwd": MLAVarlenBwdKernel,
        "mla_varlen_bwd_ws": MLAVarlenBwdWSKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "mla_varlen_bwd": MLAVarlenBwdInterface
    }

    def __init__(
        self,
        is_causal: bool = False,
        sm_scale: float | None = None,
        *,
        validate_inputs: bool = False,
        target: Target = None,
    ) -> None:
        """Configure the op. Tensor shapes and input dtype come from each call.

        Args:
            is_causal: Whether a query row reads only keys at or before its own position.
                Default ``False``, matching the forward.
            sm_scale: Score scale, or ``None`` for ``D_qk ** -0.5``, the query-key head
                dimension including the rope part.
            validate_inputs: Check on the CPU that each offset tensor starts at 0, never
                decreases and ends at its packed length.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Back-propagate ``do`` through one packed attention call.

        Args:
            do: ``(T_q, H, D_v)`` gradient of the loss with respect to ``o``, in the query
                dtype.
            q: ``(T_q, H, D_qk)`` packed queries, float16 or bfloat16.
            k: ``(T_kv, H, D_qk)`` packed keys in the query dtype.
            v: ``(T_kv, H, D_v)`` packed values in the query dtype.
            o: ``(T_q, H, D_v)`` forward output in the query dtype.
            lse: ``float32`` ``(H, T_q)`` natural-log log-sum-exp of the forward.
            cu_seqlens_q: ``int32`` ``(B + 1,)`` request offsets into ``q``, from 0 to
                ``T_q``, nondecreasing.
            cu_seqlens_kv: ``int32`` ``(B + 1,)`` request offsets into ``k`` and ``v``,
                from 0 to ``T_kv``, nondecreasing.

        Returns:
            ``dq`` ``(T_q, H, D_qk)``, ``dk`` ``(T_kv, H, D_qk)`` and ``dv``
            ``(T_kv, H, D_v)``, in the query dtype.
        """
        inputs = tuple(
            tensor.contiguous() for tensor in (q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv)
        )
        return self.kernel_for(
            "mla_varlen_bwd",
            lambda: MLABwdCall(
                batch=cu_seqlens_q.shape[0] - 1,
                heads=q.shape[1],
                dim_qk=q.shape[2],
                dim_v=v.shape[2],
                is_causal=self.is_causal,
                sm_scale=self.sm_scale,
                dtype=q.dtype,
                device=q.device,
            ),
        )(*inputs)
