"""
GLA (Gated Linear Attention) decode (single-step recurrence).

    S_new = diag(exp(gk)) @ S + outer(k, v)
    o     = scale * q^T @ S_new

where gk is per-key-dimension log-space gate [B, H, DK].

The fp16/bf16 kernel runs one block a ``(batch, head, value slice)`` and reads each state
element once; the fp32 kernel runs one block a head.
"""

import math

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import LOG2E
from tileops._kernels.linear_attention.call_spec import GLADecodeCall, GLADecodeFwdInterface
from tileops._kernels.linear_attention.step_decode import (
    StepDecodeKernel,
    StreamedStateDecodeKernel,
    load_state_without_tma,
)

__all__ = ["GLADecodeFP32Kernel", "GLADecodeKernel"]


@target_cache(maxsize=32)
def _gla_decode_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    block_v: int,
    threads: int,
    dtype: str,
    scale: float,
):
    """Build the decode program for one value-slice width.

    Args:
        batch: Sequences in the call.
        head: Heads a sequence.
        dim_k: The key dimension, reduced inside a block.
        dim_v: The value dimension, split into ``block_v``-wide slices, one a block.
        block_v: The value columns one block owns.
        threads: Upper bound on the threads a block.
        dtype: TileLang name of the input and output dtype.
        scale: Query scale.

    Returns:
        The built TileLang program.
    """
    accum_dtype = "float32"
    if dim_v % block_v != 0:
        raise ValueError(f"dim_v={dim_v} must be divisible by block_v={block_v}")
    value_blocks = dim_v // block_v
    # The key-axis exchange pairs threads by XOR, so both group counts must be powers of two;
    # otherwise each thread holds the whole slice width.
    vector = math.gcd(block_v, 16 // getattr(torch, dtype).itemsize)
    if (block_v // vector) & (block_v // vector - 1):
        vector = block_v
    column_groups = block_v // vector
    row_groups = math.gcd(dim_k & -dim_k, max(threads // column_groups, 1))
    block_threads = row_groups * column_groups

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False},
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
    )
    def decode():
        state_layout = T.Fragment(
            [dim_k, block_v],
            forward_thread_fn=lambda i, j: (i % row_groups) * column_groups + j // vector,
            forward_index_fn=lambda i, j: (i // row_groups) * vector + j % vector,
        )

        @T.prim_func
        def gla_decode(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            gk: T.Tensor([batch, head, dim_k], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(value_blocks, head, batch, threads=block_threads) as (vb, hid, bid):
                v0 = vb * block_v
                h_in = T.alloc_fragment([dim_k, block_v], dtype)
                h_out = T.alloc_fragment([dim_k, block_v], dtype)
                h_new = T.alloc_fragment([dim_k, block_v], accum_dtype)
                o_frag = T.alloc_fragment([block_v], accum_dtype)
                T.annotate_layout({h_in: state_layout, h_out: state_layout, h_new: state_layout})
                q_shared = T.alloc_shared([dim_k], accum_dtype)
                k_shared = T.alloc_shared([dim_k], accum_dtype)
                decay_shared = T.alloc_shared([dim_k], accum_dtype)
                v_shared = T.alloc_shared([block_v], accum_dtype)

                T.copy(state[bid, hid, 0, v0], h_in)
                for i in T.Parallel(dim_k):
                    q_shared[i] = T.cast(q[bid, hid, i], accum_dtype)
                    k_shared[i] = T.cast(k[bid, hid, i], accum_dtype)
                    decay_shared[i] = T.exp2(T.cast(gk[bid, hid, i], accum_dtype) * LOG2E)
                for j in T.Parallel(block_v):
                    v_shared[j] = T.cast(v[bid, hid, v0 + j], accum_dtype)

                for i, j in T.Parallel(dim_k, block_v):
                    h_new[i, j] = (
                        decay_shared[i] * T.cast(h_in[i, j], accum_dtype)
                        + k_shared[i] * v_shared[j]
                    )
                for i, j in T.Parallel(dim_k, block_v):
                    h_out[i, j] = T.cast(h_new[i, j], dtype)
                T.copy(h_out, new_state[bid, hid, 0, v0])

                for i, j in T.Parallel(dim_k, block_v):
                    h_new[i, j] = q_shared[i] * h_new[i, j]
                T.reduce_sum(h_new, o_frag, dim=0, batch=vector)
                for j in T.Parallel(block_v):
                    o[bid, hid, v0 + j] = T.cast(scale * o_frag[j], dtype)

        return gla_decode

    return decode()


class GLADecodeKernel(StepDecodeKernel, GLADecodeFwdInterface):
    """GLA single-step decode kernel, one block a ``(batch, head, value slice)``, fp32 accumulation."""

    supported_archs: list[int] = [80, 89, 90]
    general = True
    per_key_gate = True

    def forward(self, q, k, v, gk, state):
        return self._launch(q, k, v, gk, state)

    def _compile(self, cfg: dict):
        """The decode program *cfg* states."""
        call = self.call
        return _gla_decode_tl(
            call.batch,
            call.heads,
            call.dim_k,
            call.dim_v,
            cfg["block_v"],
            cfg["threads"],
            self.dtype_str,
            call.scale,
        )

    def _block_v_options(self) -> list[int]:
        """The value-slice widths this shape builds, narrowest first."""
        dim_v = self.call.dim_v
        return [b for b in (8, 16, 32, 64, 128) if b < dim_v and dim_v % b == 0] + [dim_v]

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_v": block_v, "threads": threads}
            for block_v in self._block_v_options()
            for threads in (128, 256)
        ]

    @property
    def default_config(self) -> dict:
        """The narrowest value slice whose blocks fit one wave, and 128 threads."""
        call = self.call
        sequences = call.batch * call.heads
        options = self._block_v_options()
        block_v = next(
            (b for b in options if sequences * (call.dim_v // b) <= self.build_target.sm_count),
            options[-1],
        )
        return {"block_v": block_v, "threads": 128}


@target_cache(maxsize=32)
def _gla_decode_fp32_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int,
    scale: float,
):
    """FP32 decode kernel: one block a head computes ``S @ q_gated`` by scalar fp32
    accumulation, then streams the state through shared memory in *k_tile*-row tiles."""
    dtype = "float32"
    accum_dtype = "float32"
    if dim_k % k_tile != 0:
        raise ValueError(f"dim_k={dim_k} must be divisible by k_tile={k_tile}")
    without_tma = load_state_without_tma(dim_k, k_tile)

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=[*CUDA_COMPILE_FLAGS],
    )
    def _decode_func(num_stages, threads=128):
        @T.prim_func
        def gla_decode_fp32(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            gk: T.Tensor([batch, head, dim_k], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)
                # The pipelined loop reads no global memory, which TileLang's warp
                # specialization requires here.
                q_shared = T.alloc_shared([dim_k], dtype)
                k_shared = T.alloc_shared([dim_k], dtype)
                v_shared = T.alloc_shared([dim_v], dtype)
                gk_shared = T.alloc_shared([dim_k], dtype)

                for i in T.Parallel(dim_k):
                    q_shared[i] = q[bid, hid, i]
                    k_shared[i] = k[bid, hid, i]
                    gk_shared[i] = gk[bid, hid, i]
                for i in T.Parallel(dim_v):
                    v_shared[i] = v[bid, hid, i]

                T.fill(sq_frag, 0.0)

                # S @ q_gated where q_gated = q * exp(gk)
                for kk in T.Serial(dim_k):
                    gk_val = gk_shared[kk]
                    alpha_i = T.exp2(gk_val * LOG2E)
                    q_gated = q_shared[kk] * alpha_i
                    for j in T.Parallel(dim_v):
                        sq_frag[j] = sq_frag[j] + q_gated * state[bid, hid, kk, j]

                qk_dot[0] = 0.0
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q_shared[kk] * k_shared[kk]

                # o = scale * (S @ q_gated) + scale * (q . k) * v
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = scale * sq_frag[j] + scale * qk_dot[0] * v_shared[j]

                # new_state[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]
                # No intermediate variables inside T.Parallel: TileLang fails to bind them.
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile, disable_tma=without_tma)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = (
                            T.exp2(gk_shared[kt * k_tile + kk] * LOG2E) * h_tile[kk, j]
                            + k_shared[kt * k_tile + kk] * v_shared[j]
                        )

        return gla_decode_fp32

    return _decode_func


class GLADecodeFP32Kernel(StreamedStateDecodeKernel, GLADecodeFwdInterface):
    """GLA decode for fp32 inputs, one block a head; the state streams in key tiles."""

    supported_archs: list[int] = [80, 89, 90]
    per_key_gate = True

    @classmethod
    def refusal(cls, call: GLADecodeCall) -> str | None:
        if call.dtype != torch.float32:
            return f"serves float32, got {call.dtype}"
        return super().refusal(call)

    def forward(self, q, k, v, gk, state):
        return self._launch(q, k, v, gk, state)

    def _compile(self, cfg: dict):
        """The fp32 decode program *cfg* states."""
        call = self.call
        return _gla_decode_fp32_tl(
            call.batch, call.heads, call.dim_k, call.dim_v, cfg["k_tile"], call.scale
        )(cfg["num_stages"], cfg["threads"])
