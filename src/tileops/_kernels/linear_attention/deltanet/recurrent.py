"""
DeltaNet decode (single-step recurrence, ungated).

    old_val  = S @ k                     # matvec
    v_new    = beta * (v - old_val)
    o        = S @ q + (q . k) * v_new
    S_new    = S + outer(k, v_new)
"""

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention.call_spec import (
    DeltaNetDecodeCall,
    DeltaNetDecodeFwdInterface,
    head_count_refusal,
)
from tileops._kernels.linear_attention.step_decode import (
    StepDecodeKernel,
    StreamedStateDecodeKernel,
    load_state_without_tma,
)

__all__ = [
    "DeltaNetDecodeColumnKernel",
    "DeltaNetDecodeFP32Kernel",
    "DeltaNetDecodeKernel",
    "DeltaNetDecodeWarpKernel",
]


@target_cache(maxsize=32)
@tilelang.jit(out_idx=[5, 6])
def _deltanet_decode_column_kernel(heads: int):
    """Four warps share a 16-column stripe of one 128-row BF16 decode state."""
    width, columns = 128, 16

    @T.prim_func
    def main(
        q: T.Tensor((1, heads, width), "bfloat16"),
        k: T.Tensor((1, heads, width), "bfloat16"),
        v: T.Tensor((1, heads, width), "bfloat16"),
        beta: T.Tensor((1, heads), "bfloat16"),
        state: T.Tensor((1, heads, width, width), "bfloat16"),
        out: T.Tensor((1, heads, width), "bfloat16"),
        new_state: T.Tensor((1, heads, width, width), "bfloat16"),
    ):
        with T.Kernel(width // columns, heads, threads=128) as (tile, head):
            stored = T.alloc_fragment((4, 32, columns), "bfloat16")
            keys = T.alloc_fragment((4, 32), "bfloat16")
            queries = T.alloc_fragment((4, 32), "bfloat16")
            products = T.alloc_fragment((4, 32, columns), "float32")
            partial = T.alloc_fragment((4, columns), "float32")
            recall = T.alloc_shared((4, columns), "float32")
            projection = T.alloc_shared((4, columns), "float32")
            # Each lane holds eight rows of two adjacent columns, so the state is
            # read once for both matvecs and the update uses packed BF16 stores.
            layout = T.Fragment(
                (4, 32, columns),
                forward_thread_fn=lambda w, r, c: w * 32 + (r // 8) * 8 + c // 2,
                forward_index_fn=lambda w, r, c: (r % 8) * 2 + c % 2,
            )
            vector_layout = T.Fragment(
                (4, 32),
                replicate=8,
                forward_thread_fn=lambda w, r, replica: w * 32 + (r // 8) * 8 + replica,
                forward_index_fn=lambda w, r: r % 8,
            )
            T.annotate_layout(
                {stored: layout, products: layout, keys: vector_layout, queries: vector_layout}
            )
            for w, r in T.Parallel(4, 32):
                keys[w, r] = k[0, head, w * 32 + r]
                queries[w, r] = q[0, head, w * 32 + r]
            for w, r, c in T.Parallel(4, 32, columns):
                stored[w, r, c] = state[0, head, w * 32 + r, tile * columns + c]
                products[w, r, c] = T.cast(stored[w, r, c], "float32") * T.cast(
                    keys[w, r], "float32"
                )
            T.reduce_sum(products, partial, dim=1)
            T.copy(partial, recall)
            for w, r, c in T.Parallel(4, 32, columns):
                total = (recall[0, c] + recall[1, c]) + (recall[2, c] + recall[3, c])
                delta = T.cast(beta[0, head], "float32") * (
                    T.cast(v[0, head, tile * columns + c], "float32") - total
                )
                updated = T.cast(stored[w, r, c], "float32") + T.cast(keys[w, r], "float32") * delta
                new_state[0, head, w * 32 + r, tile * columns + c] = T.cast(updated, "bfloat16")
                products[w, r, c] = updated * T.cast(queries[w, r], "float32")
            T.reduce_sum(products, partial, dim=1)
            T.copy(partial, projection)
            for c in T.Parallel(columns):
                out[0, head, tile * columns + c] = T.cast(
                    (projection[0, c] + projection[1, c]) + (projection[2, c] + projection[3, c]),
                    "bfloat16",
                )

    return main


class DeltaNetDecodeColumnKernel(Kernel, DeltaNetDecodeFwdInterface):
    """Use a four-warp column reduction for single-request BF16 Hopper decode."""

    supported_archs = [90]
    preferred_over = frozenset({"deltanet_decode_warp"})

    @classmethod
    def refusal(cls, call: DeltaNetDecodeCall) -> str | None:
        if (
            call.batch != 1
            or call.dtype != torch.bfloat16
            or (call.dim_k, call.dim_v) != (128, 128)
        ):
            return "requires one BF16 decode request with 128-dimensional keys and values"
        return head_count_refusal(call.heads) or super().refusal(call)

    def __init__(self, call: DeltaNetDecodeCall):
        super().__init__(device_index=call.device_index)
        self.heads = call.heads

    def _compile(self, cfg):
        return _deltanet_decode_column_kernel(self.heads)

    def forward(self, q, k, v, beta, state):
        return self.program(self._compile)(*(t.contiguous() for t in (q, k, v, beta, state)))


@target_cache(maxsize=32)
def _deltanet_decode_warp_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    v_tile: int,
    raw_group_size: int,
    raw_maxrregcount: int,
    dtype: str,
):
    reason = DeltaNetDecodeWarpKernel.shape_refusal(dtype, dim_k, dim_v)
    if reason is not None:
        raise ValueError(f"DeltaNet warp decode {reason}")
    if dim_v % v_tile != 0:
        raise ValueError(f"dim_v={dim_v} must be divisible by v_tile={v_tile}")
    if raw_group_size != 2:
        raise ValueError("raw_group_size must equal 2 for the fixed two-lane reductions.")
    if raw_group_size * v_tile != 32:
        raise ValueError("raw_group_size * v_tile must equal one warp")
    if dim_k % raw_group_size != 0:
        raise ValueError(f"dim_k={dim_k} must be divisible by raw_group_size={raw_group_size}")

    total_blocks = batch * head * (dim_v // v_tile)
    k_chunk = dim_k // raw_group_size

    raw_compile_flags = [*CUDA_COMPILE_FLAGS, "-DENABLE_BF16", "--use_fast_math"]
    if raw_maxrregcount > 0:
        raw_compile_flags.append(f"--maxrregcount={raw_maxrregcount}")

    @tilelang.jit(
        out_idx=[-2, -1],
        compile_flags=raw_compile_flags,
    )
    def _decode_func(threads=32):
        @T.prim_func
        def deltanet_decode_warp(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(total_blocks, threads=threads) as (block,):
                tx = T.get_thread_binding()
                vid = block % (dim_v // v_tile)
                nh = block // (dim_v // v_tile)
                bid = nh // head
                hid = nh - bid * head
                v_lane = tx // raw_group_size
                k_rank = tx - v_lane * raw_group_size
                v_idx = vid * v_tile + v_lane
                k_begin = k_rank * k_chunk

                k_shared = T.alloc_shared([dim_k], "float32")
                q_shared = T.alloc_shared([dim_k], "float32")
                h = T.alloc_local([k_chunk], "float32")
                old_partial = T.alloc_var("float32", init=0.0)
                out_partial = T.alloc_var("float32", init=0.0)
                v_new = T.alloc_var("float32", init=0.0)

                for i in T.serial(T.ceildiv(dim_k, 32)):
                    kk = tx + i * 32
                    if kk < dim_k:
                        k_shared[kk] = T.cast(k[bid, hid, kk], "float32")
                        q_shared[kk] = T.cast(q[bid, hid, kk], "float32")
                T.sync_threads()

                beta_val = T.cast(beta[bid, hid], "float32")

                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    h_val = T.cast(state[bid, hid, kk, v_idx], "float32")
                    h[ii] = h_val
                    old_partial += h_val * k_shared[kk]

                old_val = old_partial + T.shfl_down(old_partial, 1, width=raw_group_size)
                if k_rank == 0:
                    v_new = beta_val * (T.cast(v[bid, hid, v_idx], "float32") - old_val)
                v_new = T.shfl_sync(v_new, v_lane * raw_group_size, width=raw_group_size)

                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    h_new = h[ii] + k_shared[kk] * v_new
                    new_state[bid, hid, kk, v_idx] = T.cast(h_new, dtype)
                    out_partial += h_new * q_shared[kk]

                out_val = out_partial + T.shfl_down(out_partial, 1, width=raw_group_size)
                if k_rank == 0:
                    o[bid, hid, v_idx] = T.cast(out_val, dtype)

        return deltanet_decode_warp

    return _decode_func


@target_cache(maxsize=32)
def _deltanet_decode_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int,
    dtype: str,
):
    accum_dtype = "float32"
    if dim_k % k_tile != 0:
        raise ValueError(f"dim_k={dim_k} must be divisible by k_tile={k_tile}")
    without_tma = load_state_without_tma(dim_k, k_tile)

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
    )
    def _decode_func(num_stages, threads=128):
        @T.macro
        def _decode_body(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                h_tile_o = T.alloc_shared([k_tile, dim_v], dtype)
                sk_frag = T.alloc_fragment([dim_v], accum_dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                v_new = T.alloc_shared([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)

                beta_val = T.cast(beta[bid, hid], accum_dtype)

                # Full-fp32 matvecs: TileLang does not reliably lower a tensor-core
                # fragment copy here for fp16/bf16.
                T.fill(sk_frag, 0.0)
                T.fill(sq_frag, 0.0)
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile_o, disable_tma=without_tma)
                    for kk in T.Serial(k_tile):
                        k_val = T.cast(k[bid, hid, kt * k_tile + kk], accum_dtype)
                        q_val = T.cast(q[bid, hid, kt * k_tile + kk], accum_dtype)
                        for j in T.Parallel(dim_v):
                            h_val = T.cast(h_tile_o[kk, j], accum_dtype)
                            sk_frag[j] = sk_frag[j] + k_val * h_val
                            sq_frag[j] = sq_frag[j] + q_val * h_val

                # q . k is a scalar reduction, so keep it separate from the
                # matvec loop whose inner work is parallelized over dim_v.
                qk_dot[0] = T.float32(0.0)
                for kk in T.Serial(dim_k):
                    qk_dot[0] += T.cast(q[bid, hid, kk], accum_dtype) * T.cast(
                        k[bid, hid, kk], accum_dtype
                    )

                # v_new = beta * (v - S @ k)
                for j in T.Parallel(dim_v):
                    v_new[j] = beta_val * (T.cast(v[bid, hid, j], accum_dtype) - sk_frag[j])

                # o = S @ q + (q . k) * v_new
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = T.cast(sq_frag[j] + qk_dot[0] * v_new[j], dtype)

                # new_state = S + outer(k, v_new)
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile, disable_tma=without_tma)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = T.cast(
                            T.cast(h_tile[kk, j], accum_dtype)
                            + T.cast(k[bid, hid, kt * k_tile + kk], accum_dtype) * v_new[j],
                            dtype,
                        )

        @T.prim_func
        def deltanet_decode(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            _decode_body(q, k, v, beta, state, o, new_state)

        return deltanet_decode

    return _decode_func


class DeltaNetDecodeKernel(StreamedStateDecodeKernel, DeltaNetDecodeFwdInterface):
    """DeltaNet single-step decode kernel (ungated): pipelined state prefetch, fp32 matvecs."""

    supported_archs: list[int] = [80, 89, 90]
    general = True

    def forward(self, q, k, v, beta, state):
        return self._launch(q, k, v, beta, state)

    def _compile(self, cfg: dict):
        """The decode program *cfg* states."""
        call = self.call
        return _deltanet_decode_tl(
            call.batch, call.heads, call.dim_k, call.dim_v, cfg["k_tile"], self.dtype_str
        )(cfg["num_stages"], cfg["threads"])


class DeltaNetDecodeWarpKernel(StepDecodeKernel, DeltaNetDecodeFwdInterface):
    """SM90 low-precision decode kernel for the DK=DV=128 DeltaNet case.

    One warp handles one `(batch, head, V tile)`, two lanes cooperate on each
    output value, and each lane keeps its half of the K dimension live in fp32
    local storage.
    """

    supported_archs: list[int] = [90]

    @classmethod
    def refusal(cls, call: DeltaNetDecodeCall) -> str | None:
        return cls.shape_refusal(
            Kernel.dtype_to_str(call.dtype), call.dim_k, call.dim_v
        ) or super().refusal(call)

    @staticmethod
    def shape_refusal(dtype: str, dim_k: int, dim_v: int) -> str | None:
        """Why this kernel cannot serve this shape, or ``None``.

        *dtype* is the TileLang name the builder takes, which the region converts the
        call's dtype to; the builder asks the same question of the arguments it is handed.
        """
        if dtype not in ("float16", "bfloat16"):
            return "requires float16 or bfloat16"
        if dim_k != 128 or dim_v != 128:
            return "requires dim_k == dim_v == 128"
        return None

    def forward(self, q, k, v, beta, state):
        return self._launch(q, k, v, beta, state)

    def _compile(self, cfg: dict):
        """The program *cfg* states."""
        call = self.call
        return _deltanet_decode_warp_tl(
            call.batch,
            call.heads,
            call.dim_k,
            call.dim_v,
            cfg["v_tile"],
            cfg["raw_group_size"],
            cfg["raw_maxrregcount"],
            self.dtype_str,
        )(cfg["threads"])

    @property
    def default_config(self) -> dict:
        return {
            "threads": 32,
            "v_tile": 16,
            "raw_group_size": 2,
            "raw_maxrregcount": 146,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        """The register caps swept around the default config."""
        base = self.default_config
        return [
            {**base, "raw_maxrregcount": raw_maxrregcount}
            for raw_maxrregcount in (0, 128, 132, 146, 160)
        ]


@target_cache(maxsize=32)
def _deltanet_decode_fp32_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int,
):
    """FP32 decode kernel using element-wise matvec instead of T.gemm.

    T.gemm on fp32 inputs rounds to TF32, whose error compounds through the recurrent
    state over decode steps, so S@k and S@q accumulate in scalar fp32.
    """
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
        def deltanet_decode_fp32(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                # Fragment accumulators for S@k and S@q (full fp32, no TF32)
                sk_frag = T.alloc_fragment([dim_v], accum_dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                v_new = T.alloc_shared([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)

                beta_val = T.cast(beta[bid, hid], accum_dtype)

                T.fill(sk_frag, 0.0)
                T.fill(sq_frag, 0.0)

                for kk in T.Serial(dim_k):
                    k_val = k[bid, hid, kk]
                    q_val = q[bid, hid, kk]
                    for j in T.Parallel(dim_v):
                        h_val = state[bid, hid, kk, j]
                        sk_frag[j] = sk_frag[j] + k_val * h_val
                        sq_frag[j] = sq_frag[j] + q_val * h_val

                qk_dot[0] = 0.0
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q[bid, hid, kk] * k[bid, hid, kk]

                # v_new = beta * (v - S @ k)
                for j in T.Parallel(dim_v):
                    v_new[j] = beta_val * (v[bid, hid, j] - sk_frag[j])

                # o = S @ q + (q . k) * v_new
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = sq_frag[j] + qk_dot[0] * v_new[j]

                # new_state = S + outer(k, v_new)
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile, disable_tma=without_tma)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = (
                            h_tile[kk, j] + k[bid, hid, kt * k_tile + kk] * v_new[j]
                        )

        return deltanet_decode_fp32

    return _decode_func


class DeltaNetDecodeFP32Kernel(StreamedStateDecodeKernel, DeltaNetDecodeFwdInterface):
    """FP32 DeltaNet decode kernel, for fp32 inputs, with no TF32 tensor-core products."""

    supported_archs: list[int] = [80, 89, 90]

    @classmethod
    def refusal(cls, call: DeltaNetDecodeCall) -> str | None:
        if call.dtype != torch.float32:
            return "requires float32"
        return super().refusal(call)

    def forward(self, q, k, v, beta, state):
        return self._launch(q, k, v, beta, state)

    def _compile(self, cfg: dict):
        """The fp32 decode program *cfg* states."""
        call = self.call
        return _deltanet_decode_fp32_tl(
            call.batch, call.heads, call.dim_k, call.dim_v, cfg["k_tile"]
        )(cfg["num_stages"], cfg["threads"])
