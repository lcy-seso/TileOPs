"""What the single-step GLA and DeltaNet decode kernels share: construction from the call
record, the launch over ``(q, k, v, gate, state)`` and the tuning sweep."""

import warnings

import torch

from tileops._kernels.autotune import TUNING_FATAL, candidate_busy_ms
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention.call_spec import (
    DeltaNetDecodeCall,
    GLADecodeCall,
    head_count_refusal,
)

__all__ = ["StepDecodeKernel", "StreamedStateDecodeKernel", "load_state_without_tma"]


def load_state_without_tma(dim_k: int, k_tile: int) -> bool:
    """Whether a decode program copies its state without TMA: where one key tile holds the
    whole state, the one-shot bulk copy costs more than the plain copy."""
    return k_tile == dim_k


class StepDecodeKernel(Kernel):
    """One recurrence step over a caller-owned state, built from its call record.

    A subclass supplies ``_compile(config)``, ``autotune_configs`` and a ``forward`` under
    its interface's parameter names that passes the five tensors to ``_launch``.
    """

    # GLA gates every key channel ``(batch, heads, dim_k)``; DeltaNet's beta is one value a
    # head ``(batch, heads)``.
    per_key_gate: bool = False

    @classmethod
    def refusal(cls, call: GLADecodeCall | DeltaNetDecodeCall) -> str | None:
        return head_count_refusal(call.heads) or super().refusal(call)

    def __init__(self, call: GLADecodeCall | DeltaNetDecodeCall, config: dict | None = None):
        super().__init__(device_index=call.device_index)
        self.call = call
        self.dtype = call.dtype
        self.init_config(config)
        self.program(self._compile)

    def autotune(self, warmup: int = 10, rep: int = 20) -> None:
        """Time ``_compile(config)`` for each candidate and publish the fastest.

        A candidate that fails to build or launch is skipped with a warning; when every one
        fails, the published config stays.
        """
        call = self.call
        rows = (call.batch, call.heads)
        options = {"device": self._build_device, "dtype": self.dtype}
        # A GLA log-space gate below zero decays the state; a DeltaNet beta lies in (0, 1).
        if self.per_key_gate:
            gate = -torch.rand(*rows, call.dim_k, **options)
        else:
            gate = torch.rand(*rows, **options)
        inputs = (
            torch.randn(*rows, call.dim_k, **options),
            torch.randn(*rows, call.dim_k, **options),
            torch.randn(*rows, call.dim_v, **options),
            gate,
            torch.randn(*rows, call.dim_k, call.dim_v, **options),
        )
        best_time = float("inf")
        best_config: dict | None = None
        failures: list[str] = []
        for config in self.tuning_candidates(self.autotune_configs):
            try:
                fn = self._compile(config)
                t = candidate_busy_ms(lambda _fn=fn: _fn(*inputs), warmup, rep)
            except TUNING_FATAL:
                raise
            except Exception as exc:
                failures.append(f"{config}: {type(exc).__name__}: {exc}")
                continue
            if t < best_time:
                best_time = t
                best_config = config
        name = type(self).__name__
        if failures:
            warnings.warn(
                f"{name} skipped {len(failures)} autotune candidate(s): {failures}",
                RuntimeWarning,
                stacklevel=2,
            )
        if best_config is None:
            warnings.warn(
                f"{name} found no successful autotune candidate; keeping the published config.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self._publish(best_config, self._compile)

    def _launch(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the published program on contiguous ``(q, k, v, gate, state)``."""
        return self.program(self._compile)(*(tensor.contiguous() for tensor in tensors))


class StreamedStateDecodeKernel(StepDecodeKernel):
    """A decode kernel whose program streams the state through shared memory in
    ``k_tile``-row tiles, swept over the key tiles that divide ``dim_k``."""

    # The key tile the untuned program splits the state into, so it decides which key
    # dims the kernel serves.
    _DEFAULT_K_TILE = 16

    @classmethod
    def refusal(cls, call: GLADecodeCall | DeltaNetDecodeCall) -> str | None:
        if call.dim_k % cls._DEFAULT_K_TILE:
            return f"requires dim_k a multiple of {cls._DEFAULT_K_TILE}, got {call.dim_k}"
        return super().refusal(call)

    @property
    def default_config(self) -> dict:
        return {"num_stages": 2, "threads": 128, "k_tile": self._DEFAULT_K_TILE}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"num_stages": num_stages, "threads": threads, "k_tile": k_tile}
            for k_tile in (16, 32, 64)
            if self.call.dim_k % k_tile == 0
            for num_stages in (1, 2, 3)
            for threads in (128, 256)
        ]
