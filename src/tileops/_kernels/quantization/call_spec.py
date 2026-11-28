"""Call records for quantization kernels."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod

import torch

from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import KernelInterface

__all__ = [
    "DequantizeCall",
    "FP4QuantCall",
    "FP4QuantFwdInterface",
    "FP8QuantCall",
    "FP8QuantFwdInterface",
    "FP8QuantPerBlockFwdInterface",
    "INT4QuantPerGroupFwdInterface",
    "INT8DequantPerBlockFwdInterface",
    "INT8DequantPerChannelFwdInterface",
    "INT8DequantPerTensorFwdInterface",
    "INT8QuantPerBlockFwdInterface",
    "INT8QuantPerChannelFwdInterface",
    "INT8QuantPerTensorFwdInterface",
    "QuantizeCall",
    "SmoothQuantFwdInterface",
]


@dataclasses.dataclass(frozen=True)
class DequantizeCall(CallSpec):
    """One INT8 dequantize call, as the op knows it after reading ``q``'s shape."""

    m: int = 0
    k: int = 0
    out_dtype: torch.dtype | None = None


@dataclasses.dataclass(frozen=True)
class QuantizeCall(CallSpec):
    """The facts of one quantize call: the 2-D input's shape and dtype, and its group size.

    The block shapes and output dtypes are fixed by each op's signature, so the op a
    kernel serves states them and the record does not.
    """

    # The input's leading extent, ``M`` or ``N``.
    rows: int = 0
    # The extent the scales group along, ``K``.
    cols: int = 0
    dtype: torch.dtype = torch.float16
    # ``INT4QuantPerGroupFwdOp``'s ``group_size``; ``None`` for an op without one.
    group_size: int | None = None


class INT8DequantPerTensorFwdInterface(KernelInterface):
    """INT8 dequantize with one scale for the matrix: ``x = (q.float() * scale).to(out_dtype)``."""

    request = DequantizeCall

    @abstractmethod
    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize *q*; nothing is written in place.

        Both tensors are contiguous on ``call.device``.

        Args:
            q: ``int8`` ``(call.m, call.k)``.
            scale: ``float32`` ``(1,)``.

        Returns:
            A new ``(call.m, call.k)`` tensor in ``call.out_dtype``.
        """


class INT8DequantPerChannelFwdInterface(KernelInterface):
    """INT8 dequantize with one scale per row: ``x = (q.float() * scale[:, None]).to(out_dtype)``."""

    request = DequantizeCall

    @abstractmethod
    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize *q*; nothing is written in place.

        Both tensors are contiguous on ``call.device``.

        Args:
            q: ``int8`` ``(call.m, call.k)``.
            scale: ``float32`` ``(call.m,)``.

        Returns:
            A new ``(call.m, call.k)`` tensor in ``call.out_dtype``.
        """


class INT8DequantPerBlockFwdInterface(KernelInterface):
    """INT8 dequantize with one scale per 128 codes of a row: ``x = (q.float() * scale[m, k // 128]).to(out_dtype)``."""

    request = DequantizeCall

    @abstractmethod
    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize *q*; nothing is written in place.

        Both tensors are contiguous on ``call.device``.

        Args:
            q: ``int8`` ``(call.m, call.k)``.
            scale: ``float32`` ``(call.m, ceil(call.k / 128))``.

        Returns:
            A new ``(call.m, call.k)`` tensor in ``call.out_dtype``.
        """


class INT8QuantPerTensorFwdInterface(KernelInterface):
    """Symmetric INT8 quantization of a whole matrix against one amax."""

    request = QuantizeCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize *x*; nothing is written in place.

        Args:
            x: ``[call.rows, call.cols]``, contiguous, in ``call.dtype`` on ``call.device``.

        Returns:
            A new ``q`` shaped like *x* in ``int8`` and a new ``scale`` ``[1]`` in
            ``float32``.
        """


class INT8QuantPerChannelFwdInterface(KernelInterface):
    """Symmetric INT8 quantization of each row against its own amax."""

    request = QuantizeCall

    @abstractmethod
    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize each of the ``call.rows`` rows of ``w``; nothing is written in place.

        Args:
            w: ``[call.rows, call.cols]``, contiguous, in ``call.dtype`` on ``call.device``.

        Returns:
            A new ``q`` shaped like *w* in ``int8`` and a new ``scale`` ``[call.rows]`` in
            ``float32``.
        """


class INT4QuantPerGroupFwdInterface(KernelInterface):
    """DeepSpeed-compatible asymmetric INT4 quantization and packing."""

    request = QuantizeCall

    @abstractmethod
    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize each group of ``call.group_size`` elements of *w*; nothing is written in place.

        Args:
            w: ``[call.rows, call.cols]``, contiguous, in ``float16`` on ``call.device``.

        Returns:
            New ``int8`` packed codes ``[call.rows, call.cols // 2]`` and new ``float32``
            scale and offset pairs ``[call.rows * call.cols // call.group_size, 2]``.
        """


class INT8QuantPerBlockFwdInterface(KernelInterface):
    """Symmetric INT8 quantization of each 128-element block of a row against its own amax."""

    request = QuantizeCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize each block of the ``call.rows`` rows of ``x``; nothing is written in place.

        Args:
            x: ``[call.rows, call.cols]``, contiguous, in ``call.dtype`` on ``call.device``.

        Returns:
            A new ``q`` shaped like *x* in ``int8`` and a new ``scale``
            ``[call.rows, ceil(call.cols / 128)]`` in ``float32``.
        """


class FP8QuantPerBlockFwdInterface(KernelInterface):
    """Block-scaled ``float8_e4m3fn`` quantization of each 128x128 tile against its own amax."""

    request = QuantizeCall

    @abstractmethod
    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize each tile of the ``call.rows x call.cols`` weight *w*; nothing is written in place.

        Args:
            w: ``[call.rows, call.cols]``, contiguous, in ``call.dtype`` on ``call.device``.

        Returns:
            A new ``q`` shaped like *w* in ``float8_e4m3fn`` and a new ``scale``
            ``[ceil(call.rows / 128), ceil(call.cols / 128)]`` in ``float32``.
        """


class SmoothQuantFwdInterface(KernelInterface):
    """SmoothQuant: divide each column by its smoothing factor, then INT8 per row."""

    request = QuantizeCall

    @abstractmethod
    def forward(self, x: torch.Tensor, smooth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize each of the ``call.rows`` rows of ``x / smooth``; nothing is written in place.

        Args:
            x: ``[call.rows, call.cols]``, contiguous, in ``call.dtype`` on ``call.device``.
            smooth: ``[call.cols]``, contiguous, in ``float32`` on ``call.device``.

        Returns:
            A new ``q`` shaped like *x* in ``int8`` and a new ``scale`` ``[call.rows]`` in
            ``float32``.
        """


@dataclasses.dataclass(frozen=True)
class FP4QuantCall(CallSpec):
    """One per-block e2m1 quantization of a ``[*S, n]`` tensor, blocks of ``block_size``
    columns.

    The leading axes ``*S`` are not part of the call: one kernel serves any token count.
    """

    n: int = 0
    block_size: int = 32
    dtype: torch.dtype = torch.bfloat16
    inplace: bool = False


class FP4QuantFwdInterface(KernelInterface):
    """Block-wise quantization to e2m1 under power-of-two scales."""

    request = FP4QuantCall

    @abstractmethod
    def forward(
        self, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Scale each block by the power of two its maximum rounds up to, and round to e2m1.

        A block is ``call.block_size`` adjacent columns of the last axis. Its scale is
        ``2 ** ceil(log2(m * r))``, ``m`` the block's absolute maximum floored at
        ``6 * 2 ** -126`` and ``r`` the float32 nearest ``1 / 6``; each element's code is
        ``x / scale`` clamped to ``[-6, 6]`` and rounded to the nearest e2m1 value, ties to
        the even code, its sign kept for a zero. A block holding an infinity or a NaN has no
        defined result. Nothing but *input_tensor* under ``call.inplace`` is written in
        place, and no output aliases the input.

        Args:
            input_tensor: ``[*S, call.n]`` in ``call.dtype`` on ``call.device``, contiguous;
                with ``call.inplace``, of any layout.

        Returns:
            ``(q, scale)`` on ``call.device``: new contiguous ``uint8`` ``[*S, call.n // 2]``
            codes, two a byte with the even column in the low nibble, and new contiguous
            ``float8_e8m0fnu`` ``[*S, call.n // call.block_size]`` scales. With
            ``call.inplace``, *input_tensor* is overwritten with each element's e2m1 value
            times its scale and ``(None, None)`` returns.
        """


@dataclasses.dataclass(frozen=True)
class FP8QuantCall(CallSpec):
    """One per-group FP8 quantization of a ``[*S, k]`` tensor, groups of ``group_size`` columns.

    The leading axes ``*S`` are not part of the call: one kernel serves any token count.
    """

    k: int = 0
    group_size: int = 0
    eps: float = 1e-10
    dtype: torch.dtype = torch.bfloat16
    column_major_scales: bool = False
    tma_aligned_scales: bool = False
    use_ue8m0: bool | None = None
    scale_dtype: torch.dtype = torch.float32
    inplace: bool = False


class FP8QuantFwdInterface(KernelInterface):
    """Group-wise quantization to ``float8_e4m3fn``."""

    request = FP8QuantCall

    @abstractmethod
    def forward(
        self, input_tensor: torch.Tensor, out_q: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Scale each group by its own maximum, writing the codes into *out_q* when given.

        A group is ``call.group_size`` adjacent columns of the last axis. Its scale is the
        absolute maximum over the group, floored at ``call.eps``, divided by 448; each element
        is multiplied by the reciprocal of that scale and clamped to the ``float8_e4m3fn``
        range. With ``call.use_ue8m0`` the scale is ``2 ** ceil(log2(m * r))``, ``m`` the
        floored maximum and ``r`` the float32 nearest ``1 / 448``, and each code is
        ``x / scale`` clamped and rounded to nearest even, exactly. A group holding an
        infinity or a NaN, or one whose floored maximum is below ``448 / 3.4e38`` so that the
        reciprocal overflows float32, has no defined result: whether the maximum and the
        clamp carry the non-finite value is left to the implementation. Nothing but *out_q*,
        or with ``call.inplace`` *input_tensor*, is written in place, and no output aliases
        the input.

        Args:
            input_tensor: ``[*S, call.k]`` in ``call.dtype`` on ``call.device``, contiguous;
                with ``call.inplace``, of any layout.
            out_q: ``[*S, call.k]`` ``float8_e4m3fn`` on ``call.device``, contiguous, the
                codes are stored into; or None.

        Returns:
            ``(q, scale)`` on ``call.device``: the codes, *out_q* where given and otherwise a
            new contiguous ``float8_e4m3fn`` tensor shaped as the input, and new contiguous
            ``call.scale_dtype`` ``[*S, call.k // call.group_size]`` scales. With
            ``call.inplace``, *input_tensor* is overwritten with each element's code times its
            scale and ``(None, None)`` returns.
        """
