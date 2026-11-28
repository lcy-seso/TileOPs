"""Call records and interfaces of the indexing kernels."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod

import torch

from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import KernelInterface

__all__ = [
    "ConcatCall",
    "ConcatFwdInterface",
    "EmbeddingCall",
    "EmbeddingFwdInterface",
    "MaskedScatterCall",
    "MaskedScatterFwdInterface",
]


@dataclasses.dataclass(frozen=True)
class ConcatCall(CallSpec):
    """The joined axis, each input's row width and dtype, and the output dtype.

    An input's row width is the product of its extents from ``dim`` on: the contiguous run of
    elements it holds for one index of the axes before ``dim``.
    """

    dim: int = 0
    widths: tuple[int, ...] = ()
    dtypes: tuple[torch.dtype, ...] = ()
    out_dtype: torch.dtype | None = None


class ConcatFwdInterface(KernelInterface):
    """Tensors joined along one axis into a new tensor."""

    request = ConcatCall

    @abstractmethod
    def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Join *tensors* along ``call.dim``.

        Args:
            tensors: Contiguous tensors on ``call.device`` of one rank, equal in every axis
                but ``call.dim``. Element ``j`` has dtype ``call.dtypes[j]`` and row width
                ``call.widths[j]``.

        Returns:
            A new contiguous tensor in ``call.out_dtype``: each input converted to it and
            placed after the previous inputs along ``call.dim``.
        """


@dataclasses.dataclass(frozen=True)
class EmbeddingCall(CallSpec):
    """The row width and dtypes of one lookup; the index count and table rows are dynamic."""

    dim: int = 0
    dtype: torch.dtype | None = None
    index_dtype: torch.dtype | None = None
    renorm: bool = False


class EmbeddingFwdInterface(KernelInterface):
    """Gather the table rows the indices name."""

    request = EmbeddingCall

    @abstractmethod
    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Copy row ``input[...]`` of *weight* for every index.

        Args:
            input: Indices in ``[0, V)``, any shape, in ``call.index_dtype``, contiguous.
            weight: The table ``[V, call.dim]`` in ``call.dtype``, contiguous.

        Returns:
            A new tensor ``[*input.shape, call.dim]`` in ``call.dtype`` on ``call.device``.
        """


@dataclasses.dataclass(frozen=True)
class MaskedScatterCall(CallSpec):
    """The run length and dtype of one call; the run count and source size are dynamic."""

    run_len: int = 0
    dtype: torch.dtype | None = None


class MaskedScatterFwdInterface(KernelInterface):
    """Write source elements into the true positions of a broadcast mask."""

    request = MaskedScatterCall

    @abstractmethod
    def forward(
        self, input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Return ``input`` broadcast with ``mask``, its true positions filled from ``source``.

        Args:
            input: Any shape, in ``call.dtype``, contiguous.
            mask: bool, broadcastable with ``input``, contiguous; the output's trailing runs
                of ``call.run_len`` elements each share one mask value.
            source: Any shape in ``call.dtype``, contiguous, holding at least as many
                elements as the broadcast mask has true positions.

        Returns:
            A new tensor of the broadcast shape in ``call.dtype``; the k-th true position in
            row-major order holds ``source.flatten()[k]``.
        """
