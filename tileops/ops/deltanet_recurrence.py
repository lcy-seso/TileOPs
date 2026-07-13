from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.deltanet_recurrence import (
    DeltaNetDecodeFP32Kernel,
    DeltaNetDecodeKernel,
    DeltaNetDecodeRawCudaFlaStyleKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

from .op_base import Op

__all__ = ["DeltaNetDecodeOp"]


class DeltaNetDecodeOp(Op):
    """DeltaNet decode (single-step recurrence, ungated).

    Computes one step of the delta rule (no gate):
        v_new = beta * (v - S @ k)
        o     = S @ q + (q . k) * v_new
        S_new = S + outer(k, v_new)

    Layout: BHD (batch, head, dim).
    Supports float32, float16, and bfloat16 with fp32 accumulation.

    For fp32 dtype, dispatches to a dedicated FP32 kernel that uses
    element-wise matvec instead of T.gemm to avoid TF32 mantissa truncation.
    """

    @staticmethod
    def _raw_cuda_decode_arch_supported() -> bool:
        try:
            sm_version = get_sm_version()
        except Exception:
            return False
        return sm_version in DeltaNetDecodeRawCudaFlaStyleKernel.supported_archs

    @staticmethod
    def _should_use_raw_cuda_decode(
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        tune: bool,
    ) -> bool:
        if dtype not in (torch.float16, torch.bfloat16) or dim_k != 128 or dim_v != 128:
            return False
        return DeltaNetDecodeOp._raw_cuda_decode_arch_supported()

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)

        # Dispatch:
        #   fp32 -> FP32 kernel (no TF32)
        #   fp16/bf16 DK=DV=128 on Hopper -> raw CUDA warp-per-Vtile kernel
        #   other fp16/bf16 shapes -> default TileLang kernel
        use_raw_cuda_decode = self._should_use_raw_cuda_decode(dim_k, dim_v, dtype, tune)
        if dtype == torch.float32:
            kernel_cls = self.kernel_map["DeltaNetDecodeFP32Kernel"]
        elif use_raw_cuda_decode:
            kernel_cls = self.kernel_map["DeltaNetDecodeRawCudaFlaStyleKernel"]
        else:
            kernel_cls = self.kernel_map["DeltaNetDecodeKernel"]
        kernel_dtype = Kernel.dtype_to_str(dtype)
        self.kernel = kernel_cls(
            batch, heads, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernels = {
            "DeltaNetDecodeKernel": DeltaNetDecodeKernel,
            "DeltaNetDecodeFP32Kernel": DeltaNetDecodeFP32Kernel,
        }
        if self._raw_cuda_decode_arch_supported():
            kernels["DeltaNetDecodeRawCudaFlaStyleKernel"] = (
                DeltaNetDecodeRawCudaFlaStyleKernel
            )
        return kernels

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kernel(q, k, v, beta, state)
