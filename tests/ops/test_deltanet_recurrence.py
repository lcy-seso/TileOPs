

import pytest
import torch

import tileops.ops.deltanet_recurrence as deltanet_ops
import tileops.ops.op_base as op_base_ops
from tests.test_base import FixtureBase, TestBase
from tileops.kernels.deltanet_recurrence import DeltaNetDecodeRawCudaFlaStyleKernel
from tileops.ops import DeltaNetDecodeOp
from workloads.deltanet import DeltaNetDecodeTest as _DeltaNetDecodeTestWorkload


def deltanet_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step delta rule (ungated)."""
    q, k, v = q.float(), k.float(), v.float()
    beta = beta.float()
    state = state.float()

    old_val = torch.einsum("bhkv,bhk->bhv", state, k)
    beta_unsq = beta.unsqueeze(-1)
    v_new = beta_unsq * (v - old_val)

    o_inter = torch.einsum("bhkv,bhk->bhv", state, q)
    qk_dot = torch.einsum("bhk,bhk->bh", q, k).unsqueeze(-1)
    o_intra = qk_dot * v_new
    o = o_inter + o_intra

    new_state = state + k.unsqueeze(-1) * v_new.unsqueeze(-2)

    return o, new_state


class DeltaNetDecodeTest(_DeltaNetDecodeTestWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        o, new_state = deltanet_decode_torch(q, k, v, beta, state)
        return o.to(self.dtype), new_state.to(self.dtype)


# =============================================================================
# Torch reference implementation (test-only)
# =============================================================================


# =============================================================================
# Correctness tests
# =============================================================================


def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 5e-4, "rtol": 5e-4}
    elif dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    else:  # bfloat16
        return {"atol": 2e-2, "rtol": 2e-2}


class DeltaNetDecodeFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, dim_k, dim_v, dtype, tune", [
            pytest.param(1, 4, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(1, 4, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 4, 64, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(2, 8, 64, 64, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 4, 128, 128, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 4, 128, 128, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 4, 128, 128, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 8, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 8, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@DeltaNetDecodeFixture
def test_deltanet_decode(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    test = DeltaNetDecodeTest(batch, heads, dim_k, dim_v, dtype)
    op = DeltaNetDecodeOp(batch, heads, dim_k, dim_v, dtype, tune=tune)
    tols = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), **tols)


@DeltaNetDecodeFixture
def test_deltanet_decode_multi_step(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """Test multiple sequential decode steps to verify state propagation."""
    torch.manual_seed(42)
    num_steps = 8
    B, H, DK, DV = batch, heads, dim_k, dim_v

    op = DeltaNetDecodeOp(B, H, DK, DV, dtype, tune=tune)
    tols = _get_tolerances(dtype)

    state_op = torch.zeros(B, H, DK, DV, device="cuda", dtype=dtype)
    state_ref = torch.zeros(B, H, DK, DV, device="cuda", dtype=dtype)

    for _ in range(num_steps):
        q = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=dtype) * 0.1
        beta = torch.rand(B, H, device="cuda", dtype=dtype) * 0.5

        o_ref, state_ref = deltanet_decode_torch(q, k, v, beta, state_ref)
        o_ref = o_ref.to(dtype)
        state_ref = state_ref.to(dtype)

        with torch.no_grad():
            o_op, state_op = op(q, k, v, beta, state_op)

        torch.testing.assert_close(o_op, o_ref, **tols)
        torch.testing.assert_close(state_op, state_ref, **tols)


class _DispatchMarkerKernel:
    supported_archs = None

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class _DefaultDispatchKernel(_DispatchMarkerKernel):
    pass


class _FP32DispatchKernel(_DispatchMarkerKernel):
    pass


class _RawDispatchKernel(_DispatchMarkerKernel):
    pass


def _dispatch_kernel_map() -> dict:
    return {
        "DeltaNetDecodeKernel": _DefaultDispatchKernel,
        "DeltaNetDecodeFP32Kernel": _FP32DispatchKernel,
        "DeltaNetDecodeRawCudaFlaStyleKernel": _RawDispatchKernel,
    }


def _mock_dispatch_arch(monkeypatch, sm_version: int) -> None:
    monkeypatch.setattr(deltanet_ops, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(op_base_ops, "get_sm_version", lambda: sm_version)


@pytest.mark.parametrize(("dtype", "tune"), [
    pytest.param(torch.float16, False, marks=pytest.mark.smoke, id="smoke-fp16"),
    pytest.param(torch.bfloat16, False, marks=pytest.mark.smoke, id="smoke-bf16"),
    pytest.param(torch.bfloat16, True, marks=pytest.mark.full, id="full-bf16-tuned"),
])
def test_deltanet_decode_raw_cuda_dispatch_selects_raw_on_supported_sm90(
    monkeypatch,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    _mock_dispatch_arch(monkeypatch, 90)

    op = DeltaNetDecodeOp(
        1,
        32,
        128,
        128,
        dtype=dtype,
        kernel_map=_dispatch_kernel_map(),
        tune=tune,
    )

    assert isinstance(op.kernel, _RawDispatchKernel)
    assert op.kernel.kwargs["tune"] is tune


@pytest.mark.smoke
def test_deltanet_decode_raw_cuda_dispatch_falls_back_on_unsupported_sm(
    monkeypatch,
) -> None:
    _mock_dispatch_arch(monkeypatch, 80)

    op = DeltaNetDecodeOp(
        1,
        32,
        128,
        128,
        dtype=torch.bfloat16,
        kernel_map=_dispatch_kernel_map(),
        tune=False,
    )

    assert isinstance(op.kernel, _DefaultDispatchKernel)


@pytest.mark.smoke
@pytest.mark.parametrize(("dim_k", "dim_v"), [(64, 128), (128, 64)])
def test_deltanet_decode_raw_cuda_dispatch_falls_back_on_non_128_shapes(
    monkeypatch,
    dim_k: int,
    dim_v: int,
) -> None:
    _mock_dispatch_arch(monkeypatch, 90)

    op = DeltaNetDecodeOp(
        1,
        32,
        dim_k,
        dim_v,
        dtype=torch.bfloat16,
        kernel_map=_dispatch_kernel_map(),
        tune=False,
    )

    assert isinstance(op.kernel, _DefaultDispatchKernel)
    assert op.kernel.kwargs["tune"] is False


@pytest.mark.smoke
def test_deltanet_decode_raw_cuda_dispatch_uses_fp32_kernel_for_fp32(
    monkeypatch,
) -> None:
    _mock_dispatch_arch(monkeypatch, 90)

    op = DeltaNetDecodeOp(
        1,
        32,
        128,
        128,
        dtype=torch.float32,
        kernel_map=_dispatch_kernel_map(),
        tune=False,
    )

    assert isinstance(op.kernel, _FP32DispatchKernel)
    assert op.kernel.kwargs["tune"] is False


@pytest.mark.smoke
def test_deltanet_decode_raw_cuda_config_requires_full_warp_mapping() -> None:
    with pytest.raises(ValueError, match="threads .* must equal raw_group_size \\* v_tile"):
        DeltaNetDecodeRawCudaFlaStyleKernel(
            1,
            32,
            128,
            128,
            dtype="bfloat16",
            config={
                "threads": 16,
                "v_tile": 16,
                "raw_group_size": 2,
                "raw_maxrregcount": 146,
            },
        )


@pytest.mark.smoke
def test_deltanet_decode_raw_cuda_config_requires_two_lane_group() -> None:
    with pytest.raises(ValueError, match="raw_group_size must equal 2"):
        DeltaNetDecodeRawCudaFlaStyleKernel(
            1,
            32,
            128,
            128,
            dtype="bfloat16",
            config={
                "threads": 32,
                "v_tile": 8,
                "raw_group_size": 4,
                "raw_maxrregcount": 146,
            },
        )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
