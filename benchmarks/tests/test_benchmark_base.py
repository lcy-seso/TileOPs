"""Unit tests for benchmarks.benchmark_base.

Verifies that the generic ``BenchmarkBase`` / ``ManifestBenchmark`` accept
any duck-typed workload rather than requiring ``WorkloadBase`` inheritance.
"""

import pytest
import torch

from benchmarks.benchmark_base import (
    _attributed_latency_samples_ms,
    _CUPTIAttributionError,
    _ShiftingTensorPool,
    bench_kernel,
    workloads_to_params,
)

# Duck-typed test workloads


@pytest.mark.smoke
def test_workloads_to_params_include_extra_propagates_dim():
    """When a workload entry carries ``dim``, ``include_extra=True`` should
    surface it in the pytest param triple.
    """
    # End-to-end with the manifest: include_extra=True must still yield
    # well-formed triples with the (shape, dtype, extra) mapping. The
    # contract being asserted is per-triple shape/dtype/extra typing; it
    # must not depend on the ordering of SumFwdOp.workloads (which is QA
    # curated and may be reordered without regressing the helper).
    triples = workloads_to_params("SumFwdOp", include_extra=True)
    assert len(triples) > 0
    assert any("dim" in p.values[2] for p in triples), (
        "at least one SumFwdOp workload must propagate a dim param"
    )
    for p in triples:
        shape, dtype, extra = p.values
        assert isinstance(shape, tuple)
        assert isinstance(dtype, torch.dtype)
        assert isinstance(extra, dict)
    # A workload with no extras must yield an empty dict, not a missing slot.
    assert any(p.values[2] == {} for p in triples)


def test_multi_input_op_raises_keyerror():
    """Multi-input ops (q/k/v) raise instead of binding a wrong tensor."""
    with pytest.raises(KeyError, match="exactly one manifest tensor input"):
        workloads_to_params("GroupedQueryAttentionFwdOp")


def _kernel(name: str, start_ns: int, end_ns: int, correlation_id: int) -> dict:
    return {
        "name": name,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "correlation_id": correlation_id,
    }


def test_attribution_excludes_prepare_and_keeps_the_operator_gap():
    """Prepare activity is identified by its region, not by where it sits."""
    records = [
        _kernel("copy", 1_000, 2_000, 1),
        _kernel("fill", 2_100, 3_000, 2),
        _kernel("op-a", 4_000, 6_000, 3),
        _kernel("op-b", 9_000, 10_000, 4),
        _kernel("copy", 20_000, 21_000, 5),
        _kernel("op-a", 23_000, 24_000, 6),
        _kernel("op-b", 29_000, 31_000, 7),
    ]
    # Region 0 is every prepare block; iteration i's call is region i + 1.
    external_ids = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 2, 7: 2}

    samples_ms = _attributed_latency_samples_ms(records, external_ids, n_repeat=2)

    # Operator envelopes are 6 us and 8 us; the 3/5 us inter-kernel gaps stay inside the
    # call, and iteration 1 preparing with one kernel instead of two changes nothing.
    assert samples_ms == pytest.approx([0.006, 0.008])


def test_attribution_measures_a_call_whose_kernel_count_varies():
    """A dynamic path launching an extra kernel is measured, not rejected."""
    records = [
        _kernel("op", 1_000, 2_000, 1),
        _kernel("op", 10_000, 11_000, 2),
        _kernel("op-extra", 11_500, 13_000, 3),
    ]

    samples_ms = _attributed_latency_samples_ms(
        records, {1: 1, 2: 2, 3: 2}, n_repeat=2,
    )

    assert samples_ms == pytest.approx([0.001, 0.003])


@pytest.mark.parametrize(
    "records, external_ids, message",
    [
        # A kernel no region claims: activity the trial cannot account for.
        (
            [_kernel("a", 1_000, 2_000, 1), _kernel("stray", 2_000, 3_000, 99)],
            {1: 1},
            "belong to no timed region.*stray",
        ),
        # An iteration whose call never reached the GPU.
        ([_kernel("a", 1_000, 2_000, 1)], {1: 1}, "launched no CUDA kernel"),
    ],
)
def test_attribution_fails_closed(records, external_ids, message):
    with pytest.raises(_CUPTIAttributionError, match=message):
        _attributed_latency_samples_ms(records, external_ids, n_repeat=2)


def test_shifting_tensor_pool_preserves_layout_values_and_alignment():
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
    pool = _ShiftingTensorPool((source, 7), total_iterations=3, seed=123)
    pointers = []

    for _ in range(3):
        shifted, scalar = pool.next_args()
        assert scalar == 7
        assert shifted.stride() == source.stride()
        torch.testing.assert_close(shifted, source)
        pointers.append(shifted.data_ptr())
        shifted.zero_()

    assert len(set(pointers)) == 3
    assert all(
        (pointer - pointers[0]) % _ShiftingTensorPool._POOL_ALIGNMENT == 0
        for pointer in pointers[1:]
    )
    expected = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
    torch.testing.assert_close(source, expected)
    with pytest.raises(RuntimeError, match="ShiftingTensorPool exhausted"):
        pool.next_args()


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_cupti_failure_fails_closed_by_default(monkeypatch):
    """A callable launching no CUDA kernel cannot be attributed by CUPTI."""
    monkeypatch.setenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "0")
    with pytest.raises(RuntimeError, match="CUDA-events fallback is disabled"):
        bench_kernel(lambda: sum(range(64)))


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kernel_runtime_error_propagates():
    """Genuine RuntimeErrors must reach the caller, not the fallback path."""
    def boom():
        raise RuntimeError("kernel failure")

    with pytest.raises(RuntimeError, match="kernel failure"):
        bench_kernel(boom)
