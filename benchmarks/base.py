"""The benchmark a bench file writes: a workload in, a recorded result out.

Timing lives in :mod:`benchmarks.timing`, reporting in :mod:`benchmarks.report`. Both
are re-exported here, so a bench file keeps importing what it always did.
"""

import statistics
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

import pytest
import torch

from benchmarks.report import BenchmarkReport
from benchmarks.timing import (
    CUPTIError,
    _capture_bench_meta,
    _sample_spread_ms,
    bench_kernel,
)
from tileops.manifest import (
    WORKLOAD_RESERVED_KEYS,
    load_manifest,
    load_workloads,
    single_input_workload_contract,
)

__all__ = [
    "BenchmarkBase",
    "BenchmarkReport",
    "CUPTIError",
    "ManifestBenchmark",
    "bench_kernel",
    "workload_field_params",
    "workloads_to_params",
]

W = TypeVar("W")


def _workload_contract(op_name: str) -> tuple[str, frozenset[str]]:
    """Resolve the shared workload contract for an op known to exist."""
    sig = load_manifest()[op_name].get("signature") or {}
    contract = single_input_workload_contract(sig)
    if contract is None:
        raise KeyError(
            f"workloads_to_params({op_name!r}) needs exactly one manifest "
            "tensor input; multi-input ops use their own bench files."
        )
    return contract


class BenchmarkBase(Generic[W], ABC):
    """Abstract base class for op benchmarking.

    Generic over workload type so subclasses can declare the exact
    capability they need.  ``WorkloadBase`` remains the typical in-repo
    implementation, but the public contract is the type parameter.

    Subclass must implement calculate_flops() and calculate_memory().
    """

    def __init__(self, workload: W):
        self.workload = workload

    @abstractmethod
    def calculate_flops(self) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def calculate_memory(self) -> Optional[float]:
        raise NotImplementedError

    def profile(self, functor: Any, *inputs: Any) -> dict:
        """Profile a callable and return its structured result."""
        with torch.no_grad():
            return self._build_result(bench_kernel(functor, args=inputs))

    def profile_autograd(self, functor: Any) -> dict:
        """Profile a zero-arg closure that builds an autograd graph."""
        return self._build_result(bench_kernel(functor))

    def compare(
        self,
        functors: dict[str, Any],
        *inputs: Any,
        record_as: Any = None,
        params: Optional[dict] = None,
    ) -> dict[str, dict]:
        """Time several implementations against each other and record them all.

        Each implementation is timed twice, once in the given order and once in
        reverse, so clock and thermal drift across the case lands on all of them
        equally instead of on whichever ran last.

        A value is either a callable, timed on the shared *inputs*, or a
        ``(callable, args)`` pair for an implementation that takes its own.
        """
        plan = {
            tag: value if isinstance(value, tuple) else (value, inputs)
            for tag, value in functors.items()
        }
        tags = list(plan)
        samples: dict[str, list[float]] = {tag: [] for tag in tags}
        meta: dict[str, dict] = {}
        for tag in tags + tags[::-1]:
            functor, args = plan[tag]
            with torch.no_grad():
                samples[tag].extend(bench_kernel(functor, args=args))
            meta[tag] = _capture_bench_meta()
        results = {tag: self._build_result(samples[tag], meta[tag]) for tag in tags}
        if record_as is not None:
            for tag in tags:
                BenchmarkReport.record(record_as, params or {}, results[tag], tag=tag)
        return results

    def _build_result(self, samples: list[float], meta: Optional[dict] = None) -> dict:
        if not samples:
            raise ValueError("bench_kernel returned no samples")
        latency = statistics.median(samples)
        result = {"latency_ms": latency, "n_samples": len(samples)}
        p10, p90 = _sample_spread_ms(samples)
        if p10 is not None:
            result["latency_p10_ms"], result["latency_p90_ms"] = p10, p90
        # How the number was measured must travel with it: a run that fell back
        # to CUDA events is not comparable with a CUPTI-timed one.
        result.update(meta if meta is not None else _capture_bench_meta())
        flops = self.calculate_flops()
        if flops is not None:
            result["tflops"] = flops / latency * 1e-9
        memory = self.calculate_memory()
        if memory is not None:
            result["bandwidth_tbs"] = memory / latency * 1e-9
        return result


# Manifest-driven benchmark helpers


def _workload_extra_params(w: dict, shape_key: str) -> dict[str, Any]:
    """Return op-call params on a workload entry, stripping reserved keys."""
    reserved = WORKLOAD_RESERVED_KEYS | {shape_key}
    return {
        k: v
        for k, v in w.items()
        if isinstance(k, str) and k not in reserved and not k.startswith("__")
    }


def workloads_to_params(op_name: str, include_extra: bool = False) -> list:
    """Convert manifest workload dicts for *op_name* to pytest params.

    Each entry becomes ``pytest.param(shape, dtype, id=...)``; with
    ``include_extra=True`` a third element carries the op-call params
    declared on the workload entry (e.g. ``{"dim": 0}``).
    """
    workloads = load_workloads(op_name)  # canonical not-found error
    shape_key, allowed = _workload_contract(op_name)
    params = []
    for w in workloads:
        if shape_key not in w:
            raise KeyError(
                f"workload {w.get('label', w)!r} of {op_name!r} is missing "
                f"{shape_key!r} (derived from the signature's input name)."
            )
        unknown = sorted(
            repr(k) for k in w
            if not isinstance(k, str) or (k not in allowed and not k.startswith("__"))
        )
        if unknown:
            raise KeyError(
                f"workload {w.get('label', w)!r} of {op_name!r} has unknown "
                f"keys {unknown}; allowed: {sorted(allowed)}."
            )
        shape = tuple(w[shape_key])
        label = w.get("label", "x".join(str(s) for s in shape))
        extra = _workload_extra_params(w, shape_key) if include_extra else {}
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            # Copy ``extra`` per parametrization so mutation in one test case
            # cannot leak into later cases sharing the workload entry.
            param_args = (
                (shape, dtype, dict(extra))
                if include_extra
                else (shape, dtype)
            )
            params.append(pytest.param(*param_args, id=f"{label}-{dtype_str}"))
    return params


def workload_field_params(workloads: list, keys: tuple) -> list:
    """Turn manifest workload dicts into pytest params.

    First workload is marked ``smoke``, the rest ``full``. Keys ending in
    ``dtype`` are resolved to ``torch.dtype`` values.
    """
    params = []
    for i, w in enumerate(workloads):
        args = [getattr(torch, w[k]) if k.endswith("dtype") else w[k] for k in keys]
        params.append(
            pytest.param(
                *args,
                marks=pytest.mark.smoke if i == 0 else pytest.mark.full,
                id=w["label"],
            )
        )
    return params


class ManifestBenchmark(BenchmarkBase[Any]):
    """Generic benchmark that reads FLOP/memory counts from an Op instance.

    Accepts an op name, an instantiated Op, and the workload that produced
    the inputs.  Roofline numbers come from ``op.eval_roofline()``, so the
    workload needs no ``shape`` / ``dtype`` metadata — it is retained only
    for subclasses that read their own fields off it.  Dynamic-shape ops may
    bind roofline variables during ``forward()``, so this helper calls
    ``op.eval_roofline()`` only while building a result after profiling has
    executed the op.

    Usage::

        op = SumFwdOp(dim=0)
        bm = ManifestBenchmark("SumFwdOp", op, workload)
        result = bm.profile(op, *inputs)
    """

    def __init__(
        self,
        op_name: str,
        op: Any,
        workload: Any,
    ):
        super().__init__(workload)
        self._op_name = op_name
        self._op = op
        self._roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            flops, mem_bytes = self._op.eval_roofline()
            self._roofline_cache = (float(flops), float(mem_bytes))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]
