"""How a call is timed: CUPTI activity collection, and the loop around it.

The timing layer owns the measurement and nothing else -- it knows how to run a callable
n times and return each run's device latency. What the numbers mean, and where they are
written, belong to the layers above.
"""

import contextlib
import logging
import os
import sys
import threading
from typing import Any, Callable, Optional

import torch

_logger = logging.getLogger("tileops.bench")

# Per-phase wall-time budgets in ms; iteration counts derive from them, so a
# short kernel gets many samples and a long one few.
DRY_RUN_MS = 25.0
REPEAT_MS = 100.0
_CALIBRATION_ITERS = 3
# Counts are clamped: attribution requires every repeat to reach the GPU, so an
# unbounded count turns one hiccup into a failed case.
_MIN_ITERS = 10
_MAX_ITERS = 200

# Latest bench_kernel measurement metadata; deviations from the default protocol are
# surfaced in results by BenchmarkBase._build_result.
_bench_meta = threading.local()
_cuda_runtime = None

# CUPTI activity collection, via NVIDIA's cupti-python binding.

_CUPTI = None
_COLLECTOR_ACTIVE = False
_CALLBACKS_REGISTERED = False
_BUFFER_BYTES = 8 * 1024 * 1024
_BUFFER_ALIGN = 8
_RECORDS: list[dict[str, Any]] = []
# correlation id of a CUDA API call -> the external id in force when it was made.
_EXTERNAL_IDS: dict[int, int] = {}
# A record's kind arrives as its numeric value; resolving the enum per record costs more
# than the callback can afford. Filled by _load_cupti.
_KERNEL_KIND = -1
_EXTERNAL_KIND = -1

# Preparing a call (input rotation, L2 flush) is region 0 whichever iteration it belongs
# to -- its activity is only ever excluded. Iteration i's call is region i + 1.
_PREPARE_REGION = 0

# L2 cache flush buffer, allocated lazily.
_l2_flush_cache: Optional[torch.Tensor] = None


def _clamp_iters(raw: float) -> int:
    return max(_MIN_ITERS, min(_MAX_ITERS, int(raw)))


class _CUPTIAttributionError(Exception):
    """A CUPTI trace could not be attributed to logical benchmark calls."""


class CUPTIError(RuntimeError):
    """The CUPTI collector is unavailable or could not be operated."""


def _load_cupti():
    global _CUPTI
    if _CUPTI is not None:
        return _CUPTI
    try:
        from cupti import cupti
    except Exception as exc:  # noqa: BLE001
        raise CUPTIError(
            "cupti-python is unavailable. Install it with "
            "`pip install --no-deps cupti-python==13.2.0`; --no-deps is required "
            "or it downgrades torch's cuda-bindings pin."
        ) from exc
    global _KERNEL_KIND, _EXTERNAL_KIND
    _KERNEL_KIND = int(cupti.ActivityKind.CONCURRENT_KERNEL)
    _EXTERNAL_KIND = int(cupti.ActivityKind.EXTERNAL_CORRELATION)
    _CUPTI = cupti
    return _CUPTI


def _buffer_requested():
    return _BUFFER_BYTES, _BUFFER_ALIGN


def _buffer_completed(records) -> None:
    # Copy the fields out and keep no record alive: the binding's other
    # accessors misread a newer libcupti's struct and raise, including from
    # __del__ at shutdown.
    for record in records:
        if int(record.kind) == _EXTERNAL_KIND:
            _EXTERNAL_IDS[int(record.correlation_id)] = int(record.external_id)
            continue
        if int(record.kind) != _KERNEL_KIND:
            continue
        _RECORDS.append({
            "name": str(record.name),
            "start_ns": int(record.start),
            "end_ns": int(record.end),
            "correlation_id": int(record.correlation_id),
        })


@contextlib.contextmanager
def _phase_session():
    """Own one session, so a failed trial leaves nothing behind for the next."""
    global _COLLECTOR_ACTIVE, _CALLBACKS_REGISTERED
    if _COLLECTOR_ACTIVE:
        raise RuntimeError("CUPTI collector is already active")
    cupti = _load_cupti()
    # A kernel record names the API call that launched it, an external correlation
    # record names the region that call came from; neither half alone says which
    # iteration a kernel belongs to. torch launches through the runtime API and
    # TileLang through the driver API, so both or one layer goes unattributed.
    kinds = (
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.EXTERNAL_CORRELATION,
    )
    try:
        if not _CALLBACKS_REGISTERED:
            cupti.activity_register_callbacks(_buffer_requested, _buffer_completed)
            _CALLBACKS_REGISTERED = True
        _RECORDS.clear()
        _EXTERNAL_IDS.clear()
        for kind in kinds:
            cupti.activity_enable(kind)
    except Exception as exc:  # noqa: BLE001
        raise CUPTIError(f"CUPTI collector failed to start: {exc}") from exc
    _COLLECTOR_ACTIVE = True
    try:
        yield
    finally:
        _COLLECTOR_ACTIVE = False
        try:
            for kind in kinds:
                cupti.activity_disable(kind)
        except Exception as exc:  # noqa: BLE001
            raise CUPTIError(f"CUPTI collector failed to stop: {exc}") from exc


@contextlib.contextmanager
def _region(external_id: int):
    """Tag every CUDA API call made in this block with *external_id*."""
    cupti = _load_cupti()
    kind = cupti.ExternalCorrelationKind.CUSTOM2  # CUSTOM0/1 are Kineto's
    cupti.activity_push_external_correlation_id(kind, external_id)
    try:
        yield
    finally:
        cupti.activity_pop_external_correlation_id(kind)


def _flush() -> list[dict[str, Any]]:
    """Return the records completed since the previous flush."""
    cupti = _load_cupti()
    torch.cuda.synchronize()
    try:
        cupti.activity_flush_all(1)  # CUPTI_ACTIVITY_FLAG_FLUSH_FORCED
    except Exception as exc:  # noqa: BLE001
        raise CUPTIError(f"CUPTI flush failed: {exc}") from exc
    drained = list(_RECORDS)
    _RECORDS.clear()
    return drained


def collect_repeats(
    run_one: Callable[[int], None],
    n_repeat: int,
    prepare_one: Callable[[int], None],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Run the timed repeats, tagging each region so its activity is identifiable."""
    with _phase_session():
        for i in range(n_repeat):
            with _region(_PREPARE_REGION):
                prepare_one(i)
            with _region(i + 1):
                run_one(i)
        return _flush(), dict(_EXTERNAL_IDS)


def _kernel_span_us(kernels: list[dict]) -> float:
    if not kernels:
        return 0.0
    start_ns = min(int(kernel["start_ns"]) for kernel in kernels)
    end_ns = max(int(kernel["end_ns"]) for kernel in kernels)
    return (end_ns - start_ns) / 1000.0


def _ordered_trace_kernels(records: list[dict]) -> list[dict]:
    return sorted(
        records,
        key=lambda kernel: (int(kernel["start_ns"]), int(kernel["end_ns"])),
    )


def _cuda_events_fallback_enabled() -> bool:
    return os.getenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "0") == "1"


def _attributed_latency_samples_ms(
    records: list[dict],
    external_ids: dict[int, int],
    n_repeat: int,
) -> list[float]:
    """Return each iteration's operator latency, in milliseconds.

    Grouping by region is the whole of the attribution: nothing is inferred from the
    order or the identity of what ran, so a call whose kernel count varies between
    iterations is measured rather than rejected.
    """
    by_region: dict[int, list[dict]] = {}
    untagged: list[str] = []
    for kernel in records:
        external_id = external_ids.get(kernel["correlation_id"])
        if external_id is None:
            untagged.append(kernel["name"])
            continue
        by_region.setdefault(external_id, []).append(kernel)

    if untagged:
        raise _CUPTIAttributionError(
            f"{len(untagged)} of {len(records)} CUPTI kernel records belong to no timed "
            f"region, so the trial is missing activity it cannot account for: "
            f"{', '.join(sorted(set(untagged))[:4])}"
        )

    samples_us = []
    empty = []
    for repeat in range(n_repeat):
        operator = by_region.get(repeat + 1)
        if not operator:
            empty.append(repeat)
            continue
        samples_us.append(_kernel_span_us(operator))

    if empty:
        raise _CUPTIAttributionError(
            f"{len(empty)} of {n_repeat} timed iterations launched no CUDA kernel "
            f"(first: iteration {empty[0]}); a call that reaches the GPU in none of "
            f"them is not being measured"
        )
    return [sample_us * 1e-3 for sample_us in samples_us]


def _sample_spread_ms(samples: list[float]) -> tuple[float, float] | tuple[None, None]:
    """Return the 10th and 90th percentile of one op's timed samples.

    The reported latency is a median. Without the spread around it, a stable
    measurement and one dominated by launch jitter read the same downstream.
    """
    if len(samples) < 2:
        return None, None
    ordered = sorted(samples)
    last = len(ordered) - 1
    # Nearest rank, rounded rather than truncated: truncating collapses both
    # percentiles onto the minimum for small sample counts.
    return ordered[round(0.1 * last)], ordered[round(0.9 * last)]


# L2 cache flush buffer, allocated lazily.

_l2_flush_cache: Optional[torch.Tensor] = None


def _reset_persisting_l2_cache() -> None:
    global _cuda_runtime
    if _cuda_runtime is None:
        from cuda.bindings import runtime as cuda_runtime

        _cuda_runtime = cuda_runtime

    result = _cuda_runtime.cudaCtxResetPersistingL2Cache()
    if isinstance(result, tuple):
        result = result[0]
    torch.cuda.check_error(int(result))


def _get_l2_flush_cache() -> torch.Tensor:
    global _l2_flush_cache
    if _l2_flush_cache is None:
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
        if l2_bytes <= 0:
            _logger.warning(
                "L2 cache size query returned %d; flushing a 256 MB buffer "
                "instead", l2_bytes,
            )
            l2_bytes = int(256e6)
        _l2_flush_cache = torch.empty(2 * l2_bytes, dtype=torch.int8, device="cuda")
    return _l2_flush_cache


def _native_output_suppressor():
    """Return an fd-level output suppressor that is safe under pytest capture.

    tilelang's ``suppress_stdout_stderr`` dup2's ``/dev/null`` over
    ``sys.stdout.fileno()``; under pytest fd capture that fileno is the
    capture tmpfile and the redirect corrupts it (``EBADF`` on later reads).
    Suppress only when stdout/stderr are the process fds 1/2.
    """
    try:
        native = sys.stdout.fileno() == 1 and sys.stderr.fileno() == 2
    except (AttributeError, OSError, ValueError):
        # Streams without a real descriptor (io.StringIO, capsys) or with
        # fileno() unsupported: fd-level suppression is impossible.
        native = False
    if not native:
        return contextlib.nullcontext()
    from tilelang.profiler.bench import suppress_stdout_stderr
    return suppress_stdout_stderr()


# NVIDIA SOL-ExecBench–style benchmark


def _capture_bench_meta() -> dict:
    """Snapshot how the last measurement was taken."""
    return {
        key: value
        for key in ("timing", "fallback_reason")
        if (value := getattr(_bench_meta, key, None)) is not None
    }


def bench_kernel(
    fn: Callable,
    args: tuple[Any, ...] = (),
    dry_run_ms: float = DRY_RUN_MS,
    repeat_ms: float = REPEAT_MS,
) -> list[float]:
    """Time *fn* with CUPTI kernel-activity attribution.

    A calibration pass measures one iteration, then warmup and measurement each
    run for their millisecond budget, so a short op is sampled many times and a
    long one few. L2 is cleared and inputs rotated before every iteration. Each
    call spans the earliest to the latest activity of its discovered sequence,
    keeping inter-kernel gaps. Attribution fails closed unless
    ``TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=1``.

    Returns:
        Per-iteration latencies in **milliseconds**.
    """
    if not isinstance(args, tuple):
        raise TypeError(
            f"bench_kernel expects a tuple of args, got {type(args).__name__}. "
            "Check that gen_inputs() returns a tuple."
        )

    allow_fallback = _cuda_events_fallback_enabled()
    _bench_meta.timing = None
    _bench_meta.fallback_reason = None
    cache = _get_l2_flush_cache()

    def _flush_l2():
        _reset_persisting_l2_cache()
        cache.zero_()

    # Calibrate on the raw args, before the pool exists to be sized. The flush
    # is inside the timed region, so counts self-limit for tiny kernels.
    def _call_raw():
        return fn(*args) if args else fn()

    _flush_l2()
    _call_raw()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_CALIBRATION_ITERS):
        _flush_l2()
        _call_raw()
    end.record()
    torch.cuda.synchronize()
    per_iter_ms = max(start.elapsed_time(end) / _CALIBRATION_ITERS, 1e-6)

    n_warmup = _clamp_iters(dry_run_ms / per_iter_ms)
    n_repeat = _clamp_iters(repeat_ms / per_iter_ms)

    def _run(i):
        return fn(*args) if args else fn()

    def _prepare_iteration(i):
        _flush_l2()
        torch.cuda.synchronize()

    _prepare_iteration(0)
    _run(0)
    torch.cuda.synchronize()
    for i in range(n_warmup):
        _prepare_iteration(i)
        _run(i)
    torch.cuda.synchronize()

    try:
        with _native_output_suppressor():
            records, external_ids = collect_repeats(_run, n_repeat, _prepare_iteration)
            samples = _attributed_latency_samples_ms(records, external_ids, n_repeat)
        _bench_meta.timing = "cupti"
    except (_CUPTIAttributionError, CUPTIError) as exc:
        if not allow_fallback:
            raise RuntimeError(
                f"CUPTI profiling failed: {exc}. CUDA-events fallback is disabled "
                "(TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0), which keeps the run from "
                "silently mixing two timing methods."
            ) from exc
        _bench_meta.timing = "cuda-events"
        _bench_meta.fallback_reason = str(exc)
        _logger.warning("CUPTI timing failed (%s); falling back to CUDA events.", exc)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        for i in range(n_repeat):
            _prepare_iteration(i)
            starts[i].record()
            _run(i)
            ends[i].record()
        torch.cuda.synchronize()
        samples = [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]

    torch.cuda.empty_cache()
    return samples
