# Op Structural Compliance Checklist

Items marked **[REQUIRED]** must pass — block the PR if they fail.
Items marked **[RECOMMENDED]** should pass — note in PR body if skipped with reason.

## Correctness & Safety

- [ ] **[REQUIRED]** `Kernel.__init__` validates dtype against `SUPPORTED_DTYPES` and raises `ValueError` for unsupported types — template kernels (`UnaryKernel`, `BinaryKernel`, `FusedGatedKernel`) inherit this from their base class; independent kernels that inherit from `Kernel` ABC directly must add validation explicitly
- [ ] **[REQUIRED]** No hardcoded narrow-type constants in kernels (e.g. `T.cast(1.0, "float16")`) — use `x.dtype` or an explicit wide intermediate type
- [ ] **[REQUIRED]** fp16/bf16 intermediate math that can overflow (cubic terms, division, exp) is promoted to fp32
- [ ] **[REQUIRED]** Runtime validation uses `ValueError`/`TypeError`, never `assert` (stripped under `python -O`)
- [ ] **[REQUIRED]** Output dtype matches PyTorch reference semantics (e.g. comparison ops → `bool`, not float 0/1)

## Kernel Structure

- [ ] **[REQUIRED]** `with T.Kernel()` is inside `@T.prim_func`; complex kernels factor reusable sub-routines into `@T.macro` helpers called from within the `T.Kernel` scope
- [ ] **[REQUIRED]** `Kernel.forward` accepts only GPU tensors and only calls `self.kernel(config...)(tensors)` — no format conversion, batching, or dtype cast
- [ ] **[RECOMMENDED]** `_<op_name>_kernel(static_params) -> Callable` closure function exists
- [ ] **[RECOMMENDED]** `@tilelang.jit(out_idx=[...])` wraps a config-parameterised inner function
- [ ] **[RECOMMENDED]** No Python builtins (`float()`, `int()`, `math.cos()`) on TileLang IR nodes
- [ ] **[RECOMMENDED]** Tile-ops (`T.clear`, `T.copy`, `T.gemm`) are at `T.Kernel` scope, not inside `T.Parallel`

## Op Structure

- [ ] **[REQUIRED]** `Op.forward` owns all pre/post-processing (reshape, contiguous, dtype cast); delegates GPU work to `self.kernel(...)` only
- [ ] **[REQUIRED]** `@torch.library.custom_op` + `.register_fake` wrapper exists for torch.compile compatibility
- [ ] **[RECOMMENDED]** `default_config` and `autotune_configs` properties are defined on the Kernel class
- [ ] **[RECOMMENDED]** `supported_archs` class attribute is set on the Kernel class
- [ ] **[RECOMMENDED]** `accum_dtype` is hardcoded in kernel — never a property, config key, or parameter
- [ ] **[RECOMMENDED]** Template-based Ops: `__init__` signature ends with `kernel_map=None, tune=False`; `dispatch_kernel(kernel_map)` called before kernel use. Independent Ops: `Kernel.__init__` signature ends with `config=None, tune=False`

## Delivery

- [ ] **[REQUIRED]** Unit tests in `tests/ops/` with reference comparison (FP16 atol=1e-3, BF16 atol=1.6e-2)
- [ ] **[REQUIRED]** Tests cover unsupported-dtype rejection paths (expect `ValueError`)
- [ ] **[REQUIRED]** Dtype support matrix documented in PR body
- [ ] **[REQUIRED]** No issue references (`#123`, `TODO: see #456`) in source or test files — issues track goals in GitHub, not in code
- [ ] **[RECOMMENDED]** Benchmark class in `benchmarks/`
- [ ] **[RECOMMENDED]** `__init__.py` exports are synchronized (`__all__` + explicit re-exports)
