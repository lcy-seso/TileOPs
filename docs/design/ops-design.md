# Op Interface Design

Op responsibilities, implemented class structure, execution behavior and scaffolding guide. Interface details: [reference](ops-design-reference.md). Per-slot rules: [op-slot-rules.md](op-slot-rules.md).

## Concepts

Every operator is split into two classes — **Op** (host-side: validates inputs, dispatches to Kernel, assembles output) and **Kernel** (device-side: owns the TileLang program, tile configuration, JIT compilation). The two layers are independently modifiable — changing a Kernel's tile strategy does not require changing the Op.

### Op base responsibilities

The implementation groups three responsibilities in `Op` and its generated helpers:

| Responsibility        | Implementation                                                                                           |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| Contract              | `_Plan` generates construction, input, shape, effect and roofline functions from the manifest.           |
| Execution runtime     | `Op` binds targets, selects and caches entries, holds delegates, tunes kernels and manages active calls. |
| Execution observation | `Op.last_call`, resource queries and roofline methods expose completed calls and held resources.         |

`Op` and codegen jointly implement these three responsibilities.

### Class structure

![Op class diagram](diagrams/op-base.svg)

[PlantUML source](diagrams/op-base.puml). The diagram shows selected members of the implemented classes. Solid arrows are references, dashed arrows dependencies, and the hollow triangle inheritance. Terracotta italics mark abstract declarations; class names are bold. `_name` is an internal Python name and `[property]` a property, without enforced access restrictions.

- `Op` is abstract. Concrete classes implement `forward()`; manifest codegen supplies `_infer_output_shapes()`, `_validate_dtypes()` and `eval_roofline()`.
- Each manifest-backed class holds `_signature`, a shared `_Plan`. Parameters, `_construction_ix`, effect-branch caches, target binding and execution resources live on instances.
- `_Boundary` registers custom operators and generates `_call_boundary()`. The generated function retains the boundary object; operator bodies resolve an instance handle and invoke `Op._serve()`.
- `SignatureCall` holds checked indices, tensor shapes and dtypes, effects, metadata references and completed stage calls. `Op._signature_call` retains the last completed record.

`op_base.py` owns execution and observation. `_signature_codegen.py` defines `_Plan`, `_Boundary`, `SignatureCall` and generated methods. `_params_codegen.py` supplies parameter names through `PARAM_NAMES_ATTRIBUTE`. `compile_boundary.py` holds weak instance references for custom operators.

The manifest owns output order, shape, dtype and effects. Packed representations also follow the operator's declared semantics: `INT4QuantPerGroupFwdOp` returns signed `int8` packed weights and `float32` group parameters. `GemmW4A16FwdOp` consumes its own `uint8` weight layout with separate scales and zero points. A composition checks representation compatibility and supplies an explicit conversion where needed.

### Construction and execution

`Op` has no base constructor. A concrete constructor assigns its parameters, target and tuning policy, then calls `dispatch_kernel(kernel_map)`. That method loads registrations, checks construction, installs the kernel map and registers the instance. `_Plan.construct()` stores derived facts in `_construction_ix`; parameter attributes remain ordinary Python attributes. `target` is constructor-only by convention.

| Entry                         | Execution and recording                                                                                                                                                                                                                                           |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Eager call without a boundary | `__call__()` checks the signature, resolves the target, executes `forward()` or the target, validates results and retains the call.                                                                                                                               |
| Compile boundary              | Concrete `forward()` calls `_call_boundary()`; the real custom operator enters `_serve()`, which executes `_eager_forward()` or the target with checks and recording. `_eager_forward()` is a concrete implementation convention, not an abstract method on `Op`. |
| Traced composition            | `__call__()` skips its eager signature and recording path. The graph traces `forward()` and child boundaries; no parent record is produced by tracing.                                                                                                            |
| Empty writes                  | A call with at least one declared write and no elements in any write constructs the signature-defined result without binding or launching an implementation. It retains a `SignatureCall`.                                                                        |
| Meta/fake                     | The registered fake checks inputs and infers outputs. A direct eager meta call retains a record with empty stage lists; fake tracing does not replace `last_call`.                                                                                                |

Active calls use a thread-local stack. Success validates outputs before `_keep_call()`; `except Exception` drops the active call. A failure handled while initially unresolved invokes `_unsettle()`, which clears execution caches and recursively resets held delegates, including injected ones. A failure on an already settled instance does not invoke that reset. The previous record is retained. These handlers do not catch `BaseException` interruptions, and do not undo tensor writes.

### Execution observation

Resource queries describe currently held entries; `last_call` describes the most recently completed checked call. A cached specialization need not have run in that call. `run_config()` reports the instance config or first configured kernel, not necessarily the latest call's configuration.

`SignatureCall` is a frozen dataclass containing mutable mappings and references to metadata tensors. `values()` reads those tensors when analysis runs; later caller mutation can change the values an earlier record exposes. Meta records carry no metadata values. Stage collection maps child object identity to one stage and appends completed calls in order; holding the same child under multiple stages does not preserve distinct invocation labels.

Generated `eval_roofline()` evaluates the shared plan over `last_call`; `compute_roof()` and `roofline_inputs()` remain Op hooks. Formulas describe work and theoretical traffic, not timing or numerical correctness. Partial KV-cache writes may make the read/write split unavailable even when total bytes are known.

### Input validation and numerical verification

Generated checks enforce declared shapes, dtypes, placement and output effects. Metadata-value predicates designated as caller obligations remain the caller's responsibility. Ops exposing `validate_inputs=False`, including MeanPooling and GQA variants, can opt into synchronous content checks in their built-in bodies. Those checks run on each enabled call and cannot run during CUDA Graph capture; a whole-op target bypasses the built-in body. MeanPooling does not cache prior metadata validation.

Numerical verification belongs to workloads: `ref_program` consumes the live inputs and `verification()` supplies the shared comparison contract. Numerical tests execute through Op dispatch; an implementation or configuration is pinned with `kernel_map`. The Op's structural result checks do not run a reference computation. See [testing.md](testing.md#tests).

### Class hierarchy

```
Op                          ← L1: shared host-side infrastructure
  └── FamilyBase            ← L2: family-specific forward() flow (optional)
        └── ConcreteOp      ← L3: leaf class emitted by the scaffold
```

- **L1 (`Op`):** shared host-side plumbing (dispatch, get-or-build kernel caching, kernel enumeration, autotune) plus the methods generated from the manifest signature: the call checks, `_infer_output_shapes`, `_validate_dtypes` and `eval_roofline`.
- **L2 (`FamilyBase`):** per-family shared `forward()` pipeline (one per family). **Not produced by this playbook** — see [Family-Base Refactoring](#family-base-refactoring).
- **L3 (`ConcreteOp`):** this playbook's target. New ops start by inheriting L1 directly (T2 shape); see [Family-Base Refactoring](#family-base-refactoring) for when a family graduates to L2.

### Execution timing

**Do it at the first moment all required information is known, do it once, cache the result.** What an op knows at construction is its `signature.params`; every index the call's tensors carry is solved per call.

**Shape is not a constructor parameter when the tensors carry it.** Only `signature.params` belongs in `__init__`, plus the code-owned execution-policy parameters of [manifest.md table 7](manifest.md#t-policy). A dimension declared nowhere is not construction information: it arrives with the call, and taking it twice lets an instance disagree with the tensors it is handed. What the kernel is compiled for goes in the memory key instead, so a second shape builds a second kernel.

**Dtype is not a constructor parameter when the inputs determine it.** An op reads it from the input tensors in `forward()`: a caller who passes fp16 tensors gets the fp16 kernel without having said so twice, and an op can no longer be constructed in a state that disagrees with the tensors it is about to be handed.

**An output dtype is a dtype expression of the signature** — a `DType` index solved from the inputs, a constant, a dtype primitive, or a dtype parameter the caller passes at construction ([manifest.md](manifest.md#dtypes)).

The kernel is dtype-specialized, so this makes kernel construction uniformly deferred to the first `forward()` — for fixed-rank and arbitrary-rank ops alike — keyed by every input that selects a specialization, dtype among them. `dispatch_kernel()` stays in `__init__`: resolving the kernel *class* needs no tensor. It also needs no device, and must not ask for one — see [Kernel selection](#kernel-selection).

Generated signature checks enforce dtypes at eager and custom-op boundaries. `_validate_dtypes()` exposes the same checks; execution bodies repeat none. Traced compositions skip the parent check and retain their child boundaries. Roofline timing and formula semantics are in [roofline.md](roofline.md); see [Parameter Design](ops-design-reference.md#parameter-design) for fixed-rank vs arbitrary-rank details and [Codegen Details](ops-design-reference.md#codegen) for calling conventions.

### Kernel selection

**Construction reads no device property.** Installing the kernel map resolves classes only; a device that cannot run the op is refused when a kernel is first selected, built or called. Why: the tensors arrive later, perhaps on a device the process has not touched.

**A kernel interface is a place an op calls a kernel.** Its class publishes the call contract: the call spec, the tensors handed, what returns, and what `entry_for(call)` owes. An op opens one only where semantics or that contract changes. Why: a backend implements an interface from the contract alone.

**An implementation is a kernel class that inherits an interface, registered under a key.** It is one complete algorithm; tile sizes, split counts and fusion among fixed stages are its plan. Why: it is the unit a backend replaces and selection chooses.

**A call spec holds the immutable facts of one call.** Device facts derive from its device on a miss. Why: equal call specs denote one call, so one resolved entry serves every recurrence.

**Availability filters before selection.** An implementation is available where its `devices` and `supported_archs` allow. Why: where a kernel runs is a fact of the implementation, not of the calls it serves.

**Applicability states the calls an implementation serves, positively.** An op checks no implementation's limits, and the signature holds only what the algorithm requires ([manifest.md § Refinements](manifest.md#refinements)). Why: each implementation answers for itself.

Hardware resource limits also belong to implementation selection. `RMSNormFwdOp` exposes one `rms_norm` interface; its regular and streaming kernels declare their applicability and precedence. Shared-memory limits do not narrow the Op contract.

**Precedence picks among the available implementations that apply.** `general` is below every other, and `preferred_over` names the implementations one wins over, transitively and acyclically; no winner is an error, several an ambiguity. Why: a declared relation composes implementations unaware of each other, where order or numeric priority cannot.

**An entry is what `entry_for` builds, shared by build identity.** A hit is one lookup by interface and call spec; a miss selects, then builds or reuses. Tuning acts on the entry. Why: a hit costs one lookup.

**Reject before dependent work.** When the required call facts are already known, a multi-stage body resolves support for later stages before launching their preprocessing. `GQABwdOp` acquires its backward entry before running preprocess. `select_implementation(interface, call)` can check support without building an entry; it belongs to `Op` and uses the same selection rules as `kernel_for`.

**Adding a kernel takes two hooks.** Register the implementation, then state `applies`, adding `preferred_over` only where it overlaps another non-general implementation. Undeclared, an implementation is available on the CUDA devices of every architecture, applies to every call and has no precedence. Why: a single-implementation op only inherits its interface.

Choosing the interface sits above selection, dtype specialization beside it. See [S13](op-slot-rules.md#slot-s13).

### Target boundary

**A target replaces the whole op.** A target that registers a builder for an op serves every call of it, and its kernel is called with the tensors its builder was described with. The op's own body is the in-tree implementation and does not run for a target.

**`kernel_map=` replaces what runs under a key, not the key's rule.** The key keeps its registered implementation's applicability and precedence, and is available wherever either class runs; the replacement, like every implementation, inherits the key's interface and is built through its own `entry_for`; a selected key whose replacement cannot serve the call is an error, never a fallback. Why: what selects a key stays declared in one place, and changing which calls a kernel serves is registration.

**The op layer guarantees a target the manifest, and nothing more.** The generated checks run before the target is called. Every tensor is on the call device except those declaring `device: cpu`, every tensor declaring `contiguous: true` is contiguous, and the call, a caller-supplied output buffer included, meets the signature.

**A traced op is one graph node whichever target serves it.** An op on the [compile boundary](#compile-dispatch-boundary) chooses between the in-tree kernels and a target inside its operator.

**A composite needs no builder of its own.** An op that builds no kernel of its own runs its composition when the target registers no builder for it; each sub-op, given the composite's `target`, settles on a target itself.

**An op with no call-time tensor input is placed by the call-device rule** of [manifest.md § Call Semantics](manifest.md#call-semantics).

### Kernel caching and enumeration

L1 owns get-or-build. An op names the **kernel interface** a kernel serves, and the selected implementation's `entry_for` names the **identity** of the specialization and the factory that builds it. The factory runs on the first miss for that identity and never again. An op MUST NOT carry a get-or-build of its own — no cache dict, no build guarded on a kernel attribute being unset. Holding what L1 returned in `self.kernel` is not one.

The identity is opaque to L1 and must carry every input that can change what gets built. The selected implementation names those axes in its own `entry_for`, because only it knows what its constructor reads.

Call-spec identity and entry identity have different scopes. RMSNorm calls with different row counts can share an entry specialized by normalized width, epsilon and dtype. Further JIT specializations inside that entry remain kernel-owned and execute outside tracing. `Op` enumerates entries, not every compiled function hidden inside them.

The entry, not the kernel, is the unit built once. A specialization that must build several kernels together returns them as one immutable entry from one factory; kernels keyed independently of each other are separate interfaces.

Auxiliary tensors remain implementation-owned: RoPE and GQA cache frequency tables, and MeanPooling caches placeholder tensors. These caches are separate from the base kernel-entry caches and are not cleared by `_unsettle()`.

`iter_kernels()` enumerates entries and delegates explicitly, never by reflecting over attributes. Reflection could only guess: a kernel nested deeper than the traversal went, or held in an attribute of an unrecognised type, was silently invisible. Declaring turns that silent omission into a missing declaration.

**Sub-ops follow the rule for kernels.** A sub-op's constructor arguments may come from the call, so an instance built at construction cannot show what a composite holds.

- A composite declares the sub-op classes it may hold in `delegate_types`: its composition is a fact of the class, checkable before any call.
- It holds every sub-op through `delegate_for`, once per identity, whether built at construction, built per call or injected. Created children inherit the composite's execution policy; injected instances are held as supplied. Stage names match manifest composition.
- `kernel_delegates()` is derived from what `delegate_for` holds, so enumeration is complete by construction. The base tuning walk includes injected children; it makes no owned/borrowed distinction.

`delegate_for` is eager, like `kernel_for`: a sub-op that depends on the call is built in `_eager_forward`, never on a traced path.

`built_kernels(interface)` is the backend-neutral view: one entry per identity, whoever built it. `iter_kernels()`, and through it `autotune()` and `run_config()`, act on the TileOPs `Kernel` instances the entries hold. A target's builder is not passed `tune`, so a tuning request that cannot reach it warns instead of being dropped.

`GemmW4A16FwdOp` overrides `autotune()` to warn and refuse generic tuning. Its `repack()` helper validates packed weights and calls the built-in `w4a16_repack` interface directly; it does not enter whole-op target dispatch or produce a completed-call record.

## Scaffolding an Op from a Manifest Entry

The scaffold emits a T2 (L1-direct) op file from one manifest entry. The call checks, `_infer_output_shapes`, `_validate_dtypes` and `eval_roofline` are generated from the entry and are not scaffolded. Each step has typed **Input** (manifest fields consumed), **Output** (the code fragment produced), **Validation** (concrete check), and a **Reference** link to the authoritative slot rule in [`op-slot-rules.md`](op-slot-rules.md). Examples scaffold the fictional `ExampleCumsumFwdOp` (cumulative-sum semantics) in T2 (L1-direct) form from an equally fictional manifest entry; nothing in them mirrors a shipped file.

### Step 1: File header + imports

**Input.** The Kernel classes the op dispatches to, and the family's kernel interfaces. The kernel map is owned by the code, not the manifest.

**Output.**

```python
"""Cumulative sum operator (host-side Op layer).

Provides:
  - ExampleCumsumFwdOp: y = cumsum(x, dim=-1)
"""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.reduction.call_spec import (
    ExampleCumsumCall,
    ExampleCumsumFwdInterface,
)
from tileops.kernels.reduction.example_cumsum import ExampleCumsumKernel
from tileops.manifest.primitives import normalize_axis
from tileops.ops.op_base import Op
```

**Validation.** Every concrete-Kernel import matches one `kernel_types` value verbatim, and every kernel interface one `interfaces` value. The `Kernel` and `KernelInterface` base imports and the `tileops.ops.op_base` import are fixed.

**Reference.** [Slot S1](op-slot-rules.md#slot-s1), [S2](op-slot-rules.md#slot-s2), [S3](op-slot-rules.md#slot-s3), [S4](op-slot-rules.md#slot-s4).

### Step 2: Class declaration + docstring + `__all__`

**Input.** Manifest entry key (= class name); what the op computes.

**Output.**

```python
__all__ = ["ExampleCumsumFwdOp"]


class ExampleCumsumFwdOp(Op):
    """Cumulative sum operator: y = cumsum(x, dim=-1).

    Output has the same shape and dtype as input.
    """
```

**Validation.** Class name ≡ manifest entry key, byte-exact (`ExampleCumsumFwdOp`). The class docstring has no `Args:` block: construction parameters are documented on `__init__` (Step 3).

**Reference.** [Slot S5](op-slot-rules.md#slot-s5), [S6](op-slot-rules.md#slot-s6), [S7](op-slot-rules.md#slot-s7).

### Step 3: `__init__` signature and body

**Input.** `signature.params`, and the execution-policy parameters of [manifest.md table 7](manifest.md#t-policy) the op takes.

**Output.**

```python
def __init__(
    self,
    dim: int = -1,
    *,
    target: Target = None,
    kernel_map: Optional[Dict[str, Kernel]] = None,
    tune: bool = False,
):
    """Build the op.

    Args:
        dim: Reduction dimension (default -1).
        target: Backend target to serve this op, or None to decide from the input device.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """
    self.dim = dim
    self.target = target
    self.tune = tune
    self.dispatch_kernel(kernel_map)
```

**Validation.** Every `__init__` kwarg has an `Args:` entry in its docstring; no extras. `__init__` matches `signature.params` item by item ([manifest.md](manifest.md#parameters)), followed by the table-7 execution-policy parameters it takes. `dtype` is not a kwarg — it is read from the input in `forward()`. A param declaring `kw_only: true` goes after `*`.

**Reference.** [Slot S12](op-slot-rules.md#slot-s12), [S13](op-slot-rules.md#slot-s13).

### Step 4: `kernel_types` + `interfaces` + `forward`

**Input.** `signature.inputs`; the kernels and kernel interfaces of Step 1.

**Optional inputs.** An `optional: true` input takes a `None` default in `forward`, and presence is read from the call rather than settled at construction, so one instance serves both ways of calling the op. Where the presence changes what gets built, it belongs in the kernel cache key alongside the shapes.

**Output.**

```python
class ExampleCumsumFwdOp(Op):
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "example_cumsum_fwd": ExampleCumsumKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "example_cumsum_fwd": ExampleCumsumFwdInterface
    }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The generated signature checks have run: dtype, shape, dim range.
        dim = normalize_axis(self.dim, x.ndim)
        x = x.contiguous()  # handed over as the manifest declares it
        call = ExampleCumsumCall(
            device=x.device, shape=tuple(x.shape), dim=dim, dtype=x.dtype
        )
        return self.kernel_for("example_cumsum_fwd", call)(x)
```

**Validation.**

- `forward` repeats no check the signature states. It checks no device kind: a kernel states which devices it runs on.
- The kernel comes from `self.kernel_for`, never a cache dict the op owns:
  - The call spec carries `x.dtype`, so a call with another dtype resolves a second entry rather than reusing the first. The implementation's own `entry_for` names what it is built from; the op defines none.
- The op never trims kernel output, and never reshapes its input for the kernel: a kernel that pads or permutes internally takes and returns the shapes the manifest declares.

**Reference.** [Slot S14](op-slot-rules.md#slot-s14), [S15](op-slot-rules.md#slot-s15), [S16](op-slot-rules.md#slot-s16).

### Step 5: Generated methods

**Input.** The whole entry.

**Output.** Nothing to write. `_infer_output_shapes`, `_validate_dtypes`, the call checks around `forward` and `eval_roofline` are generated from the signature and the `roofline` field; see [manifest.md § Call Semantics](manifest.md#call-semantics) and [roofline.md §4.4](roofline.md#44-op-codegen).

**Validation.** `python scripts/validate_manifest.py`.

**Reference.** [Slot S17](op-slot-rules.md#slot-s17), [S18](op-slot-rules.md#slot-s18), [S19](op-slot-rules.md#slot-s19).

**Compute roof.** `Op.compute_roof()` names the GPU-profile unit that prices the FLOPs `eval_roofline()` counts; the base default `"cuda_core.fp32"` covers CUDA-core fp32 arithmetic. An op whose FLOPs are matmul contractions overrides it — normally `tensor_core_roof` of the contraction's input dtype read from `self.last_call`, branching on instance state (a backend switch) where the contraction dtype differs from the input dtype. Contract and rationale: [`roofline.md §1.4`](roofline.md#14-compute-roof).

### Step 6: Package registration

**Input.** The class name (Step 2) and the op's source filename.

**Output.** Two files, both with a matching `__all__` entry.

Implementation package, `src/tileops/ops/reduction/__init__.py`:

```python
# --- ExampleCumsumKernel ops ---
from tileops.ops.reduction.example_cumsum import ExampleCumsumFwdOp
```

Public path, `src/tileops/reduction.py` — this is the one callers import from:

```python
from tileops.ops.reduction import ExampleCumsumFwdOp
```

**Validation.** The implementation import sits under its family's grouping comment block, and both files carry a matching `__all__` entry — miss the second and the op is unreachable from `tileops.reduction`.

**Reference.** [Slot S20](op-slot-rules.md#slot-s20).

### Slot coverage

| Step | Slots produced |
| ---- | -------------- |
| 1    | S1, S2, S3, S4 |
| 2    | S5, S6, S7     |
| 3    | S12, S13       |
| 4    | S14, S15, S16  |
| 5    | S17, S18, S19  |
| 6    | S20            |

## Out of Scope

This playbook emits exactly the 16 slots above. The following are **not** produced by the scaffold — each needs separate treatment:

- **Family-specific protocol variables and hooks.** `_op_kind`, and the hooks of [Optional Hooks (Appendix)](ops-design-reference.md#optional-hooks-appendix) (reduction). Kernel-dispatch-convention-dependent; cannot be mechanically derived from the manifest. See [Family-Base Protocol (Appendix)](ops-design-reference.md#base-class-protocol).
- **Family-base (T1) subclassing.** See [Family-Base Refactoring](#family-base-refactoring).
- **Kernel implementations themselves.** The playbook's scope is the Op (host) layer. See [Implementing a Kernel](#implementing-a-kernel) for the kernel-side interface surface.
- **`fullgraph` compile registration.** Declaring a compile boundary is the class's claim that it supports `fullgraph=True`; its cold compile test, registered in `tests/compile_contract.py`, is the evidence, and the registered set equals the implemented classes declaring a boundary.
- **Compile dispatch boundary.** See [Compile Dispatch Boundary](#compile-dispatch-boundary).

## Implementing a Kernel

Kernel implementation is not covered by this playbook. The device-side interface a scaffolded Op depends on — the kernel interface's `forward` and the classmethod `entry_for`, required `kernel`, optional `default_config` / `autotune_configs` / `supported_archs` — is specified in [Kernel base class attributes](ops-design-reference.md#base-class-protocol).

## Compile Dispatch Boundary

Contract for every op registered for `fullgraph=True` compilation while resolving kernels at call time.

**Invariant.** A dynamo-traced `forward` MUST NOT construct a `Kernel` or enter a TileLang builder. Kernel-cache misses run TileLang JIT machinery that dynamo cannot trace; an eager warm-up before `torch.compile` only hides the miss path and does not satisfy the cold-call contract.

**Decisions.**

- A class declaring `compile_boundary = True` claims `fullgraph=True` support. The manifest records nothing; the registered compile tests are the evidence, and their set equals the implemented classes declaring a boundary.
- The operators are generated from the manifest entry, one `torch.library.custom_op` per effect branch ([manifest.md § Effects](manifest.md#effects)), so no op writes registration code and a schema cannot drift from its entry. The operator is what makes the graph node this op's, and it stays the same node when a target serves the op.
- `forward` only chooses which operator to call. The operator's eager body runs the generated checks once, then the in-tree kernels (`_eager_forward`) or the target; its fake comes from the signature.
- An op's operators write exactly the inputs the manifest marks `mutated`; the validator holds them equal.
- The operator's name is derived from the family and the class; an op does not choose it.
- The boundary covers forward-only compilation. No operator carries an autograd formula, so a backward op's operator refuses an input that tracks history; a caller that needs to backpropagate wires its own `torch.autograd.Function` around the forward and backward ops.
- An op with no tensor input has no node to own and registers no boundary. An op that builds no kernel in `forward` does not need the boundary; the invariant still applies to it.

## Family-Base Refactoring

The scaffold emits T2 (L1-direct) ops only; once a family accumulates 2-3 ops sharing an identical `forward()` flow, a separate family-specific refactoring, outside this playbook, extracts an L2 base and rewrites the concrete ops as T1 thin wrappers — see [Development Path](ops-design-reference.md#development-path) for when to extract and [Adding a New Family Base](ops-design-reference.md#adding-a-new-family-base) for the process. Family bases MUST NOT normalize genuine per-op behavior differences.

## Further Reference

- [Slot Rules](op-slot-rules.md) — full Rule / Derivation / Example / Common mistakes per slot
- [Codegen Details](ops-design-reference.md#codegen) — calling conventions, consistency enforcement
- [Base Class Protocol](ops-design-reference.md#base-class-protocol) — `Op` and `Kernel` base class attributes
- [Naming Conventions](ops-design-reference.md#naming-conventions) — class / `kernel_map` / builder function rules
- [Parameter Design](ops-design-reference.md#parameter-design) — construction time versus call time
- [manifest.md](manifest.md) — manifest entry structure, signature, workloads, call semantics
- [roofline.md](roofline.md) — roofline formula syntax, codegen, evaluator surface boundary
