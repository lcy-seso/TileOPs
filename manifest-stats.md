## Manifest status

![ops](https://img.shields.io/badge/ops-180-blue) ![implemented](https://img.shields.io/badge/implemented-175%20%2F%20180%20%2897%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-5-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 15 | 2 | 17 | `█████████░` 88% | 76 |
| `convolution` | 3 | 0 | 3 | `██████████` 100% | 43 |
| `elementwise` | 70 | 0 | 70 | `██████████` 100% | 148 |
| `gemm` | 6 | 0 | 6 | `██████████` 100% | 62 |
| `linear_attention` | 7 | 3 | 10 | `███████░░░` 70% | 47 |
| `mamba` | 7 | 0 | 7 | `██████████` 100% | 29 |
| `moe` | 10 | 0 | 10 | `██████████` 100% | 69 |
| `normalization` | 10 | 0 | 10 | `██████████` 100% | 50 |
| `pool` | 13 | 0 | 13 | `██████████` 100% | 41 |
| `position_encoding` | 6 | 0 | 6 | `██████████` 100% | 13 |
| `quantization` | 1 | 0 | 1 | `██████████` 100% | 3 |
| `reduction` | 19 | 0 | 19 | `██████████` 100% | 62 |
| `scan` | 2 | 0 | 2 | `██████████` 100% | 4 |
| `sequence_modeling` | 5 | 0 | 5 | `██████████` 100% | 15 |
| `spectral` | 1 | 0 | 1 | `██████████` 100% | 9 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 180 / 180 (100%) |
| `roofline` (func or flops+bytes) | 180 / 180 (100%) |
| `source.kernel_map` | 179 / 180 (99%) |
| `source.bench_manifest_driven` | 178 / 180 (99%) |

**Workloads:** 671 total — 3.72 per implemented op.

### Conformance gaps

- Implemented ops without `kernel_map`: **0**
- Implemented ops without `roofline`: **0**
- Implemented ops without `source.bench_manifest_driven`: **0**
- Implemented ops with fewer than two workloads: **0**

<details><summary>Spec-only ops (5)</summary>

| | | |
| --- | --- | --- |
| `DeltaNetInferenceFwdOp` | `GLAInferenceFwdOp` | `GatedDeltaNetFwdOp` |
| `GroupedQueryAttentionPagedFwdOp` | `GroupedQueryAttentionVarlenFwdOp` |  |

</details>
