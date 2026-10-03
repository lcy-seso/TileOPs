## Manifest status

![ops](https://img.shields.io/badge/ops-200-blue) ![implemented](https://img.shields.io/badge/implemented-171%20%2F%20200%20%2886%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-29-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 11 | 9 | 20 | `██████░░░░` 55% | 97 |
| `convolution` | 3 | 0 | 3 | `██████████` 100% | 43 |
| `elementwise` | 70 | 0 | 70 | `██████████` 100% | 229 |
| `fft` | 1 | 0 | 1 | `██████████` 100% | 10 |
| `gemm` | 6 | 0 | 6 | `██████████` 100% | 77 |
| `linear_attention` | 7 | 4 | 11 | `██████░░░░` 64% | 54 |
| `mamba` | 7 | 0 | 7 | `██████████` 100% | 29 |
| `moe` | 10 | 0 | 10 | `██████████` 100% | 75 |
| `norm` | 10 | 1 | 11 | `█████████░` 91% | 64 |
| `pool` | 13 | 0 | 13 | `██████████` 100% | 44 |
| `quantization` | 1 | 9 | 10 | `█░░░░░░░░░` 10% | 31 |
| `reduction` | 21 | 0 | 21 | `██████████` 100% | 94 |
| `rope` | 6 | 0 | 6 | `██████████` 100% | 19 |
| `sampling` | 0 | 6 | 6 | `░░░░░░░░░░` 0% | 28 |
| `sequence_modeling` | 5 | 0 | 5 | `██████████` 100% | 17 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 110 / 200 (55%) |
| `roofline` (func or flops) | 200 / 200 (100%) |

**Workloads:** 911 total — 4.60 per implemented op.

### Conformance gaps

- Implemented ops without `roofline`: **0**
- Implemented ops without `workloads`: **0**
- Implemented ops with fewer than two workloads: **0**

<details><summary>Spec-only ops (29)</summary>

| | | |
| --- | --- | --- |
| `ChainSpeculativeSamplingFwdOp` | `DeepSeekSparseAttentionPagedFwdOp` | `DeltaNetInferenceFwdOp` |
| `FP8QuantPerBlockFwdOp` | `FusedQKNormRopeFwdOp` | `GLAInferenceFwdOp` |
| `GatedDeltaNetFwdOp` | `GroupedQueryAttentionPagedFwdOp` | `GroupedQueryAttentionVarlenFwdOp` |
| `INT4QuantPerGroupFwdOp` | `INT8DequantPerBlockFwdOp` | `INT8DequantPerChannelFwdOp` |
| `INT8DequantPerTensorFwdOp` | `INT8QuantPerBlockFwdOp` | `INT8QuantPerChannelFwdOp` |
| `INT8QuantPerTensorFwdOp` | `KimiDeltaAttentionFwdOp` | `MergeAttentionStatesFwdOp` |
| `MinPMaskFwdOp` | `MultiHeadLatentAttentionKVCacheWriteFwdOp` | `MultiHeadLatentAttentionPagedFwdOp` |
| `MultiHeadLatentAttentionVarlenFwdOp` | `PagedKVCacheGatherFwdOp` | `PagedKVCacheWriteFwdOp` |
| `SamplingFromProbsFwdOp` | `SmoothQuantFwdOp` | `TopKMaskFwdOp` |
| `TopKTopPMaskFwdOp` | `TopPMaskFwdOp` |  |

</details>
