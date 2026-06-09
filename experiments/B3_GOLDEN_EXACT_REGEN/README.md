# B3_GOLDEN_EXACT_REGEN

## Objective

Regenerate a submission-equivalent Golden Baseline candidate with the current safe Prompt1-Prompt6 workflow and verify that the known Public LB reference can still be reproduced.

This is a **control experiment**, not an improvement experiment.

## Baseline reference

- **Reference name:** B3 Golden Baseline
- **Expected Public LB:** `0.86` equivalent
- **Gate/X split:** `gate=16 / x=16`
- **Preceding rejected candidate:** `B3_GATE15_X17_ASYMMETRIC_INPROJ_SPLIT`
- **Reason for control:** `gate=15 / x=17` regressed to `0.85` weak despite passing visible structural diagnostics.

## Variables fixed to Golden

Keep all of the following unchanged from the Golden Baseline:

- Training data
- Adapter rank map
- Target modules
- Conversion script
- Gate/X split (`16/16`)
- Generation configuration
- Packaging and root `submission.zip` creation process

## Explicit non-goals

Do **not** use this experiment to test any of the following:

- New SFT data
- Cryptarithm carry, leading-zero, or mapping-conflict patches
- Gate/X asymmetric split variants
- Rank map micro-adjustments
- In-projection redistribution
- Up/down rank micro-adjustments
- fp64-only SVD trials
- SVD-error-based rank reallocation

## Required evidence before any submission

Record these values before promoting the regenerated control artifact:

| Evidence | Required value / status |
|---|---|
| Gate/X split | `16/16` |
| Rank violations | none |
| NaN/Inf | none |
| `max_rank_seen` | expected Golden-compatible value |
| Tensor count | expected Golden-compatible value |
| Adapter structure | matches Golden target modules and rank map |
| Root archive | `submission.zip` only when explicitly generated in the execution phase |

## Adoption condition

Adopt this as the current control only if the Public LB reproduces the Golden Baseline at `0.86` equivalent.

## Failure interpretation

If this control fails to reproduce the Golden LB, do not start SFT. Treat the failure as evidence of drift in at least one of:

- Environment
- Adapter generation flow
- Conversion flow
- Packaging flow
- Root `submission.zip` contents
- Submission procedure

## Next approved direction after successful control

After Golden exact reproducibility is confirmed, the next improvement candidate should be:

```text
Experiment: S1_CRYPTARITHM_CARRY_PLUS100
Change: add only 100 carry-focused cryptarithm training examples
Fixed: Golden adapter structure, target modules, rank map, conversion, gate/x=16/16, generation, packaging
```
