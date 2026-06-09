# Rollback to Golden Baseline

## Current Rejected Candidate

- **Experiment:** `B3_GATE15_X17_ASYMMETRIC_INPROJ_SPLIT`
- **Public LB result:** `0.85` weak
- **Golden Baseline Public LB:** `0.86`
- **Decision:** `REJECT`

Although the candidate passed visible Prompt5 structural diagnostics (`num_tensors=12010`, `max_rank_seen=32`, no observed rank violations, no observed NaN/Inf, and `in_proj_module_count=23`), it regressed on Public LB. Treat this as evidence that structural adapter health does not imply leaderboard quality.

## Rollback Target

Return to the Golden Baseline configuration:

- `gate=16`
- `x=16`
- Golden adapter rank map unchanged
- Golden target modules unchanged
- Golden conversion script unchanged
- Golden generation and packaging flow unchanged

## Rank / SVD / Weight Surgery Status

Pause the following experiment families until Golden exact reproducibility is reconfirmed and a new direction is approved:

- `gate/x` asymmetric split variants, including `14/18`, `17/15`, and `18/14`
- Rank map micro-adjustments
- In-projection redistribution
- Up/down rank micro-adjustments
- fp64-only SVD trials
- SVD-error-based rank reallocation

## Golden Exact Regeneration Plan

The next control experiment is:

```text
Experiment: B3_GOLDEN_EXACT_REGEN
Purpose: Reproduce the Golden Baseline 0.86 candidate with the current safe Prompt1-Prompt6 process.
Changes: none
Gate/X: 16/16
Adoption condition: Public LB = 0.86 equivalent
Failure meaning: current environment, zip packaging, or adapter generation flow has drifted from Golden.
```

Use `experiments/B3_GOLDEN_EXACT_REGEN/README.md` as the execution checklist for that control. This record-only phase creates the experiment directory and plan but does **not** regenerate adapters, rebuild `submission.zip`, submit to Kaggle, start SFT, or launch additional rank/split experiments.

## Next Improvement Direction After Control

If `B3_GOLDEN_EXACT_REGEN` confirms the 0.86 baseline, shift improvement work away from weight surgery and toward fixed-structure SFT experiments, starting with `S1_CRYPTARITHM_CARRY_PLUS100`.
