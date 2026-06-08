# B3_GATE15_X17_ASYMMETRIC_INPROJ_SPLIT LB Result Report

## Summary

- **Experiment:** `B3_GATE15_X17_ASYMMETRIC_INPROJ_SPLIT`
- **Candidate change:** asymmetric in-projection split with `gate=15 / x=17`
- **Public LB:** `0.85` weak / below Golden Baseline
- **Golden Baseline:** `0.86`
- **Decision:** `REJECT`
- **Rollback target:** Golden Baseline with `gate=16 / x=16`

## Prompt5 Structural Diagnostics Observed Before Submission

The candidate passed the visible structural checks available during generation and packaging:

- `num_tensors=12010`
- `max_rank_seen=32`
- Rank violations: none observed
- NaN/Inf: none observed
- `in_proj_module_count=23`

## Interpretation

The Public LB regression indicates that local adapter structure health checks are necessary but not sufficient for leaderboard performance. In particular, the result shows that reallocating rank toward apparently high-error SVD areas does not guarantee an LB improvement.

## Decision

`B3_GATE15_X17_ASYMMETRIC_INPROJ_SPLIT` is rejected. Do not continue nearby asymmetric gate/x split variants (`14/18`, `17/15`, `18/14`) until Golden exact reproducibility has been re-established and a new experiment strategy is approved.

## Next Step

Return to Golden Baseline control and prepare `B3_GOLDEN_EXACT_REGEN` as the next experiment. No adapter regeneration, submission zip regeneration, Kaggle submission, new rank split experiment, or SFT work is part of this record-only phase.
