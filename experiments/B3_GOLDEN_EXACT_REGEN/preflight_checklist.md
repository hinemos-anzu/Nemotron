# B3_GOLDEN_EXACT_REGEN Preflight Checklist

This checklist must be completed before any later execution phase regenerates an adapter, creates a root `submission.zip`, or submits to Kaggle.

## Experiment identity

- [ ] Experiment name is exactly `B3_GOLDEN_EXACT_REGEN`.
- [ ] Purpose is control-only Golden Baseline reproduction, not an improvement attempt.
- [ ] Expected Public LB reference is `0.86` equivalent.
- [ ] Rollback target is Golden Baseline `gate=16 / x=16`.
- [ ] The preceding candidate `B3_GATE15_X17_ASYMMETRIC_INPROJ_SPLIT` remains rejected.

## Golden fixed conditions

All boxes must remain checked throughout execution:

- [ ] Training data unchanged.
- [ ] Adapter rank map unchanged.
- [ ] `target_modules` unchanged.
- [ ] Conversion script unchanged.
- [ ] Gate/X split fixed at `16/16`.
- [ ] Generation config unchanged.
- [ ] Root `submission.zip` packaging unchanged.

## Prohibited actions in this preparation phase

- [ ] Do not start SFT.
- [ ] Do not create a cryptarithm patch.
- [ ] Do not perform rank, SVD, or weight-surgery edits.
- [ ] Do not run any gate/x asymmetric split.
- [ ] Do not regenerate an adapter.
- [ ] Do not regenerate `submission.zip`.
- [ ] Do not submit to Kaggle.

## Required files before execution

- [ ] `README.md` describes the control objective and non-goals.
- [ ] `golden_fixed_conditions.md` captures all Golden invariants.
- [ ] `diagnostics_template.md` is ready to fill before promotion/submission.
- [ ] `runbook.md` describes the exact later execution path.

## Go / no-go rule

Only proceed to a later execution phase if every fixed condition above can be satisfied without creating any new experimental variable. If any condition is unknown, stop and resolve the source-of-truth mismatch before generating artifacts.
