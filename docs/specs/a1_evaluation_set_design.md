# A1 Evaluation Set Design Spec v1.0

## Purpose
This document defines the shared evaluation OS for the Planner, Generator, and Kaggle execution role.
The primary objective is to decide faster whether a candidate improvement is effective, ineffective, or inconclusive, and to transfer the learning to the next experiment.

## Shared Principles
- One experiment, one main variable.
- The same evaluation sets must be reused across experiments until this spec is revised.
- Every run must produce before/after comparisons against the current baseline.
- Public LB is not the first gate. Local/shared evaluation gates decide promotion to Kaggle.
- A failed experiment is still useful only if its failure mode is recorded in the shared log template.

## Evaluation Stack
The evaluation stack is divided into three tiers.

### 1. Quick Gate Set
**Role**
- Earliest Go/No-Go gate.
- Reject obvious regressions before heavier evaluation or Kaggle runs.

**Target size**
- 50 to 100 samples.

**Composition rules**
- Include all high-frequency easy categories.
- Include at least one representative subset for the current main bottleneck categories.
- Include at least 10 known brittle samples from prior failures.
- Include both pre-conversion and post-conversion sensitive samples when adapter conversion is involved.

**Recommended category mix**
- easy / formatting-sensitive: 35%
- equation / symbolic pattern: 20%
- bit manipulation: 20%
- known hard / unstable: 25%

**Success use case**
- Finish within the shortest possible cycle.
- Catch format breakage, easy task regressions, and major category collapses.

**Promotion rule**
A candidate passes Quick Gate only when all of the following hold:
- total accuracy is not materially below baseline
- easy-task accuracy does not regress beyond the allowed threshold
- format failure rate does not worsen
- no catastrophic category collapse is observed

### 2. Diagnostic Set
**Role**
- Explain why a run improved or regressed.
- Support category-level and failure-mode analysis.

**Target size**
- 100 to 200 samples.

**Composition rules**
- Include all recent false negatives and brittle cases from Quick Gate.
- Include low-min-logprob suspects.
- Include conversion-sensitive and alignment-sensitive samples.
- Include a balanced spread across the major task categories.

**Required annotations per sample**
- category
- difficulty bucket
- failure mode label
- conversion sensitivity flag
- low-logprob suspicion flag
- source of inclusion (historical failure / random holdout / synthetic / manual)

**Main output**
- category-wise delta
- failure-mode delta
- brittle sample delta
- pre/post-conversion delta

### 3. Promotion Set
**Role**
- Decide whether a candidate is strong enough to spend Kaggle budget on.

**Target size**
- 300 to 500 samples.

**Composition rules**
- Strict held-out discipline.
- Stable category distribution.
- Include enough samples per priority category to detect meaningful regression.
- Do not overfit this set by repeated manual tuning without spec revision.

**Promotion rule**
A candidate is promoted to Kaggle only when:
- it beats or matches baseline on total accuracy
- it improves the intended target category or bottleneck
- it does not introduce unacceptable easy-task or format regression
- its observed gain survives Diagnostic review

## Allowed Status Labels
Each experiment must end with one status:
- PASS_TO_PROMOTION
- FAIL_QUICK_GATE
- FAIL_DIAGNOSTIC
- PROMOTE_TO_KAGGLE
- HOLD_INCONCLUSIVE
- BLOCKED_ENVIRONMENT

## Threshold Policy
Exact numeric thresholds are owned by Planner and may be revised in a companion threshold file.
Until then, use the following default policy:
- easy-task regression: near-zero tolerance
- format failure regression: near-zero tolerance
- target-category regression: not allowed when that category is the experiment target
- total accuracy: must not be materially worse than baseline

## Dataset Revision Policy
The evaluation sets are shared assets. They can be changed only when:
- leakage is discovered
- category coverage is shown to be insufficient
- repeated false promotion or false rejection is observed
- Planner issues a version bump

When revised, the following must be updated together:
- this design spec
- category manifest
- experiment log template
- baseline reference results

## Directory Convention
Recommended shared paths in the repo:
- `data/eval/quick_gate_v1.*`
- `data/eval/diagnostic_v1.*`
- `data/eval/promotion_v1.*`
- `reports/eval/baseline_reference_v1.md`
- `logs/experiments/`

## Handoff Contract
- Planner owns prioritization, thresholds, and promotion decisions.
- Generator owns data preparation, metric generation, and structured experiment artifacts.
- Kaggle execution role runs only promoted candidates and returns Kaggle-facing evidence back into the shared logs.
