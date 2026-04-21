# Baseline Reference Report v1
**Strategy:** S1 — Held-out Validation / Evaluation Foundation
**Evaluation set version:** v1
**Status:** STRUCTURE_FROZEN — awaiting model inference run (TICKET_S1_5)
**Created:** 2026-04-20
**Owner:** Generator

---

## 1. Purpose

This report defines the v1 evaluation baseline: the zero-point against which all future
candidate runs are measured.  Actual numeric accuracy is populated by TICKET_S1_5
(baseline measured reference) after model inference is executed.  The structural baseline
(set membership, category distribution, failure mode labeling) is frozen as of this report.

---

## 2. Frozen Evaluation Assets

| Asset | Path | Samples | Status |
|---|---|---:|---|
| Quick Gate v1 | `data/eval/quick_gate_v1.jsonl` | 75 | FROZEN |
| Diagnostic v1 | `data/eval/diagnostic_v1.jsonl` | 150 | FROZEN |
| Promotion v1 | `data/eval/promotion_v1.jsonl` | 400 | FROZEN |
| Category manifest v1 | `data/eval/category_manifest_v1.csv` | 550 unique | FROZEN |

Manifest row count (550) = 75 QG-only rows + 75 DG-unique rows + 400 PR rows.
QG samples appear as both `quick_gate_v1` and `diagnostic_v1` rows in the manifest
(same `sample_id`, different `split_name`), making the deduplicated problem count 550.

**Freeze rule:** Set membership must not change until Planner issues a version bump.
Any leakage discovery, coverage gap, or repeated false promotion must be escalated as
a manifest revision ticket, not silently patched.

---

## 3. Category Distribution

### 3.1 Quick Gate v1 (75 samples)

| Category | Count | % | Difficulty | Conv-Sensitive |
|---|---:|---:|---|---:|
| numeral | 10 | 13.3 | easy | 0 |
| unit_conversion | 8 | 10.7 | easy | 0 |
| gravity | 5 | 6.7 | easy | 0 |
| cipher | 3 | 4.0 | easy | 0 |
| **easy subtotal** | **26** | **34.7** | — | 0 |
| equation | 15 | 20.0 | medium | 0 |
| bit_manipulation | 15 | 20.0 | medium | 0 |
| conversion_sensitive | 8 | 10.7 | medium | 8 |
| low_logprob_suspect | 6 | 8.0 | hard | 6 |
| hard | 5 | 6.7 | hard | 0 |
| **hard/unstable subtotal** | **19** | **25.3** | — | 14 |
| **TOTAL** | **75** | **100** | — | **14** |

Design target met: easy 35% ✓, equation 20% ✓, bit 20% ✓, hard/unstable 25% ✓

### 3.2 Diagnostic v1 (150 samples)

| Category | Count | % | Conv-Sensitive |
|---|---:|---:|---:|
| numeral | 15 | 10.0 | 0 |
| unit_conversion | 12 | 8.0 | 0 |
| gravity | 8 | 5.3 | 0 |
| cipher | 5 | 3.3 | 0 |
| **easy subtotal** | **40** | **26.7** | 0 |
| equation | 30 | 20.0 | 0 |
| bit_manipulation | 30 | 20.0 | 0 |
| conversion_sensitive | 20 | 13.3 | 20 |
| low_logprob_suspect | 15 | 10.0 | 15 |
| hard | 15 | 10.0 | 0 |
| **TOTAL** | **150** | **100** | **35** |

### 3.3 Promotion v1 (400 samples)

| Category | Count | % | Conv-Sensitive |
|---|---:|---:|---:|
| numeral | 40 | 10.0 | 0 |
| unit_conversion | 30 | 7.5 | 0 |
| gravity | 25 | 6.2 | 0 |
| cipher | 25 | 6.2 | 0 |
| **easy subtotal** | **120** | **30.0** | 0 |
| equation | 80 | 20.0 | 0 |
| bit_manipulation | 80 | 20.0 | 0 |
| conversion_sensitive | 40 | 10.0 | 40 |
| low_logprob_suspect | 40 | 10.0 | 40 |
| hard | 40 | 10.0 | 0 |
| **TOTAL** | **400** | **100** | **80** |

---

## 4. Baseline Accuracy — PENDING MODEL RUN

The table below is the required reporting template.  All values are populated by
TICKET_S1_5.  Until that ticket is completed, this report serves as the structural
baseline only.

### 4.1 Quick Gate baseline results (to be filled by S1.5)

| Category | Sample Count | Baseline Accuracy | Format Failure Rate | Extraction Failure Rate | Notes |
|---|---:|---:|---:|---:|---|
| numeral | 10 | PENDING | PENDING | PENDING | |
| unit_conversion | 8 | PENDING | PENDING | PENDING | |
| gravity | 5 | PENDING | PENDING | PENDING | |
| cipher | 3 | PENDING | PENDING | PENDING | |
| **easy total** | **26** | **PENDING** | **PENDING** | **PENDING** | |
| equation | 15 | PENDING | PENDING | PENDING | |
| bit_manipulation | 15 | PENDING | PENDING | PENDING | |
| conversion_sensitive | 8 | PENDING | PENDING | PENDING | pre-conv vs post-conv to be added |
| low_logprob_suspect | 6 | PENDING | PENDING | PENDING | |
| hard | 5 | PENDING | PENDING | PENDING | |
| **TOTAL** | **75** | **PENDING** | **PENDING** | **PENDING** | |

### 4.2 Diagnostic baseline results (to be filled by S1.5)

| Category | Baseline Accuracy | Delta-from-QG | Dominant Failure Mode | Conv Pre/Post Delta |
|---|---:|---:|---|---:|
| numeral | PENDING | PENDING | FORMAT_FAILURE | N/A |
| unit_conversion | PENDING | PENDING | FORMAT_FAILURE | N/A |
| gravity | PENDING | PENDING | FORMAT_FAILURE | N/A |
| cipher | PENDING | PENDING | EXTRACTION_FAILURE | N/A |
| equation | PENDING | PENDING | OPERATOR_CONFUSION | N/A |
| bit_manipulation | PENDING | PENDING | BIT_RULE_FAILURE | N/A |
| conversion_sensitive | PENDING | PENDING | PREPOST_CONVERSION_REGRESSION | PENDING |
| low_logprob_suspect | PENDING | PENDING | LOW_LOGPROB_COLLAPSE | PENDING |
| hard | PENDING | PENDING | UNKNOWN | N/A |

### 4.3 Promotion baseline results (to be filled by S1.5)

| Category | Baseline Accuracy | Notes |
|---|---:|---|
| easy (all) | PENDING | |
| equation | PENDING | |
| bit_manipulation | PENDING | |
| conversion_sensitive | PENDING | |
| low_logprob_suspect | PENDING | |
| hard | PENDING | |
| **TOTAL** | **PENDING** | |

---

## 5. Failure Mode Taxonomy (frozen with v1)

Primary labels in use for v1:

| Label | Categories Where Expected | Regression Tolerance |
|---|---|---|
| FORMAT_FAILURE | numeral, unit_conversion, gravity, cipher | near-zero |
| EXTRACTION_FAILURE | cipher, equation | near-zero |
| EASY_TASK_REGRESSION | unit_conversion, gravity | zero |
| OPERATOR_CONFUSION | equation, hard | low |
| BIT_RULE_FAILURE | bit_manipulation, conversion_sensitive | low |
| POSITIONAL_MISMATCH | bit_manipulation | low |
| LOW_LOGPROB_COLLAPSE | low_logprob_suspect | medium |
| PREPOST_CONVERSION_REGRESSION | conversion_sensitive | medium |
| OVERFITTING_TO_DIAGNOSTIC | any | zero (guard) |
| UNKNOWN | hard | acceptable |

---

## 6. Quick Gate Pass/Fail Thresholds (provisional — Planner to confirm)

These are the default thresholds from `docs/specs/a1_evaluation_set_design.md`.
Planner may tighten or relax per experiment.

| Signal | Threshold | Basis |
|---|---|---|
| Easy-category accuracy delta | ≤ −2 pp → FAIL | Near-zero regression tolerance |
| Format failure rate delta | ≤ +2 pp → FAIL | Near-zero tolerance |
| Total accuracy delta | ≤ −3 pp → FAIL | Must not be materially worse |
| Catastrophic category collapse | Any category to 0% → FAIL | Absolute floor |
| Target-category delta (experiment-specific) | ≤ 0 pp when it is the stated target → FAIL | No null improvement |

---

## 7. Dataset Revision Conditions

This evaluation set must not change except when:
- A leakage instance is confirmed (sample in training data).
- Category coverage is shown insufficient by repeated false promotion.
- Planner issues a v2 version-bump with written rationale.

When changed: update this file, category manifest, and all prior experiment logs that
reference v1.

---

## 8. Next Required Step

**TICKET_S1_5** — baseline_measured_reference_v1
- Run model inference on all three frozen sets.
- Populate all PENDING cells in Section 4 above.
- Verify repeatability by running Quick Gate at least twice on same weights.
- Confirm that Quick Gate judgment is stable across identical runs.
- Output: `reports/eval/baseline_reference_v1_measured.md`

---

## 9. Reproducibility Notes

- All sample IDs are stable (`QG_001`–`QG_075`, `DG_001`–`DG_075`, `PR_001`–`PR_400`).
- Generator script: `scripts/gen_eval_sets_v1.py` (seed=42).
- Problem text is deterministic: same script run produces identical output.
- `expected_answer` fields are PENDING; answer keys must be generated by running
  inference on a reference model or by computing answers analytically per category.
- For Quick Gate determinism check: run identical weights twice, compare per-sample
  correctness vectors.  Any disagreement on easy samples without temperature=0 must
  be investigated before the baseline is accepted.
