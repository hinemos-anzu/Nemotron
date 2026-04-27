# B0 Reproduction Run — Result Report

## Status
HOLD — awaiting Kaggle execution
Date prepared: 2026-04-27

---

## 1. Objective

Confirm that the frozen Recovered Baseline 0.85 submission path reproduces LB 0.85
with zero code changes.

This is a reproduction run, not an improvement experiment.
No model, adapter, surgery, or RANK_MAP changes are permitted.

---

## 2. Single Main Variable

None — no variables changed.
All settings identical to commit fbb6621 (Recovered Baseline 0.85 freeze).

---

## 3. Frozen Baseline Reference

| Field | Value |
|---|---|
| branch | claude/freeze-baseline-prep-P49Z3 |
| commit | fbb6621 |
| baseline LB | 0.85 |
| decision | ADOPT_AS_RECOVERED_BASELINE |
| submission path | raw adapter → Offline Asymmetric SVD Surgery → /kaggle/working/adapter → submission.zip |

---

## 4. Submission Generation

| Field | Value |
|---|---|
| adapter source | — (to be filled after Kaggle run) |
| surgery path | Offline Asymmetric SVD Surgery (identical to fbb6621) |
| submission.zip | /kaggle/working/submission.zip |
| size | — (to be filled after Kaggle run) |

---

## 5. Validation Gate

See `reports/b0_submission_asset_check.md` for detailed per-tensor validation.

| Check | Expected | Observed |
|---|---|---|
| base_model.model.model absent | YES | — |
| base_model.model.backbone present | YES | — |
| experts.w1/w2/w3 absent | YES | — |
| gate_proj/x_proj absent | YES | — |
| in_proj count | 46 | — |
| zero-shape tensors | 0 | — |
| tensor count | 12010 | — |

---

## 6. Kaggle Result

| Field | Value |
|---|---|
| metric error | — (to be filled after Kaggle run) |
| LB score | — (to be filled after Kaggle run) |
| reproduced 0.85 | — |

---

## 7. Decision

| Field | Value |
|---|---|
| decision | HOLD |
| reason | Kaggle execution not yet run. Fill fields in sections 4–6 after submission. |

---

## 8. Pass / Hold / Fail Criteria

### PASS
```
LB score = 0.85
metric error = NO
validation gate = all PASS
```

### HOLD
```
Kaggle execution not yet run
submission asset not yet confirmed
```

### FAIL
```
metric error present
LB < 0.85
any validation check FAIL
raw adapter detected in zip
surgery diff vs fbb6621 detected
```

---

## 9. Next Action

```
if PASS:  proceed to B2 — up_proj: 16 → 24
if HOLD:  complete Kaggle execution and update this report
if FAIL:  rollback — revert to fbb6621 baseline path, diagnose diff
```

---

## 10. Change Log

| Date | Event |
|---|---|
| 2026-04-27 | B0 report created, status HOLD, awaiting Kaggle execution |
