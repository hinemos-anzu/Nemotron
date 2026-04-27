# Recovered Baseline 0.85 — Freeze Record

## Status
FROZEN — 2026-04-27

## Summary
Raw adapter direct zip submission caused `Evaluation metric raised an unexpected error` on Kaggle.
After applying Offline Asymmetric SVD Surgery, the submission produced a clean LB score of **0.85**.
This document records the failure mode, the resolution, and the canonical submission path.

---

## 1. Raw Adapter Submission — FAILURE

### What was attempted
The raw LoRA adapter (as produced by training) was zipped directly into `submission.zip` and submitted to Kaggle.

### Result
```
Evaluation metric raised an unexpected error
```
Kaggle returned this error on every raw adapter submission.

### Root cause
The raw adapter contains key prefixes and tensor shapes that are incompatible with the Kaggle evaluation harness:

| Issue | Detail |
|---|---|
| Key prefix | `base_model.model.model.*` — double `model` segment not accepted |
| MoE expert tensors | `experts.w1`, `experts.w2`, `experts.w3` — fused format, not accepted |
| Gate / x projection | `gate_proj`, `x_proj` — separate keys, not accepted |
| Zero-shape tensors | Present in raw adapter — cause runtime errors |

The Kaggle harness expects adapter keys following the surgery-normalised schema.
Submitting raw keys causes a load failure before any metric is computed, which surfaces as an unexpected metric error rather than a model accuracy error.

---

## 2. Surgery-Processed Adapter Submission — SUCCESS

### Procedure
Applied Offline Asymmetric SVD Surgery to the raw adapter before zipping.

Surgery steps executed:
1. Rename `base_model.model.model.*` → `base_model.model.backbone.*`
2. Unfuse MoE experts: `experts.w1` → `experts.{i}.up_proj`, `experts.w2` → `experts.{i}.down_proj`
3. Remove `experts.w3` zero tensors entirely
4. Merge `gate_proj + x_proj` → `in_proj`
5. Remove all zero-shape tensors
6. Generate `submission.zip` from the surgery output

### Validation results

| Check | Result |
|---|---|
| `base_model.model.model` prefix | absent |
| `base_model.model.backbone` prefix | present |
| `.experts.w1 / .w2 / .w3` | 0 tensors |
| `.gate_proj / .x_proj` | 0 tensors |
| `.in_proj` | 46 tensors |
| `.up_proj` | 5934 tensors |
| `.down_proj` | 5934 tensors |
| zero-shape tensors | 0 |
| total output tensors | 12010 |

### Kaggle LB result
**0.85** — metric error resolved, submission processed successfully.

---

## 3. Metric Error — Cause Summary

The error `Evaluation metric raised an unexpected error` was not a model quality issue.
It was a submission format incompatibility:

- The Kaggle harness loads the adapter before running inference.
- Raw adapter keys cause a load error at the harness level.
- The harness surfaces loader errors as metric errors, not as format errors, making the root cause non-obvious.
- Surgery normalises the adapter to the harness-expected schema, resolving the load error.

---

## 4. Canonical Submission Generation Path

```
Raw adapter (training output)
    │
    ▼
Offline Asymmetric SVD Surgery
    ├── key rename:  base_model.model.model → base_model.model.backbone
    ├── expert unfuse:  w1→up_proj, w2→down_proj, w3 removed
    ├── projection merge:  gate_proj + x_proj → in_proj
    └── zero-shape removal
    │
    ▼
Surgery-validated adapter
    │
    ▼
submission.zip generation
    │
    ▼
Kaggle submission
```

**Prohibited shortcut:** raw adapter → submission.zip (produces metric error)

---

## 5. Baseline Identity

- LB score: **0.85**
- Submission path: surgery-processed adapter → submission.zip
- Metric error: resolved
- Adopted as: **RECOVERED_BASELINE_0.85**

All future experiments use this result as the baseline comparison point.
No experiment is valid unless compared against this score via the same submission path.
