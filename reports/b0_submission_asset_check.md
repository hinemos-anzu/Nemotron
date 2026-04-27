# B0 Submission Asset Check

## Status
HOLD — awaiting Kaggle execution
Date prepared: 2026-04-27

## Reference baseline
- commit: fbb6621
- baseline LB: 0.85
- expected tensor count: 12010

---

## 1. submission.zip

| Field | Expected | Observed |
|---|---|---|
| path | /kaggle/working/submission.zip | — |
| size | ≈ baseline size | — |
| contains adapter_config.json | YES | — |
| contains adapter_model.* | YES | — |

---

## 2. Key Prefix Check

| Prefix | Required | Observed count |
|---|---|---|
| `base_model.model.backbone` | MUST be present | — |
| `base_model.model.model` | MUST be absent (= 0) | — |
| `*.experts.w1` | MUST be absent (= 0) | — |
| `*.experts.w2` | MUST be absent (= 0) | — |
| `*.experts.w3` | MUST be absent (= 0) | — |
| `*.gate_proj` | MUST be absent (= 0) | — |
| `*.x_proj` | MUST be absent (= 0) | — |
| `*.in_proj` | MUST be present | — |
| `*.up_proj` | MUST be present | — |
| `*.down_proj` | MUST be present | — |

---

## 3. Tensor Count Check

| Metric | Expected | Observed |
|---|---|---|
| `.in_proj` tensors | 46 | — |
| `.up_proj` tensors | 5934 | — |
| `.down_proj` tensors | 5934 | — |
| zero-shape tensors | 0 | — |
| total output tensors | 12010 | — |

---

## 4. Surgery Diff vs fbb6621

No surgery changes permitted for B0.
If any diff is detected between this run and fbb6621 surgery output, B0 is invalid.

| Diff check | Expected | Observed |
|---|---|---|
| surgery steps identical to fbb6621 | YES | — |
| no extra key renames | YES | — |
| no RANK_MAP changes | YES | — |
| no target_modules changes | YES | — |
| no inference_mode changes | YES | — |

---

## 5. Validation Gate Result

```
[ ] base_model.model.model absent
[ ] base_model.model.backbone present
[ ] experts.w1 absent
[ ] experts.w2 absent
[ ] experts.w3 absent
[ ] gate_proj absent
[ ] x_proj absent
[ ] in_proj present
[ ] zero-shape tensors = 0
[ ] tensor count = 12010
[ ] no surgery diff vs fbb6621

Overall: PENDING — fill after Kaggle run
```

---

## 6. Change Log

| Date | Event |
|---|---|
| 2026-04-27 | Asset check template created, status HOLD |
