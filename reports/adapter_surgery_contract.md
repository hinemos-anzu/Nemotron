# Adapter Surgery Contract v1

## Status
ENFORCED — 2026-04-27
Applies to all submissions from Recovered Baseline 0.85 onwards.

---

## 1. Core Rule

> **Raw adapter → submission.zip is PROHIBITED.**
> Every submission.zip MUST be generated through the Offline Asymmetric SVD Surgery pipeline.

Violation of this rule results in:
- `Evaluation metric raised an unexpected error` on Kaggle
- No LB score
- Wasted submission slot

---

## 2. Required Surgery Steps

Surgery must be applied in the following order:

### Step 1: Key prefix rename

```
base_model.model.model.*  →  base_model.model.backbone.*
```

- Applies to all keys containing the double `model.model` segment.
- The Kaggle harness expects `backbone` as the second model-level key segment.
- Skipping this step causes a model load failure.

### Step 2: MoE expert unfuse

```
*.experts.w1  →  *.experts.{i}.up_proj
*.experts.w2  →  *.experts.{i}.down_proj
*.experts.w3  →  REMOVE (zero tensors, not needed)
```

- Raw adapters store MoE expert projections as fused `w1/w2/w3` tensors.
- The evaluation harness expects unfused per-expert named projections.
- `w3` produces zero tensors after SVD; remove them entirely.
- `{i}` is the expert index, zero-based.

### Step 3: Gate / x projection merge

```
*.gate_proj  ┐
             ├─  →  *.in_proj
*.x_proj     ┘
```

- Raw adapters store gating and value projections separately.
- The harness expects them merged into a single `in_proj` tensor.
- Merge must preserve the correct concatenation order (gate first, x second).

### Step 4: Zero-shape tensor removal

- After all renames and merges, remove any remaining zero-shape tensors.
- Zero-shape tensors cause runtime errors during adapter loading.
- A tensor is zero-shape if any dimension is 0.

### Step 5: Zip generation

- Only after all four steps above are validated may the zip be generated.
- Run the mandatory validation gate (Section 4) before creating the zip.

---

## 3. Mandatory Transformations Summary

| Raw key / format | Surgery output | Rule |
|---|---|---|
| `base_model.model.model.*` | `base_model.model.backbone.*` | rename |
| `*.experts.w1` | `*.experts.{i}.up_proj` | unfuse |
| `*.experts.w2` | `*.experts.{i}.down_proj` | unfuse |
| `*.experts.w3` | removed | eliminate zero tensors |
| `*.gate_proj` | merged into `*.in_proj` | merge |
| `*.x_proj` | merged into `*.in_proj` | merge |
| zero-shape tensors | removed | eliminate |

---

## 4. Mandatory Validation Gate

Run this validation immediately after surgery and before zip generation.
Fail = do not zip, do not submit.

```
PASS conditions (all must be true):

[ ] base_model.model.model key count == 0
[ ] base_model.model.backbone key count > 0
[ ] experts.w1 key count == 0
[ ] experts.w2 key count == 0
[ ] experts.w3 key count == 0
[ ] gate_proj key count == 0
[ ] x_proj key count == 0
[ ] in_proj key count > 0
[ ] zero-shape tensor count == 0
[ ] total tensor count is in expected range (baseline: 12010)
```

If any condition is false, the surgery output is invalid.
Do not proceed to zip generation.
Re-run surgery from the raw adapter.

---

## 5. Prohibited Actions

- **NEVER** zip the raw adapter directly
- **NEVER** skip surgery because the adapter "looks correct"
- **NEVER** revert to pre-surgery zip generation code
- **NEVER** treat `submission.csv` as a submission asset for this competition
- **NEVER** run surgery only partially (all steps are required)

---

## 6. Experiment Boundary Rules

Surgery is a **fixed pipeline**, not an experimental variable.

- Surgery steps MUST NOT be changed between experiments unless Planner explicitly designates surgery as the single main variable for that experiment.
- Changing surgery steps and model/training variables in the same experiment is prohibited.
- If surgery is changed experimentally, the old and new surgery pipelines must both be validated against the Recovered Baseline before further experiments proceed.

---

## 7. Contract History

| Date | Event |
|---|---|
| 2026-04-27 | Contract established at Recovered Baseline 0.85 freeze |
