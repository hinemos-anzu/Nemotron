# Nemotron Kaggle Bundle — Missing Files Report v1
**Generated:** 2026-04-20
**Bundle version:** v1
**Working branch:** `planner/kaggle-bundle-materialization-v1`

---

## Summary

| Category | Count |
|---|---|
| Files absent from GitHub (required for run) | **0** |
| Files absent from GitHub (optional) | **2** |
| Files included as placeholder | **2** |
| Run-blocking missing files | **0** |

---

## Files Missing from GitHub at Bundle Creation Time

### Optional — Included as Placeholder

The following files did not exist on any branch at the time of bundle creation.
They have been included in the bundle as **PENDING placeholders** so the directory
structure is complete and the Kaggle execution role does not need to handle
missing-path errors.

| File | Repo Path | Reason Missing | Blocks Run? |
|---|---|---|---|
| `baseline_reference_v1_measured.md` | `reports/eval/baseline_reference_v1_measured.md` | Requires TICKET_S1_5 model inference run (not yet executed) | **NO** |
| `baseline_reference_v1_measured_kaggle.md` | `reports/eval/baseline_reference_v1_measured_kaggle.md` | Requires TICKET_S1_6 / S1_6A Kaggle execution (not yet executed) | **NO** |

Both files are reference-only. They do not affect script execution or debug harness
operation.

---

## Run-Blocking Missing Files

**NONE.**

All files required for `python kaggle/run_baseline_with_debug.py` to launch
are present in the bundle:
- `scripts/debug_harness.py` — present
- `kaggle/run_baseline_with_debug.py` — present
- `kaggle/original-nemotron-asymmetric-svd-26041602.py` — present
- `data/eval/quick_gate_v1.csv` — present
- `data/eval/category_manifest_v1.csv` — present

---

## Substitute / Workaround

None required. All run-blocking files are present.

---

## Next Required Complementary Work

| File | Complementary Ticket | Action |
|---|---|---|
| `baseline_reference_v1_measured.md` | TICKET_S1_5 | Run inference on frozen Quick Gate / Diagnostic / Promotion sets; populate accuracy values |
| `baseline_reference_v1_measured_kaggle.md` | TICKET_S1_6A | Kaggle execution role runs baseline path and returns per-sample results; Generator populates this file |

---

## NO_MISSING_FILES (for run-blocking files)

All files required to start execution are present. The 2 placeholder files are
reference documents only and do not affect execution.
