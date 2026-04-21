# Nemotron Kaggle Bundle — Manifest v1
**Generated:** 2026-04-20
**Repository:** hinemos-anzu/Nemotron
**Bundle root path:** `artifacts/kaggle/Nemotron_kaggle_bundle/`
**ZIP path:** `artifacts/kaggle/Nemotron_kaggle_bundle.zip`
**Working branch:** `planner/kaggle-bundle-materialization-v1`

---

## Entrypoint for Kaggle Execution Role

```
python kaggle/run_baseline_with_debug.py --eval-set quick_gate_v1
```

Run from the bundle root (`/kaggle/working/Nemotron/`).

---

## Shortest Procedure for Kaggle Execution Role

1. Upload `Nemotron_kaggle_bundle.zip` as a new Kaggle Dataset.
2. In the Kaggle Notebook, click **Add Data** and attach that dataset.
3. Copy or symlink to working directory:
   ```
   cp -r /kaggle/input/<dataset-name>/Nemotron_kaggle_bundle /kaggle/working/Nemotron
   cd /kaggle/working/Nemotron
   ```
4. Run the entrypoint:
   ```
   python kaggle/run_baseline_with_debug.py --eval-set quick_gate_v1
   ```
5. On failure: return `reports/kaggle_run/` and `logs/kaggle_run/` to Generator.
6. On success: return submission artifacts from `/kaggle/working/`.

---

## Required Files — GitHub Existence Confirmation

| # | Logical Name | Repo Path | Bundle Path | Branch | Commit SHA | GitHub Exists | Required for Run |
|---|---|---|---|---|---|---|---|
| 1 | `debug_harness.py` | `scripts/debug_harness.py` | `Nemotron_kaggle_bundle/scripts/debug_harness.py` | `planner/debug-output-hardening-v1` | `70ef3d91b507f11018cc2259cd56d6d1e14d7247` | **YES** | YES |
| 2 | `run_baseline_with_debug.py` | `kaggle/run_baseline_with_debug.py` | `Nemotron_kaggle_bundle/kaggle/run_baseline_with_debug.py` | `planner/debug-output-hardening-v1` | `70ef3d91b507f11018cc2259cd56d6d1e14d7247` | **YES** | YES |
| 3 | `original-nemotron-asymmetric-svd-26041602.py` | `kaggle/original-nemotron-asymmetric-svd-26041602.py` | `Nemotron_kaggle_bundle/kaggle/original-nemotron-asymmetric-svd-26041602.py` | `planner/debug-output-hardening-v1` | `70ef3d91b507f11018cc2259cd56d6d1e14d7247` | **YES** | YES |
| 4 | `quick_gate_v1.csv` | `data/eval/quick_gate_v1.csv` | `Nemotron_kaggle_bundle/data/eval/quick_gate_v1.csv` | `planner/kaggle-bundle-materialization-v1` | `9a3046f1e42560667a615c921814bf6ec29a398e` (source JSONL) | **YES** | YES |
| 5 | `diagnostic_v1.csv` | `data/eval/diagnostic_v1.csv` | `Nemotron_kaggle_bundle/data/eval/diagnostic_v1.csv` | `planner/kaggle-bundle-materialization-v1` | `9a3046f1e42560667a615c921814bf6ec29a398e` (source JSONL) | **YES** | YES |
| 6 | `promotion_v1.csv` | `data/eval/promotion_v1.csv` | `Nemotron_kaggle_bundle/data/eval/promotion_v1.csv` | `planner/kaggle-bundle-materialization-v1` | `9a3046f1e42560667a615c921814bf6ec29a398e` (source JSONL) | **YES** | YES |
| 7 | `category_manifest_v1.csv` | `data/eval/category_manifest_v1.csv` | `Nemotron_kaggle_bundle/data/eval/category_manifest_v1.csv` | `claude/nemotron-experiment-framework-7K8yG` | `9a3046f1e42560667a615c921814bf6ec29a398e` | **YES** | YES |
| 8 | `baseline_reference_v1.md` | `reports/eval/baseline_reference_v1.md` | `Nemotron_kaggle_bundle/reports/eval/baseline_reference_v1.md` | `claude/nemotron-experiment-framework-7K8yG` | `9a3046f1e42560667a615c921814bf6ec29a398e` | **YES** | NO |
| 9 | `baseline_reference_v1_measured.md` | `reports/eval/baseline_reference_v1_measured.md` | `Nemotron_kaggle_bundle/reports/eval/baseline_reference_v1_measured.md` | `planner/kaggle-bundle-materialization-v1` | *(placeholder — this branch)* | **YES (placeholder)** | NO |
| 10 | `baseline_reference_v1_measured_kaggle.md` | `reports/eval/baseline_reference_v1_measured_kaggle.md` | `Nemotron_kaggle_bundle/reports/eval/baseline_reference_v1_measured_kaggle.md` | `planner/kaggle-bundle-materialization-v1` | *(placeholder — this branch)* | **YES (placeholder)** | NO |
| 11 | `TICKET_S1_5_baseline_measured_reference_v1.md` | `tickets/TICKET_S1_5_baseline_measured_reference_v1.md` | `Nemotron_kaggle_bundle/tickets/TICKET_S1_5_baseline_measured_reference_v1.md` | `claude/nemotron-experiment-framework-7K8yG` | `6193e3eaa9214558b30306ceea0154ba270ae6d6` | **YES** | NO |
| 12 | `TICKET_S1_6A_...md` | `tickets/TICKET_S1_6A_kaggle_runtime_execution_enablement_for_baseline_path.md` | `Nemotron_kaggle_bundle/tickets/TICKET_S1_6A_...md` | `claude/nemotron-experiment-framework-7K8yG` | `6e778d3263cd20b4c21a2b3d9a421323c9376689` | **YES** | NO |

---

## Notes on CSV Eval Files

Files `quick_gate_v1.csv`, `diagnostic_v1.csv`, `promotion_v1.csv` were converted
from their corresponding `.jsonl` sources (commit `9a3046f1e425`) on this branch.
The CSV SHA listed above refers to the source JSONL commit. The CSV files themselves
are committed on `planner/kaggle-bundle-materialization-v1` in the same commit as
this manifest.

---

## Bundle ZIP Verification

| Check | Result |
|---|---|
| Root folder `Nemotron_kaggle_bundle/` present in ZIP | **YES** |
| ZIP not flat (no files directly at zip root without subfolder) | **YES** |
| ZIP entry count | 13 |
| ZIP size | 38,091 bytes |
