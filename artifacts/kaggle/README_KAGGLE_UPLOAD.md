# README — Kaggle Upload Instructions
**For:** Kaggle Execution Role
**Bundle file:** `artifacts/kaggle/Nemotron_kaggle_bundle.zip`
**Entrypoint:** `python kaggle/run_baseline_with_debug.py`

---

## What to Upload

Upload **`Nemotron_kaggle_bundle.zip`** as a new Kaggle Dataset.

- Do NOT unzip before uploading.
- Do NOT upload individual files; upload the ZIP as a single dataset.
- The ZIP contains one root folder `Nemotron_kaggle_bundle/` with all files
  in their correct relative paths.

---

## How to Add Data in Kaggle Notebook

1. Open your Kaggle Notebook.
2. Click **Add Data** (right panel).
3. Search for the dataset you just uploaded (e.g. `nemotron-kaggle-bundle`).
4. Click **Add**.
5. The bundle will be mounted at:
   `/kaggle/input/<dataset-name>/Nemotron_kaggle_bundle/`

---

## Working Directory Setup

Run in the first notebook cell:

```python
import shutil, os
src = "/kaggle/input/<dataset-name>/Nemotron_kaggle_bundle"
dst = "/kaggle/working/Nemotron"
shutil.copytree(src, dst, dirs_exist_ok=True)
os.chdir(dst)
print("Working directory:", os.getcwd())
```

Replace `<dataset-name>` with the actual Kaggle dataset slug.

---

## Execution Command

Run the entrypoint (one `Add Data` per session is enough):

```bash
# Default: Quick Gate set only (fastest, ~75 samples)
python kaggle/run_baseline_with_debug.py --eval-set quick_gate_v1

# Full diagnostic set (~150 samples)
python kaggle/run_baseline_with_debug.py --eval-set diagnostic_v1

# Full promotion set (~400 samples)
python kaggle/run_baseline_with_debug.py --eval-set promotion_v1
```

---

## After the Run — What to Return to Generator

### On SUCCESS
Return all of the following:
- `/kaggle/working/submission.zip`
- `/kaggle/working/predictions.jsonl`
- `/kaggle/working/run_complete.flag`
- `reports/kaggle_run/run_summary.md`
- `reports/kaggle_run/environment_info.md`
- `logs/kaggle_run/stage_log.jsonl`

### On FAILURE
Return all of the following (Generator needs these to diagnose):
- `reports/kaggle_run/run_summary.md`
- `reports/kaggle_run/error_report.md` ← **most important**
- `reports/kaggle_run/environment_info.md`
- `reports/kaggle_run/input_manifest.md`
- `reports/kaggle_run/output_artifacts.md`
- `logs/kaggle_run/stage_log.jsonl`
- `logs/kaggle_run/stdout.log`
- `logs/kaggle_run/stderr.log`

Do NOT paraphrase or summarise the error. Return the raw file contents.

---

## Report Paths Summary

| File | Path in Kaggle Notebook |
|---|---|
| Run summary | `reports/kaggle_run/run_summary.md` |
| Environment info | `reports/kaggle_run/environment_info.md` |
| Input manifest | `reports/kaggle_run/input_manifest.md` |
| Output artifacts | `reports/kaggle_run/output_artifacts.md` |
| **Error report** | **`reports/kaggle_run/error_report.md`** |
| Repro steps | `reports/kaggle_run/repro_steps.md` |
| Stage log | `logs/kaggle_run/stage_log.jsonl` |
| Stdout | `logs/kaggle_run/stdout.log` |
| Stderr | `logs/kaggle_run/stderr.log` |

---

## One Add Data Per Run

You only need to add the dataset **once per session**. Re-running
`python kaggle/run_baseline_with_debug.py` in the same notebook session
overwrites the report files from the previous run.

---

## Baseline Script Note

The baseline script `kaggle/original-nemotron-asymmetric-svd-26041602.py`
currently contains a **placeholder body** (raises `NotImplementedError` until
the real notebook content is inserted).

This means the first run will produce:
- A `FAILED` run summary
- `error_report.md` with `DEPENDENCY_MISSING` or `FILE_NOT_FOUND` classification
- A complete `stage_log.jsonl` up to the failure point
- Full `stderr.log` with traceback

This is expected. Return the debug reports to Generator for the next step
(TICKET_S1_6A: inserting the real baseline execution body).
