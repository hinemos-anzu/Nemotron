# Nemotron Kaggle Bundle
**Version:** v1
**Generated:** 2026-04-20T22:55:12+00:00
**Repository:** hinemos-anzu/Nemotron
**Branch:** planner/kaggle-bundle-materialization-v1

---

## Purpose

This bundle provides all files required for the Kaggle execution role to run the
Nemotron baseline evaluation path in the Kaggle source-of-truth environment.
The bundle is designed to be uploaded as a Kaggle Dataset and mounted without
any additional file-finding steps by the execution role.

---

## Key Files

| Role | Path in Bundle |
|---|---|
| **Entrypoint** (run this) | `kaggle/run_baseline_with_debug.py` |
| Baseline script (do not change logic) | `kaggle/original-nemotron-asymmetric-svd-26041602.py` |
| Debug harness library | `scripts/debug_harness.py` |
| Eval set — Quick Gate v1 | `data/eval/quick_gate_v1.csv` |
| Eval set — Diagnostic v1 | `data/eval/diagnostic_v1.csv` |
| Eval set — Promotion v1 | `data/eval/promotion_v1.csv` |
| Category manifest v1 | `data/eval/category_manifest_v1.csv` |
| Baseline reference report | `reports/eval/baseline_reference_v1.md` |
| Execution ticket | `tickets/TICKET_S1_6A_kaggle_runtime_execution_enablement_for_baseline_path.md` |

---

## Kaggle Notebook — Assumed Execution Commands

```bash
# 1. Set working directory to bundle root
cd /kaggle/working/Nemotron

# 2. Run entrypoint (Quick Gate set, default)
python kaggle/run_baseline_with_debug.py --eval-set quick_gate_v1

# 3. Or run specific eval set
python kaggle/run_baseline_with_debug.py --eval-set diagnostic_v1
python kaggle/run_baseline_with_debug.py --eval-set promotion_v1
```

---

## Script Roles

- **`kaggle/run_baseline_with_debug.py`**
  Entry point. Initialises `KaggleRunHarness`, dumps environment info,
  runs the baseline script as a subprocess with stdout/stderr tee,
  and writes all 9 debug report files automatically.

- **`kaggle/original-nemotron-asymmetric-svd-26041602.py`**
  Baseline execution script — original Nemotron asymmetric-SVD submission path.
  PROTECTED ASSET: do not change conversion flow, SVD surgery, key rename,
  merge logic, expert unfuse, or submission.zip generation.
  Currently contains a placeholder body; Kaggle execution role inserts real
  notebook content at the marked section.

- **`scripts/debug_harness.py`**
  Debug harness library. Provides `KaggleRunHarness`, `StageLogger`,
  `EnvironmentDumper`, `InputManifestWriter`, `OutputArtifactTracker`,
  `ErrorReporter`, `ReproStepsWriter`, `RunSummaryWriter`.
  On every run (success or failure) it auto-generates:
  `reports/kaggle_run/run_summary.md`,
  `reports/kaggle_run/environment_info.md`,
  `reports/kaggle_run/error_report.md`,
  `logs/kaggle_run/stage_log.jsonl`, and more.

---

## Debug Output Location After Run

| File | Location |
|---|---|
| Run summary | `reports/kaggle_run/run_summary.md` |
| Environment info | `reports/kaggle_run/environment_info.md` |
| Error report | `reports/kaggle_run/error_report.md` |
| Stage log | `logs/kaggle_run/stage_log.jsonl` |
| Stdout | `logs/kaggle_run/stdout.log` |
| Stderr | `logs/kaggle_run/stderr.log` |

Return all files under `reports/kaggle_run/` and `logs/kaggle_run/` to Generator
on failure, or on success along with submission artifacts.
