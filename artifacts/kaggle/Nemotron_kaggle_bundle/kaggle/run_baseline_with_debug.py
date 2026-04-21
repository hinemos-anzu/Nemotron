"""
kaggle/run_baseline_with_debug.py

Entry point: runs the baseline Kaggle script wrapped in the debug harness.

This wrapper is the only file that should be invoked by the Kaggle execution role
when running the baseline path.  It:
  1. Initializes KaggleRunHarness with the frozen input paths and expected artifacts.
  2. Dumps environment info before the baseline starts.
  3. Captures stdout/stderr to log files with console tee.
  4. Writes stage_log.jsonl as the baseline progresses.
  5. Always writes error_report.md (NO_ERROR or full traceback + classification).
  6. Always writes output_artifacts.md, input_manifest.md, run_summary.md, repro_steps.md.

Generator can review reports/kaggle_run/ and logs/kaggle_run/ without any
additional observation from the Kaggle execution role.

Usage:
    python kaggle/run_baseline_with_debug.py [--eval-set quick_gate_v1|diagnostic_v1|promotion_v1|all]

Environment variables (optional overrides):
    NEMOTRON_EVAL_SET   — which eval set to run (default: quick_gate_v1)
    NEMOTRON_REPORT_DIR — output report root (default: reports/kaggle_run)
    NEMOTRON_LOG_DIR    — output log root (default: logs/kaggle_run)
"""

import argparse
import os
import sys
from pathlib import Path

# Allow import from scripts/ regardless of cwd
_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from debug_harness import KaggleRunHarness  # noqa: E402

# ─── path constants ───────────────────────────────────────────────────────────

BASELINE_SCRIPT = _REPO_ROOT / "kaggle" / "original-nemotron-asymmetric-svd-26041602.py"

EVAL_SETS = {
    "quick_gate_v1": _REPO_ROOT / "data" / "eval" / "quick_gate_v1.csv",
    "diagnostic_v1": _REPO_ROOT / "data" / "eval" / "diagnostic_v1.csv",
    "promotion_v1":  _REPO_ROOT / "data" / "eval" / "promotion_v1.csv",
}

REQUIRED_INPUTS = [
    # Kaggle filesystem roots (standard; /kaggle/temp is NOT standard — excluded)
    "/kaggle/input",
    "/kaggle/working",
    # Frozen evaluation assets (CSV — JSONL not required at runtime)
    str(_REPO_ROOT / "data" / "eval" / "quick_gate_v1.csv"),
    str(_REPO_ROOT / "data" / "eval" / "diagnostic_v1.csv"),
    str(_REPO_ROOT / "data" / "eval" / "promotion_v1.csv"),
    str(_REPO_ROOT / "data" / "eval" / "category_manifest_v1.csv"),
    # Baseline script itself
    str(BASELINE_SCRIPT),
    # Model / adapter assets (Kaggle dataset mounts — informational; absence is non-blocking)
    "/kaggle/input/nemotron-adapter/adapter_model.safetensors",
    "/kaggle/input/nemotron-adapter/adapter_config.json",
    "/kaggle/input/nemotron-adapter/README.md",
    "/kaggle/input/nemotron-adapter/checkpoint_complete",
]

EXPECTED_ARTIFACTS = [
    "/kaggle/working/submission.zip",
    "/kaggle/working/run_complete.flag",
    "/kaggle/working/predictions.jsonl",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline with debug harness")
    parser.add_argument(
        "--eval-set",
        default=os.environ.get("NEMOTRON_EVAL_SET", "quick_gate_v1"),
        choices=list(EVAL_SETS.keys()) + ["all"],
        help="Evaluation set to run (default: quick_gate_v1)",
    )
    parser.add_argument(
        "--report-dir",
        default=os.environ.get("NEMOTRON_REPORT_DIR", "reports/kaggle_run"),
        help="Directory for report files",
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("NEMOTRON_LOG_DIR", "logs/kaggle_run"),
        help="Directory for log files",
    )
    args = parser.parse_args()

    eval_set = args.eval_set
    report_dir = _REPO_ROOT / args.report_dir
    log_dir = _REPO_ROOT / args.log_dir

    # Inject which eval set to run into the baseline environment
    extra_env: dict = {
        "NEMOTRON_EVAL_SET": eval_set,
        "NEMOTRON_REPO_ROOT": str(_REPO_ROOT),
    }
    if eval_set != "all":
        extra_env["NEMOTRON_EVAL_CSV"] = str(EVAL_SETS[eval_set])

    print(f"[run_baseline_with_debug] eval_set={eval_set}", flush=True)
    print(f"[run_baseline_with_debug] report_dir={report_dir}", flush=True)
    print(f"[run_baseline_with_debug] log_dir={log_dir}", flush=True)

    harness = KaggleRunHarness(
        script_path=BASELINE_SCRIPT,
        run_output_dir=report_dir,
        log_output_dir=log_dir,
        required_inputs=REQUIRED_INPUTS,
        expected_artifacts=EXPECTED_ARTIFACTS,
        extra_env=extra_env,
    )
    return harness.run()


if __name__ == "__main__":
    sys.exit(main())
