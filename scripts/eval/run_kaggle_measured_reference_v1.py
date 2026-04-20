"""
Entry script for S1.6 Kaggle measured-reference execution handoff.

Usage:
  python scripts/eval/run_kaggle_measured_reference_v1.py

Required frozen inputs:
  - data/eval/quick_gate_v1.csv
  - data/eval/diagnostic_v1.csv
  - data/eval/promotion_v1.csv
  - data/eval/category_manifest_v1.csv

Required baseline path:
  - kaggle/original-nemotron-asymmetric-svd-26041602.py

Primary outputs:
  - data/eval/baseline_measured_results_*_kaggle.csv
  - data/eval/baseline_measured_category_summary_v1_kaggle.csv
  - reports/kaggle_run/*
  - logs/kaggle_run/*

Optional debug hook:
  - FORCE_DEBUG_CRASH=1 triggers synthetic failure-path validation.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_EVAL = ROOT / "data" / "eval"
REPORT_DIR = ROOT / "reports" / "kaggle_run"
LOG_DIR = ROOT / "logs" / "kaggle_run"

BASELINE_SCRIPT = ROOT / "kaggle" / "original-nemotron-asymmetric-svd-26041602.py"

SPLITS = {
    "quick_gate_v1": DATA_EVAL / "quick_gate_v1.csv",
    "diagnostic_v1": DATA_EVAL / "diagnostic_v1.csv",
    "promotion_v1": DATA_EVAL / "promotion_v1.csv",
}
FROZEN_FILES = [
    DATA_EVAL / "quick_gate_v1.csv",
    DATA_EVAL / "diagnostic_v1.csv",
    DATA_EVAL / "promotion_v1.csv",
    DATA_EVAL / "category_manifest_v1.csv",
]

RESULT_COLUMNS = [
    "sample_id",
    "split_name",
    "category",
    "baseline_prediction_status",
    "baseline_correctness",
    "format_failure_flag",
    "extraction_failure_flag",
    "runtime_status",
    "notes",
]
SUMMARY_COLUMNS = [
    "split_name",
    "category",
    "sample_count",
    "measured_pass_count",
    "measured_fail_count",
    "measured_error_count",
    "format_failure_count",
    "extraction_failure_count",
    "pass_rate_or_accuracy",
]

EXPECTED_REPORTS = [
    REPORT_DIR / "run_summary.md",
    REPORT_DIR / "environment_info.md",
    REPORT_DIR / "input_manifest.md",
    REPORT_DIR / "output_artifacts.md",
    REPORT_DIR / "error_report.md",
    REPORT_DIR / "repro_steps.md",
]
EXPECTED_LOGS = [
    LOG_DIR / "stdout.log",
    LOG_DIR / "stderr.log",
    LOG_DIR / "stage_log.jsonl",
]
EXPECTED_DATA = [
    DATA_EVAL / "baseline_measured_results_quick_gate_v1_kaggle.csv",
    DATA_EVAL / "baseline_measured_results_diagnostic_v1_kaggle.csv",
    DATA_EVAL / "baseline_measured_results_promotion_v1_kaggle.csv",
    DATA_EVAL / "baseline_measured_category_summary_v1_kaggle.csv",
]


@dataclass
class RunState:
    start_time: str
    start_epoch: float
    end_time: str = ""
    runtime_result: str = "BLOCKED"
    last_successful_stage: str = "none"
    failed_stage: str = ""
    exception_type: str = ""
    exception_message: str = ""
    traceback_text: str = ""
    baseline_stdout: str = ""
    baseline_stderr: str = ""
    baseline_command: str = f"python {BASELINE_SCRIPT}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def safe_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stage_log(stage: str, status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    entry = {
        "timestamp": now_iso(),
        "stage": stage,
        "status": status,
        "message": message,
        "extra": extra or {},
    }
    with (LOG_DIR / "stage_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def dep_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception as e:  # noqa: BLE001
        return f"UNAVAILABLE ({type(e).__name__}: {e})"


def collect_environment_info() -> str:
    # torch probing is optional and must not break logging.
    cuda_available = "UNAVAILABLE"
    gpu_name = "UNAVAILABLE"
    torch_version = dep_version("torch")
    try:
        import torch  # type: ignore

        cuda_available = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            gpu_name = str(torch.cuda.get_device_name(0))
        else:
            gpu_name = "CPU_ONLY"
    except Exception as e:  # noqa: BLE001
        cuda_available = f"UNAVAILABLE ({type(e).__name__}: {e})"
        gpu_name = "UNAVAILABLE"

    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    required_paths = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("/kaggle/temp"),
        BASELINE_SCRIPT,
        Path("/kaggle/input/nvidia-nemotron-3-reasoning-challenge"),
        Path("/kaggle/input/models"),
    ]

    path_lines = []
    for p in required_paths:
        path_lines.append(f"- `{p}`: {'EXISTS' if p.exists() else 'MISSING'}")

    important_env = [
        "CUDA_VISIBLE_DEVICES",
        "TRANSFORMERS_OFFLINE",
        "HF_HOME",
        "KAGGLE_KERNEL_RUN_TYPE",
        "KAGGLE_URL_BASE",
        "PYTHONPATH",
        "PATH",
    ]
    env_lines = [f"- `{k}`: `{os.environ.get(k, '<unset>')}`" for k in important_env]

    deps = {
        "torch": torch_version,
        "transformers": dep_version("transformers"),
        "pandas": dep_version("pandas"),
        "sympy": dep_version("sympy"),
        "safetensors": dep_version("safetensors"),
        "kagglehub": dep_version("kagglehub"),
    }
    dep_lines = [f"- `{k}`: `{v}`" for k, v in deps.items()]

    return "\n".join(
        [
            "# Environment Info",
            "",
            f"- timestamp: `{now_iso()}`",
            f"- git branch: `{branch}`",
            f"- git commit: `{commit}`",
            f"- Python version: `{sys.version}`",
            f"- OS / platform: `{platform.platform()}`",
            f"- current working directory: `{os.getcwd()}`",
            f"- CUDA availability: `{cuda_available}`",
            f"- GPU name: `{gpu_name}`",
            "",
            "## Dependencies",
            *dep_lines,
            "",
            "## Important environment variables",
            *env_lines,
            "",
            "## Required filesystem roots and paths",
            *path_lines,
            "",
        ]
    )


def build_input_manifest() -> str:
    lines = ["# Input Manifest", "", "## Required input files/directories"]
    required = [*FROZEN_FILES, *SPLITS.values(), BASELINE_SCRIPT]

    missing = []
    for p in required:
        exists = p.exists()
        if not exists:
            missing.append(str(p))
        size = p.stat().st_size if exists and p.is_file() else 0
        count = len(list(p.iterdir())) if exists and p.is_dir() else "-"
        head = ""
        if exists and p.is_file() and p.suffix in {".csv", ".md", ".py"}:
            try:
                head = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0][:120]
            except Exception:  # noqa: BLE001
                head = "<unreadable>"
        lines.append(f"- path: `{p}` | exists: `{exists}` | size: `{size}` | count: `{count}` | head: `{head}`")

    lines.extend(["", "## Missing inputs"])
    if missing:
        lines.extend([f"- `{m}`" for m in missing])
    else:
        lines.append("- NONE")
    lines.append("")
    return "\n".join(lines)


def frozen_state() -> tuple[str, str]:
    tracked = [str(p.relative_to(ROOT)) for p in FROZEN_FILES]
    out = subprocess.check_output(["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True)
    if out.strip():
        return "MUTATED", out.strip().replace("\n", " | ")
    return "UNMODIFIED", "frozen evaluation membership unchanged"


def classify_blocker(exc_type: str, exc_msg: str) -> str:
    txt = f"{exc_type} {exc_msg}".lower()
    if "import" in txt:
        return "IMPORT_ERROR"
    if "no module named" in txt:
        return "DEPENDENCY_MISSING"
    if "no such file" in txt or "not found" in txt:
        return "FILE_NOT_FOUND"
    if "kaggle" in txt or "missing_kaggle_paths" in txt:
        return "ENVIRONMENT_MISSING"
    if txt.strip():
        return "RUNTIME_CRASH"
    return "UNKNOWN"


def run_baseline_and_capture(state: RunState) -> tuple[str, str]:
    stage_log("model_load_start", "START", "starting baseline script process")
    proc = subprocess.run(
        ["python", str(BASELINE_SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    state.baseline_stdout = proc.stdout or ""
    state.baseline_stderr = proc.stderr or ""
    safe_write(LOG_DIR / "stdout.log", state.baseline_stdout)
    safe_write(LOG_DIR / "stderr.log", state.baseline_stderr)

    if proc.returncode == 0:
        stage_log("model_load_end", "SUCCESS", "baseline script returned 0")
        return "OK", "baseline script execution succeeded"

    stage_log("model_load_end", "FAILED", "baseline script returned non-zero", {"returncode": proc.returncode})
    return "BLOCKED", f"returncode={proc.returncode}; stderr_head={(state.baseline_stderr[:400]).replace(chr(10), ' | ')}"


def build_rows(split_name: str, src_rows: list[dict[str, str]], runtime_status: str, note: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for r in src_rows:
        rows.append(
            {
                "sample_id": r["sample_id"],
                "split_name": split_name,
                "category": r["category"],
                "baseline_prediction_status": "NOT_RUN" if runtime_status != "OK" else "PREDICTED",
                "baseline_correctness": "NA",
                "format_failure_flag": "0",
                "extraction_failure_flag": "0",
                "runtime_status": runtime_status,
                "notes": note,
            }
        )
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    bucket: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in rows:
        key = (r["split_name"], r["category"])
        b = bucket[key]
        b["sample_count"] += 1
        b["format_failure_count"] += int(r["format_failure_flag"])
        b["extraction_failure_count"] += int(r["extraction_failure_flag"])
        if r["runtime_status"] != "OK":
            b["measured_error_count"] += 1
        elif r["baseline_correctness"] == "CORRECT":
            b["measured_pass_count"] += 1
        else:
            b["measured_fail_count"] += 1

    out: list[dict[str, str]] = []
    for (split_name, category), c in sorted(bucket.items()):
        p, f = c["measured_pass_count"], c["measured_fail_count"]
        ratio = f"{p / (p + f):.4f}" if (p + f) else "NA"
        out.append(
            {
                "split_name": split_name,
                "category": category,
                "sample_count": str(c["sample_count"]),
                "measured_pass_count": str(p),
                "measured_fail_count": str(f),
                "measured_error_count": str(c["measured_error_count"]),
                "format_failure_count": str(c["format_failure_count"]),
                "extraction_failure_count": str(c["extraction_failure_count"]),
                "pass_rate_or_accuracy": ratio,
            }
        )
    return out


def render_error_report(state: RunState, created_files: list[Path]) -> str:
    error_present = "YES" if state.runtime_result in {"FAILED", "BLOCKED"} else "NO"
    blocker = classify_blocker(state.exception_type, state.exception_message or state.baseline_stderr)

    expected_all = EXPECTED_REPORTS + EXPECTED_LOGS + EXPECTED_DATA
    missing_after = [str(p) for p in expected_all if not p.exists()]

    traceback_text = state.traceback_text if state.traceback_text else (state.baseline_stderr or "NO_TRACEBACK")

    created_section = [f"- `{p}`" for p in created_files] if created_files else ["- NONE"]
    missing_section = [f"- `{m}`" for m in missing_after] if missing_after else ["- NONE"]

    return "\n".join(
        [
            "# Error Report",
            "",
            f"- error_present: {error_present}",
            f"- exception type: `{state.exception_type or 'NO_ERROR'}`",
            f"- exception message: `{state.exception_message or 'NO_ERROR'}`",
            f"- failed stage: `{state.failed_stage or 'NONE'}`",
            f"- last successful stage: `{state.last_successful_stage}`",
            f"- likely blocker classification: `{blocker if error_present == 'YES' else 'UNKNOWN'}`",
            f"- exact repro command: `{state.baseline_command}`",
            "",
            "## Full traceback",
            "```text",
            traceback_text,
            "```",
            "",
            "## Files created before failure",
            *created_section,
            "",
            "## Files missing after failure",
            *missing_section,
            "",
        ]
    )


def render_output_artifacts() -> str:
    expected_all = EXPECTED_REPORTS + EXPECTED_LOGS + EXPECTED_DATA
    existing = sorted([p for p in expected_all if p.exists()])
    missing = sorted([p for p in expected_all if not p.exists()])

    lines = ["# Output Artifacts", "", "## Generated files"]
    for p in existing:
        size = p.stat().st_size if p.exists() and p.is_file() else 0
        lines.append(f"- path: `{p}` | exists: `True` | size: `{size}` | type: `expected`")

    lines.extend(["", "## Missing expected files"])
    if missing:
        lines.extend([f"- `{p}`" for p in missing])
    else:
        lines.append("- NONE")

    # Unexpected files (same dirs only)
    known = {str(p.resolve()) for p in expected_all}
    observed = list(REPORT_DIR.glob("*")) + list(LOG_DIR.glob("*"))
    unexpected = [p for p in observed if str(p.resolve()) not in known]
    lines.extend(["", "## Unexpected files"])
    if unexpected:
        lines.extend([f"- `{p}`" for p in sorted(unexpected)])
    else:
        lines.append("- NONE")
    lines.append("")
    return "\n".join(lines)


def render_run_summary(state: RunState) -> str:
    total_runtime_seconds = max(0.0, datetime.now(timezone.utc).timestamp() - state.start_epoch)
    return "\n".join(
        [
            "# Run Summary",
            "",
            "- goal: `Debug output hardening version v1`",
            f"- script executed: `{BASELINE_SCRIPT}`",
            f"- runtime result: `{state.runtime_result}`",
            f"- start time: `{state.start_time}`",
            f"- end time: `{state.end_time}`",
            f"- total runtime: `{total_runtime_seconds:.3f} sec`",
            f"- last successful stage: `{state.last_successful_stage}`",
            "- output summary: `See output_artifacts.md for exact files`",
            "- pointer to error_report.md: `reports/kaggle_run/error_report.md`",
            "- pointer to environment_info.md: `reports/kaggle_run/environment_info.md`",
            "",
        ]
    )


def render_repro_steps() -> str:
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return "\n".join(
        [
            "# Repro Steps",
            "",
            f"- 実行コマンド: `python scripts/eval/run_kaggle_measured_reference_v1.py`",
            f"- baseline 実行コマンド: `python {BASELINE_SCRIPT}`",
            "- 前提 path:",
            "  - /kaggle/input",
            "  - /kaggle/working",
            "  - /kaggle/temp",
            f"  - {BASELINE_SCRIPT}",
            "- 必要 dependency: torch, transformers, pandas, sympy, safetensors, kagglehub",
            "",
            "## Steps",
            "1. working branch と commit を確認する。",
            "2. frozen evaluation assets が改変されていないことを確認する。",
            "3. wrapper を実行する。",
            "4. reports/kaggle_run と logs/kaggle_run の自動生成物を確認する。",
            "5. data/eval/*_kaggle.csv の schema と件数を確認する。",
            "",
            f"- 実行時 branch: `{branch}`",
            f"- 実行時 commit: `{commit}`",
            "",
        ]
    )


def main() -> int:
    ensure_dirs()
    safe_write(LOG_DIR / "stdout.log", "")
    safe_write(LOG_DIR / "stderr.log", "")
    safe_write(LOG_DIR / "stage_log.jsonl", "")

    state = RunState(start_time=now_iso(), start_epoch=datetime.now(timezone.utc).timestamp())
    created_files: list[Path] = []

    try:
        stage_log("setup_start", "START", "run initialization")
        env_text = collect_environment_info()
        safe_write(REPORT_DIR / "environment_info.md", env_text)
        created_files.append(REPORT_DIR / "environment_info.md")
        stage_log("setup_end", "SUCCESS", "environment info captured")
        state.last_successful_stage = "setup_end"

        stage_log("input_check_start", "START", "input/path checks")
        manifest_text = build_input_manifest()
        safe_write(REPORT_DIR / "input_manifest.md", manifest_text)
        created_files.append(REPORT_DIR / "input_manifest.md")
        frozen, frozen_note = frozen_state()
        stage_log("input_check_end", "SUCCESS", "input/path checks completed", {"frozen_state": frozen})
        state.last_successful_stage = "input_check_end"

        if os.environ.get("FORCE_DEBUG_CRASH") == "1":
            raise RuntimeError("FORCE_DEBUG_CRASH=1 triggered synthetic failure for failure-path validation")

        runtime_status, runtime_note = run_baseline_and_capture(state)
        if runtime_status != "OK":
            state.exception_type = "BaselineProcessError"
            state.exception_message = runtime_note

        stage_log("eval_start", "START", "building measured rows")
        note = f"frozen_state={frozen}; frozen_note={frozen_note}; kaggle_attempt={runtime_note}"
        all_rows: list[dict[str, str]] = []
        for split_name, path in SPLITS.items():
            rows = build_rows(split_name, read_csv(path), runtime_status, note)
            out = DATA_EVAL / f"baseline_measured_results_{split_name}_kaggle.csv"
            write_csv(out, RESULT_COLUMNS, rows)
            all_rows.extend(rows)
            created_files.append(out)
        stage_log("eval_end", "SUCCESS", "per-split measured rows exported")
        state.last_successful_stage = "eval_end"

        stage_log("export_start", "START", "exporting summary and reports")
        summary = summarize(all_rows)
        summary_path = DATA_EVAL / "baseline_measured_category_summary_v1_kaggle.csv"
        write_csv(summary_path, SUMMARY_COLUMNS, summary)
        created_files.append(summary_path)
        stage_log("export_end", "SUCCESS", "summary export completed", {"summary_rows": len(summary)})
        state.last_successful_stage = "export_end"

        state.runtime_result = "SUCCESS" if runtime_status == "OK" else "BLOCKED"

    except Exception as e:  # noqa: BLE001
        state.runtime_result = "FAILED"
        state.exception_type = type(e).__name__
        state.exception_message = str(e)
        state.traceback_text = traceback.format_exc()
        state.failed_stage = state.last_successful_stage
        stage_log(
            "exception",
            "FAILED",
            "unhandled exception",
            {"exception_type": state.exception_type, "exception_message": state.exception_message},
        )
    finally:
        # Ensure mandatory stage markers always exist.
        if state.last_successful_stage == "setup_end":
            stage_log("model_load_end", "SKIPPED", "model load skipped")
            stage_log("eval_end", "SKIPPED", "eval skipped")
            stage_log("export_end", "SKIPPED", "export skipped")

        state.end_time = now_iso()
        error_md = render_error_report(state, created_files)
        safe_write(REPORT_DIR / "error_report.md", error_md)

        safe_write(REPORT_DIR / "output_artifacts.md", render_output_artifacts())
        safe_write(REPORT_DIR / "run_summary.md", render_run_summary(state))
        safe_write(REPORT_DIR / "repro_steps.md", render_repro_steps())

        stage_log("run_finish", "END", "run finished", {"runtime_result": state.runtime_result})

    return 0 if state.runtime_result in {"SUCCESS", "BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
