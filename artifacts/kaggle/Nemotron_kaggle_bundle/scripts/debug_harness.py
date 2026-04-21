"""
scripts/debug_harness.py

Debug output harness for Kaggle execution.
Automatically collects environment info, stage logs, error reports, and artifact
manifests so Kaggle execution evidence can be reviewed by Generator without
manual observation.

Usage — external wrapper (primary):
    from debug_harness import KaggleRunHarness
    harness = KaggleRunHarness(
        script_path="kaggle/original-nemotron-asymmetric-svd-26041602.py",
        run_output_dir="reports/kaggle_run",
        log_output_dir="logs/kaggle_run",
    )
    harness.run()

Usage — internal stage logging (optional, from inside baseline script):
    from debug_harness import StageLogger
    sl = StageLogger.get_active()
    sl.log("model_load_start", "Loading adapter weights...")
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_import(module: str, attr: str = "__version__") -> str:
    """Return version string or NOT_INSTALLED."""
    try:
        import importlib
        m = importlib.import_module(module)
        return str(getattr(m, attr, "unknown"))
    except Exception:
        return "NOT_INSTALLED"


def _file_info(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if p.is_file():
        size = p.stat().st_size
        return {"exists": True, "type": "file", "size_bytes": size,
                "size_human": _human_size(size)}
    if p.is_dir():
        try:
            entries = list(p.iterdir())
            size = sum(f.stat().st_size for f in entries if f.is_file())
            return {"exists": True, "type": "dir", "file_count": len(entries),
                    "size_bytes": size, "size_human": _human_size(size),
                    "listing": [e.name for e in sorted(entries)[:20]]}
        except PermissionError:
            return {"exists": True, "type": "dir", "error": "permission_denied"}
    return {"exists": False}


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ─── StageLogger ──────────────────────────────────────────────────────────────

class StageLogger:
    """Writes NDJSON stage events to logs/kaggle_run/stage_log.jsonl."""

    _active: Optional["StageLogger"] = None

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._last_ok_stage: str = "none"
        # truncate on init
        self._path.write_text("", encoding="utf-8")

    # Allow baseline scripts to import and attach to the running instance
    @classmethod
    def get_active(cls) -> "StageLogger":
        if cls._active is None:
            raise RuntimeError("No active StageLogger. Run via KaggleRunHarness.")
        return cls._active

    def log(self, stage: str, message: str = "",
            status: str = "ok", extra: Optional[Dict] = None) -> None:
        event = {
            "timestamp": _now(),
            "stage": stage,
            "status": status,
            "message": message,
            "extra": extra or {},
        }
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass  # never let logger crash main logic
        if status not in ("error", "exception"):
            self._last_ok_stage = stage

    def log_exception(self, stage: str, exc: BaseException) -> None:
        tb = traceback.format_exc()
        self.log(stage, str(exc), status="exception",
                 extra={"traceback": tb, "type": type(exc).__name__})

    @property
    def last_ok_stage(self) -> str:
        return self._last_ok_stage


# ─── EnvironmentDumper ────────────────────────────────────────────────────────

class EnvironmentDumper:
    """Collects environment information and writes environment_info.md."""

    REQUIRED_PATHS = [
        "/kaggle/input",
        "/kaggle/working",
        # /kaggle/temp is NOT a standard Kaggle directory — excluded
    ]

    def __init__(self, script_path: str | Path, output_path: Path) -> None:
        self._script_path = Path(script_path)
        self._output_path = output_path

    def collect_and_write(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        now = _now()

        # ── timing & identity ──────────────────────────────────────────
        info["timestamp"] = now
        info["git_branch"] = self._git("rev-parse --abbrev-ref HEAD")
        info["git_commit"] = self._git("rev-parse HEAD")

        # ── runtime ────────────────────────────────────────────────────
        info["python_version"] = sys.version
        info["python_executable"] = sys.executable
        info["platform"] = platform.platform()
        info["os_name"] = os.name
        info["cwd"] = str(Path.cwd())

        # ── GPU ────────────────────────────────────────────────────────
        cuda_info = self._cuda_info()
        info.update(cuda_info)

        # ── dependency versions ────────────────────────────────────────
        info["torch_version"] = _safe_import("torch")
        info["transformers_version"] = _safe_import("transformers")
        info["peft_version"] = _safe_import("peft")
        info["accelerate_version"] = _safe_import("accelerate")
        info["numpy_version"] = _safe_import("numpy")
        info["pandas_version"] = _safe_import("pandas")
        info["tokenizers_version"] = _safe_import("tokenizers")
        info["bitsandbytes_version"] = _safe_import("bitsandbytes")
        info["safetensors_version"] = _safe_import("safetensors")

        # ── environment variables ──────────────────────────────────────
        safe_env_keys = [
            "CUDA_VISIBLE_DEVICES", "TRANSFORMERS_CACHE", "HF_HOME",
            "KAGGLE_DATA_PROXY_TOKEN", "KAGGLE_KERNEL_RUN_TYPE",
            "PYTHONPATH", "HOME", "PATH",
        ]
        info["env_vars"] = {k: os.environ.get(k, "NOT_SET") for k in safe_env_keys}

        # ── filesystem presence ────────────────────────────────────────
        fs: Dict[str, Any] = {}
        for p in self.REQUIRED_PATHS:
            fs[p] = _file_info(p)
        fs[str(self._script_path)] = _file_info(self._script_path)
        info["filesystem_checks"] = fs

        self._write_md(info)
        return info

    def _git(self, cmd: str) -> str:
        try:
            return subprocess.check_output(
                ["git"] + cmd.split(), stderr=subprocess.DEVNULL,
                text=True).strip()
        except Exception:
            return "UNAVAILABLE"

    def _cuda_info(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"cuda_available": False, "gpu_name": "NONE",
                                   "gpu_count": 0, "cuda_version": "N/A"}
        try:
            import torch
            result["cuda_available"] = torch.cuda.is_available()
            result["gpu_count"] = torch.cuda.device_count()
            result["cuda_version"] = torch.version.cuda or "N/A"
            if torch.cuda.is_available():
                result["gpu_name"] = torch.cuda.get_device_name(0)
                result["gpu_memory_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        except Exception as e:
            result["cuda_check_error"] = str(e)
        return result

    def _write_md(self, info: Dict) -> None:
        fs = info["filesystem_checks"]

        def _fs_row(k, v):
            ex = "✓" if v.get("exists") else "✗"
            detail = ""
            if v.get("exists"):
                if v["type"] == "file":
                    detail = v["size_human"]
                elif v["type"] == "dir":
                    detail = f"{v.get('file_count','?')} files, {v.get('size_human','?')}"
            else:
                detail = "MISSING"
            return f"| `{k}` | {ex} | {detail} |"

        lines = [
            "# Environment Info",
            f"**Generated:** {info['timestamp']}",
            "",
            "## Identity",
            f"- Git branch: `{info['git_branch']}`",
            f"- Git commit: `{info['git_commit']}`",
            "",
            "## Runtime",
            f"- Python: `{info['python_version'].split()[0]}`",
            f"- Executable: `{info['python_executable']}`",
            f"- Platform: `{info['platform']}`",
            f"- CWD: `{info['cwd']}`",
            "",
            "## GPU / CUDA",
            f"- CUDA available: `{info['cuda_available']}`",
            f"- GPU count: `{info['gpu_count']}`",
            f"- GPU name: `{info['gpu_name']}`",
            f"- CUDA version: `{info['cuda_version']}`",
            f"- GPU memory (GB): `{info.get('gpu_memory_gb', 'N/A')}`",
            "",
            "## Dependency Versions",
            "| Package | Version |",
            "|---|---|",
            f"| torch | {info['torch_version']} |",
            f"| transformers | {info['transformers_version']} |",
            f"| peft | {info['peft_version']} |",
            f"| accelerate | {info['accelerate_version']} |",
            f"| numpy | {info['numpy_version']} |",
            f"| pandas | {info['pandas_version']} |",
            f"| tokenizers | {info['tokenizers_version']} |",
            f"| bitsandbytes | {info['bitsandbytes_version']} |",
            f"| safetensors | {info['safetensors_version']} |",
            "",
            "## Environment Variables",
            "| Key | Value |",
            "|---|---|",
        ]
        for k, v in info["env_vars"].items():
            display = v if len(v) < 80 else v[:77] + "..."
            lines.append(f"| {k} | `{display}` |")

        lines += [
            "",
            "## Filesystem Checks",
            "| Path | Exists | Detail |",
            "|---|---|---|",
        ]
        for k, v in fs.items():
            lines.append(_fs_row(k, v))

        _write(self._output_path, "\n".join(lines) + "\n")


# ─── InputManifestWriter ──────────────────────────────────────────────────────

class InputManifestWriter:
    """Checks required input paths and writes input_manifest.md."""

    def __init__(self, required_inputs: List[str | Path], output_path: Path) -> None:
        self._inputs = [Path(p) for p in required_inputs]
        self._output_path = output_path

    def collect_and_write(self) -> bool:
        """Returns True if all required inputs are present."""
        rows = []
        all_present = True
        for p in self._inputs:
            fi = _file_info(p)
            present = fi.get("exists", False)
            if not present:
                all_present = False
            rows.append((str(p), fi))

        lines = [
            "# Input Manifest",
            f"**Generated:** {_now()}",
            "",
            "## Required Inputs",
            "| Path | Status | Type | Size / Count | Head / Listing |",
            "|---|---|---|---|---|",
        ]
        for path_str, fi in rows:
            status = "✓ PRESENT" if fi.get("exists") else "✗ MISSING"
            typ = fi.get("type", "—")
            size = fi.get("size_human", "—") if fi.get("type") == "file" else \
                   f"{fi.get('file_count','?')} files" if fi.get("type") == "dir" else "—"
            head = ""
            if fi.get("type") == "dir" and fi.get("listing"):
                head = ", ".join(fi["listing"][:8])
                if len(fi.get("listing", [])) > 8:
                    head += ", ..."
            elif fi.get("type") == "file":
                # try to show first line for text files
                try:
                    with open(path_str, encoding="utf-8", errors="replace") as f:
                        head = f.readline().strip()[:60]
                except Exception:
                    head = "(binary)"
            lines.append(f"| `{path_str}` | {status} | {typ} | {size} | {head} |")

        if not all_present:
            missing = [str(p) for p, fi in zip(self._inputs,
                        [r[1] for r in rows]) if not fi.get("exists")]
            lines += ["", "## Missing Inputs", ""]
            for m in missing:
                lines.append(f"- `{m}`")

        _write(self._output_path, "\n".join(lines) + "\n")
        return all_present


# ─── OutputArtifactTracker ────────────────────────────────────────────────────

class OutputArtifactTracker:
    """Checks expected output artifacts and writes output_artifacts.md."""

    def __init__(self, expected_artifacts: List[str | Path], output_path: Path) -> None:
        self._expected = [Path(p) for p in expected_artifacts]
        self._unexpected: List[Path] = []
        self._output_path = output_path

    def add_unexpected(self, path: str | Path) -> None:
        self._unexpected.append(Path(path))

    def collect_and_write(self, extra_scan_dirs: Optional[List[Path]] = None) -> None:
        lines = [
            "# Output Artifacts",
            f"**Generated:** {_now()}",
            "",
            "## Expected Artifacts",
            "| Path | Status | Size |",
            "|---|---|---|",
        ]
        missing = []
        for p in self._expected:
            fi = _file_info(p)
            if fi.get("exists"):
                lines.append(f"| `{p}` | ✓ PRESENT | {fi['size_human']} |")
            else:
                lines.append(f"| `{p}` | ✗ MISSING | — |")
                missing.append(str(p))

        if self._unexpected:
            lines += ["", "## Unexpected Artifacts (present but not in expected list)",
                      "| Path | Size |", "|---|---|"]
            for p in self._unexpected:
                fi = _file_info(p)
                lines.append(f"| `{p}` | {fi.get('size_human','?')} |")

        if extra_scan_dirs:
            for d in extra_scan_dirs:
                if d.is_dir():
                    found = sorted(d.rglob("*"))
                    if found:
                        lines += [f"", f"## Scan of `{d}`",
                                  "| Path | Size |", "|---|---|"]
                        for f in found:
                            if f.is_file():
                                fi = _file_info(f)
                                lines.append(f"| `{f}` | {fi.get('size_human','?')} |")

        if missing:
            lines += ["", "## Missing Expected Artifacts", ""]
            for m in missing:
                lines.append(f"- `{m}`")

        _write(self._output_path, "\n".join(lines) + "\n")


# ─── ErrorReporter ────────────────────────────────────────────────────────────

class ErrorReporter:
    """Writes error_report.md. Always written: NO_ERROR if no exception."""

    BLOCKER_RULES = {
        "ModuleNotFoundError": "IMPORT_ERROR",
        "ImportError": "IMPORT_ERROR",
        "FileNotFoundError": "FILE_NOT_FOUND",
        "OSError": "FILE_NOT_FOUND",
        "RuntimeError": "RUNTIME_CRASH",
    }

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    def write_no_error(self) -> None:
        content = textwrap.dedent(f"""\
            # Error Report
            **Generated:** {_now()}

            ## error_present
            NO_ERROR

            All stages completed without exception.
        """)
        _write(self._output_path, content)

    def write_error(
        self,
        exc: BaseException,
        failed_stage: str,
        last_ok_stage: str,
        repro_cmd: str,
        files_created: List[str],
        files_missing: List[str],
    ) -> None:
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tb_str = "".join(tb)
        exc_type = type(exc).__name__
        blocker = self.BLOCKER_RULES.get(exc_type, "RUNTIME_CRASH")
        # Refine: if 'No such file' in message, use FILE_NOT_FOUND
        if "No such file" in str(exc) or "not found" in str(exc).lower():
            blocker = "FILE_NOT_FOUND"
        elif "CUDA" in str(exc) or "cuda" in str(exc) or "GPU" in str(exc):
            blocker = "ENVIRONMENT_MISSING"
        elif "import" in str(exc).lower() or "module" in str(exc).lower():
            blocker = "IMPORT_ERROR"

        created_block = "\n".join(f"- `{f}`" for f in files_created) or "- (none)"
        missing_block = "\n".join(f"- `{f}`" for f in files_missing) or "- (none)"

        content = textwrap.dedent(f"""\
            # Error Report
            **Generated:** {_now()}

            ## error_present
            YES

            ## Exception Type
            `{exc_type}`

            ## Exception Message
            ```
            {str(exc)}
            ```

            ## Full Traceback
            ```
            {tb_str}
            ```

            ## Failed Stage
            `{failed_stage}`

            ## Last Successful Stage
            `{last_ok_stage}`

            ## Likely Blocker Classification
            `{blocker}`

            Classification choices:
            - ENVIRONMENT_MISSING — CUDA/GPU/driver not present
            - IMPORT_ERROR — required package not installed
            - DEPENDENCY_MISSING — required file or model weight absent
            - FILE_NOT_FOUND — required path does not exist
            - RUNTIME_CRASH — exception during execution
            - UNKNOWN — cannot be classified automatically

            ## Files Created Before Failure
            {created_block}

            ## Files Missing After Failure
            {missing_block}

            ## Exact Repro Command
            ```
            {repro_cmd}
            ```
        """)
        _write(self._output_path, content)

    def write_subprocess_error(
        self,
        returncode: int,
        failed_stage: str,
        last_ok_stage: str,
        repro_cmd: str,
        stderr_tail: str,
        files_created: List[str],
        files_missing: List[str],
    ) -> None:
        # Classify from stderr
        blocker = "RUNTIME_CRASH"
        if "No module named" in stderr_tail or "ImportError" in stderr_tail:
            blocker = "IMPORT_ERROR"
        elif "No such file" in stderr_tail or "FileNotFoundError" in stderr_tail:
            blocker = "FILE_NOT_FOUND"
        elif "CUDA" in stderr_tail or "out of memory" in stderr_tail.lower():
            blocker = "ENVIRONMENT_MISSING"

        created_block = "\n".join(f"- `{f}`" for f in files_created) or "- (none)"
        missing_block = "\n".join(f"- `{f}`" for f in files_missing) or "- (none)"

        content = textwrap.dedent(f"""\
            # Error Report
            **Generated:** {_now()}

            ## error_present
            YES

            ## Exception Type
            `SubprocessNonZeroExit`

            ## Exception Message
            ```
            Subprocess returned exit code {returncode}
            ```

            ## Stderr Tail (last 50 lines)
            ```
            {stderr_tail}
            ```

            ## Full Traceback
            See `logs/kaggle_run/stderr.log` for full stderr output.

            ## Failed Stage
            `{failed_stage}`

            ## Last Successful Stage
            `{last_ok_stage}`

            ## Likely Blocker Classification
            `{blocker}`

            ## Files Created Before Failure
            {created_block}

            ## Files Missing After Failure
            {missing_block}

            ## Exact Repro Command
            ```
            {repro_cmd}
            ```
        """)
        _write(self._output_path, content)


# ─── ReproStepsWriter ─────────────────────────────────────────────────────────

class ReproStepsWriter:
    """Writes repro_steps.md."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    def write(
        self,
        script_path: str | Path,
        repro_cmd: str,
        branch: str,
        commit: str,
        required_paths: List[str],
        dependencies: List[str],
    ) -> None:
        req_block = "\n".join(f"- `{p}`" for p in required_paths)
        dep_block = "\n".join(f"- `{d}`" for d in dependencies)
        content = textwrap.dedent(f"""\
            # Repro Steps
            **Generated:** {_now()}

            ## Script Executed
            `{script_path}`

            ## Branch / Commit
            - Branch: `{branch}`
            - Commit: `{commit}`

            ## Required Paths (must exist before run)
            {req_block}

            ## Dependencies
            {dep_block}

            ## Exact Repro Command
            ```
            {repro_cmd}
            ```

            ## Reproduction Procedure
            1. Check out branch `{branch}` at commit `{commit}`.
            2. Confirm all required paths listed above exist.
            3. Install dependencies: `pip install {' '.join(dependencies[:5])} ...`
            4. Run the entry point:
               ```
               {repro_cmd}
               ```
            5. Outputs are written to `reports/kaggle_run/` and `logs/kaggle_run/`.
            6. Check `reports/kaggle_run/run_summary.md` first for overall status.
            7. If failed: see `reports/kaggle_run/error_report.md` for classified failure.
        """)
        _write(self._output_path, content)


# ─── RunSummaryWriter ─────────────────────────────────────────────────────────

class RunSummaryWriter:
    """Writes run_summary.md."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    def write(
        self,
        script_path: str | Path,
        result: str,
        start_time: str,
        end_time: str,
        last_ok_stage: str,
        output_summary: str,
        run_output_dir: Path,
    ) -> None:
        try:
            t0 = datetime.fromisoformat(start_time)
            t1 = datetime.fromisoformat(end_time)
            duration = str(t1 - t0)
        except Exception:
            duration = "UNKNOWN"

        content = textwrap.dedent(f"""\
            # Run Summary
            **Generated:** {_now()}

            ## Goal
            Execute the baseline Kaggle path with automatic debug output collection,
            so failures can be diagnosed by Generator without manual Kaggle observation.

            ## Script Executed
            `{script_path}`

            ## Runtime Result
            **{result}**

            ## Timing
            - Start time: `{start_time}`
            - End time: `{end_time}`
            - Total runtime: `{duration}`

            ## Last Successful Stage
            `{last_ok_stage}`

            ## Output Summary
            {output_summary}

            ## Pointers
            - Error report: `{run_output_dir}/error_report.md`
            - Environment info: `{run_output_dir}/environment_info.md`
            - Input manifest: `{run_output_dir}/input_manifest.md`
            - Output artifacts: `{run_output_dir}/output_artifacts.md`
            - Repro steps: `{run_output_dir}/repro_steps.md`
            - Stage log: `logs/kaggle_run/stage_log.jsonl`
            - Stdout: `logs/kaggle_run/stdout.log`
            - Stderr: `logs/kaggle_run/stderr.log`
        """)
        _write(self._output_path, content)


# ─── KaggleRunHarness ─────────────────────────────────────────────────────────

class KaggleRunHarness:
    """
    Top-level coordinator.  Wraps a target script execution with full debug
    output collection.  Writes all 9 required files regardless of success/failure.

    All file writes use try/except so a logger failure never masks the real error.
    """

    DEFAULT_EXPECTED_ARTIFACTS: List[str] = []

    REQUIRED_PACKAGES = [
        "torch", "transformers", "peft", "accelerate",
        "numpy", "pandas", "safetensors",
    ]

    def __init__(
        self,
        script_path: str | Path,
        run_output_dir: str | Path = "reports/kaggle_run",
        log_output_dir: str | Path = "logs/kaggle_run",
        required_inputs: Optional[List[str | Path]] = None,
        expected_artifacts: Optional[List[str | Path]] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.script_path = Path(script_path)
        self.run_output_dir = Path(run_output_dir)
        self.log_output_dir = Path(log_output_dir)
        self.required_inputs = required_inputs or []
        self.expected_artifacts = list(expected_artifacts or self.DEFAULT_EXPECTED_ARTIFACTS)
        self.extra_env = extra_env or {}
        self._start_time: str = ""

    def run(self) -> int:
        """
        Execute the target script.  Return exit code (0 = success).
        Always writes all debug files.
        """
        self._start_time = _now()
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        self.log_output_dir.mkdir(parents=True, exist_ok=True)

        stage_log = StageLogger(self.log_output_dir / "stage_log.jsonl")
        StageLogger._active = stage_log

        stdout_path = self.log_output_dir / "stdout.log"
        stderr_path = self.log_output_dir / "stderr.log"

        env_dumper = EnvironmentDumper(
            self.script_path,
            self.run_output_dir / "environment_info.md",
        )
        input_checker = InputManifestWriter(
            self.required_inputs,
            self.run_output_dir / "input_manifest.md",
        )
        artifact_tracker = OutputArtifactTracker(
            self.expected_artifacts,
            self.run_output_dir / "output_artifacts.md",
        )
        error_reporter = ErrorReporter(self.run_output_dir / "error_report.md")
        repro_writer = ReproStepsWriter(self.run_output_dir / "repro_steps.md")
        summary_writer = RunSummaryWriter(self.run_output_dir / "run_summary.md")

        repro_cmd = f"python {self.script_path}"
        result = "UNKNOWN"
        last_ok_stage = "none"
        returncode = 1

        # ── phase 1: environment dump (always attempt) ────────────────
        try:
            stage_log.log("setup_start", "Collecting environment info")
            env_info = env_dumper.collect_and_write()
            stage_log.log("setup_end", "Environment info written",
                          extra={"cuda": env_info.get("cuda_available"),
                                 "gpu": env_info.get("gpu_name")})
            branch = env_info.get("git_branch", "unknown")
            commit = env_info.get("git_commit", "unknown")
        except Exception as e:
            stage_log.log_exception("setup_end", e)
            branch = commit = "unknown"

        # ── phase 2: write repro steps early (does not depend on run) ─
        try:
            repro_writer.write(
                script_path=self.script_path,
                repro_cmd=repro_cmd,
                branch=branch,
                commit=commit,
                required_paths=[str(p) for p in self.required_inputs],
                dependencies=self.REQUIRED_PACKAGES,
            )
        except Exception:
            pass

        # ── phase 3: input manifest ────────────────────────────────────
        try:
            stage_log.log("input_check_start", "Checking required inputs")
            inputs_ok = input_checker.collect_and_write()
            stage_log.log("input_check_end",
                          "Input check complete",
                          status="ok" if inputs_ok else "warn",
                          extra={"all_present": inputs_ok})
        except Exception as e:
            stage_log.log_exception("input_check_end", e)
            inputs_ok = False

        # ── phase 4: execute baseline script ──────────────────────────
        try:
            stage_log.log("eval_start", f"Launching {self.script_path}")
            env = {**os.environ, **self.extra_env}
            # Set stage log path so the baseline script can optionally call
            # StageLogger.get_active() after importing this module.
            env["KAGGLE_HARNESS_STAGE_LOG"] = str(stage_log._path.resolve())

            with open(stdout_path, "w", encoding="utf-8") as fout, \
                 open(stderr_path, "w", encoding="utf-8") as ferr:
                proc = subprocess.Popen(
                    [sys.executable, str(self.script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                # Tee: write to file AND print to console
                stdout_lines, stderr_lines = [], []
                import threading

                def _tee(stream, lines_list, file_handle, console_stream):
                    for line in stream:
                        lines_list.append(line)
                        file_handle.write(line)
                        file_handle.flush()
                        console_stream.write(line)
                        console_stream.flush()

                t_out = threading.Thread(target=_tee,
                    args=(proc.stdout, stdout_lines, fout, sys.stdout))
                t_err = threading.Thread(target=_tee,
                    args=(proc.stderr, stderr_lines, ferr, sys.stderr))
                t_out.start(); t_err.start()
                t_out.join(); t_err.join()
                returncode = proc.wait()

            last_ok_stage = stage_log.last_ok_stage

            if returncode == 0:
                stage_log.log("eval_end", "Script completed successfully",
                              extra={"returncode": returncode})
                result = "SUCCESS"
            else:
                stderr_tail = "".join(stderr_lines[-50:])
                stage_log.log("eval_end", f"Script exited with code {returncode}",
                              status="error", extra={"returncode": returncode})
                result = "FAILED"
                created = [str(p) for p in self.expected_artifacts if Path(p).exists()]
                missing = [str(p) for p in self.expected_artifacts if not Path(p).exists()]
                error_reporter.write_subprocess_error(
                    returncode=returncode,
                    failed_stage="eval_end",
                    last_ok_stage=stage_log.last_ok_stage,
                    repro_cmd=repro_cmd,
                    stderr_tail=stderr_tail,
                    files_created=created,
                    files_missing=missing,
                )

        except Exception as exc:
            result = "FAILED"
            stage_log.log_exception("eval_end", exc)
            try:
                created = [str(p) for p in self.expected_artifacts if Path(p).exists()]
                missing = [str(p) for p in self.expected_artifacts if not Path(p).exists()]
                error_reporter.write_error(
                    exc=exc,
                    failed_stage="eval_end",
                    last_ok_stage=stage_log.last_ok_stage,
                    repro_cmd=repro_cmd,
                    files_created=created,
                    files_missing=missing,
                )
            except Exception:
                pass

        if result == "SUCCESS":
            try:
                error_reporter.write_no_error()
            except Exception:
                pass

        # ── phase 5: artifact manifest ─────────────────────────────────
        try:
            stage_log.log("export_start", "Writing artifact manifest")
            artifact_tracker.collect_and_write(
                extra_scan_dirs=[
                    Path("reports/kaggle_run"),
                    Path("logs/kaggle_run"),
                ]
            )
            stage_log.log("export_end", "Artifact manifest written")
        except Exception as e:
            stage_log.log_exception("export_end", e)

        # ── phase 6: run summary ───────────────────────────────────────
        end_time = _now()
        output_summary_lines = [
            f"- Script: `{self.script_path}`",
            f"- Result: **{result}**",
            f"- Stdout lines: {len(stdout_lines) if 'stdout_lines' in dir() else 'N/A'}",
            f"- Stderr lines: {len(stderr_lines) if 'stderr_lines' in dir() else 'N/A'}",
        ]
        for p in self.expected_artifacts:
            fi = _file_info(p)
            status = "✓" if fi.get("exists") else "✗"
            output_summary_lines.append(f"- `{p}`: {status}")

        try:
            summary_writer.write(
                script_path=self.script_path,
                result=result,
                start_time=self._start_time,
                end_time=end_time,
                last_ok_stage=stage_log.last_ok_stage,
                output_summary="\n".join(output_summary_lines),
                run_output_dir=self.run_output_dir,
            )
        except Exception:
            pass

        stage_log.log("run_finish", f"Harness complete. Result: {result}",
                      status="ok" if result == "SUCCESS" else "error",
                      extra={"result": result, "returncode": returncode})

        StageLogger._active = None
        print(f"\n[debug_harness] Run complete: {result}. "
              f"Reports in {self.run_output_dir}", flush=True)
        return returncode
