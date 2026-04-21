"""
kaggle/original-nemotron-asymmetric-svd-26041602.py

Baseline execution script: original Nemotron asymmetric-SVD submission path.
This is the Kaggle source-of-truth execution path referenced by:
  - TICKET_S1_6A_kaggle_runtime_execution_enablement_for_baseline_path.md
  - docs/specs/design_spec_from_research_v1.md (protected baseline assets)

PROTECTED ASSET — do not change strategy logic, conversion flow, SVD surgery,
key rename / merge logic, expert unfuse logic, or submission.zip generation.

PLACEHOLDER STATUS:
  This file is a structural placeholder.  The Kaggle execution role must replace
  the body below with the actual notebook/script content before execution.
  The interface contract (REQUIRED_INPUTS, EXPECTED_ARTIFACTS, stage log calls)
  must be preserved when the real implementation is inserted.

REQUIRED ENVIRONMENT:
  - Python 3.10+
  - torch, transformers, peft, accelerate, safetensors
  - /kaggle/input/  (mounted dataset root)
  - /kaggle/working/ (output root)
  - Adapter model weights mounted at /kaggle/input/nemotron-adapter/ (see ADAPTER_INPUTS)
"""

import os
import sys
import json
import traceback
from pathlib import Path

# ─── interface contract ───────────────────────────────────────────────────────

REQUIRED_INPUTS = [
    # Standard Kaggle environment roots — always present
    "/kaggle/input",
    "/kaggle/working",
]

# Model / adapter assets required when real body is inserted.
# Paths confirmed with Kaggle execution role during TICKET_S1_6A.
# Checked by the real implementation, NOT by this placeholder.
ADAPTER_INPUTS = [
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

# ─── optional: attach to active harness stage logger ─────────────────────────
# If run via run_baseline_with_debug.py, a StageLogger is available.

def _get_stage_logger():
    """Return active harness stage logger or a no-op logger."""
    try:
        # Try to load the stage log path set by the harness
        stage_log_path = os.environ.get("KAGGLE_HARNESS_STAGE_LOG")
        if stage_log_path:
            sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
            from debug_harness import StageLogger, _now
            import json as _json

            class _DirectLogger:
                def __init__(self, path):
                    self._path = path

                def log(self, stage, message="", status="ok", extra=None):
                    event = {
                        "timestamp": _now(),
                        "stage": stage,
                        "status": status,
                        "message": message,
                        "extra": extra or {},
                    }
                    try:
                        with open(self._path, "a", encoding="utf-8") as f:
                            f.write(_json.dumps(event) + "\n")
                    except Exception:
                        pass

            return _DirectLogger(stage_log_path)
    except Exception:
        pass

    class _NoopLogger:
        def log(self, *a, **kw): pass
    return _NoopLogger()


# ─── main execution ───────────────────────────────────────────────────────────

def main():
    sl = _get_stage_logger()
    working_dir = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
    working_dir.mkdir(parents=True, exist_ok=True)

    # ── model load ────────────────────────────────────────────────────
    sl.log("model_load_start", "Checking model / adapter assets")
    missing = [p for p in REQUIRED_INPUTS if not Path(p).exists()]
    if missing:
        msg = f"Required inputs missing: {missing}"
        sl.log("model_load_end", msg, status="error",
               extra={"missing": missing})
        raise FileNotFoundError(
            f"{msg}\n"
            "This script requires real Kaggle dataset mounts.  "
            "Run via Kaggle notebooks with the correct input datasets attached.\n"
            "See TICKET_S1_6A for environment enablement steps."
        )
    sl.log("model_load_end", "All required inputs present")

    # ── evaluation ────────────────────────────────────────────────────
    sl.log("eval_start", "Starting evaluation (placeholder body)")

    # ══════════════════════════════════════════════════════════════════
    # PLACEHOLDER: insert actual baseline execution logic here.
    # The real implementation must:
    #   1. Load the adapter-converted model weights.
    #   2. Run inference on the frozen evaluation sets.
    #   3. Write per-sample predictions to /kaggle/working/predictions.jsonl
    #   4. Package submission.zip per the original submission asset flow.
    #   5. Write /kaggle/working/run_complete.flag on success.
    # Do NOT modify conversion flow, SVD surgery, key rename, or merge logic.
    # ══════════════════════════════════════════════════════════════════
    raise NotImplementedError(
        "Baseline script body not yet implemented.  "
        "Kaggle execution role must insert the real notebook content here.  "
        "See TICKET_S1_6A_kaggle_runtime_execution_enablement_for_baseline_path.md"
    )

    sl.log("eval_end", "Evaluation complete")

    # ── export ────────────────────────────────────────────────────────
    sl.log("export_start", "Writing output artifacts")
    (working_dir / "run_complete.flag").write_text("OK\n")
    sl.log("export_end", "Artifacts written",
           extra={"artifacts": EXPECTED_ARTIFACTS})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[baseline] FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
