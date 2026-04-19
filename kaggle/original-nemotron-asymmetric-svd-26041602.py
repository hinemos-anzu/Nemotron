"""
Day2 Run1 target implementation for Kaggle notebook:
https://www.kaggle.com/code/hinemos/original-nemotron-asymmetric-svd-26041602

B1 scope only:
- training-serving misalignment fix for post-surgery adapter metadata/config alignment
- Day2 evidence collection
"""

import json
import os
import zipfile
from datetime import datetime, UTC
from pathlib import Path

from safetensors import safe_open


CANONICAL_TARGET_MODULES = {
    "k_proj",
    "o_proj",
    "in_proj",  # gate_proj/x_proj merged into in_proj after surgery
    "q_proj",
    "up_proj",
    "v_proj",
    "down_proj",
    "out_proj",
    "lm_head",
}


def _infer_target_modules_from_safetensors(adapter_model_path: Path) -> list[str]:
    """Infer target module names from post-surgery tensor keys."""
    module_names: set[str] = set()
    with safe_open(str(adapter_model_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if not key.endswith(".lora_A.weight"):
                continue
            base = key[: -len(".lora_A.weight")]
            module_name = base.split(".")[-1]
            module_names.add(module_name)

    # keep only serving-relevant module names and deterministic ordering
    aligned = sorted(m for m in module_names if m in CANONICAL_TARGET_MODULES)
    return aligned


def reconcile_serving_metadata(adapter_dir: str | Path) -> dict:
    """
    B1 core: align serving metadata/config with post-surgery adapter reality.

    Actions:
    1) read post-surgery adapter_model.safetensors
    2) infer modules actually present
    3) set adapter_config target_modules to inferred modules
    4) set inference_mode=True for serving
    5) write alignment record for auditability
    """
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    model_path = adapter_dir / "adapter_model.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    inferred_modules = _infer_target_modules_from_safetensors(model_path)
    previous_modules = config.get("target_modules", [])
    previous_inference_mode = config.get("inference_mode")

    config["target_modules"] = inferred_modules
    config["inference_mode"] = True

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    alignment = {
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "alignment_scope": "B1 training-serving misalignment",
        "adapter_dir": str(adapter_dir),
        "target_modules_before": previous_modules,
        "target_modules_after": inferred_modules,
        "inference_mode_before": previous_inference_mode,
        "inference_mode_after": True,
        "gate_x_merged_to_in_proj_assumed": True,
    }

    with open(adapter_dir / "serving_alignment.json", "w", encoding="utf-8") as f:
        json.dump(alignment, f, ensure_ascii=False, indent=2)

    return alignment


def collect_day2_evidence(
    baseline_sha: str,
    experiment_scope: str = "B1: training-serving misalignment 修正",
    one_variable_rule: bool = True,
    source_of_truth: str = "Kaggle",
    notes: list[str] | None = None,
) -> dict:
    """Collect Day2 evidence required by day2_codex_request.md."""
    notes = notes or []

    submission_zip_path = "/kaggle/working/submission.zip"
    submission_exists = os.path.exists(submission_zip_path)

    if submission_exists:
        size_bytes = os.path.getsize(submission_zip_path)
        with zipfile.ZipFile(submission_zip_path, "r") as zf:
            file_list = zf.namelist()
    else:
        size_bytes = None
        file_list = []

    evidence = {
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_of_truth": source_of_truth,
        "baseline_sha": baseline_sha,
        "experiment_scope": experiment_scope,
        "one_variable_rule": one_variable_rule,
        "submission_zip": {
            "path": submission_zip_path,
            "exists": submission_exists,
            "status": "PASS" if submission_exists else "FAIL",
            "size_bytes": size_bytes,
            "file_count": len(file_list),
            "file_list": file_list,
        },
        "submission_assets_preserved": submission_exists and size_bytes not in (None, 0) and len(file_list) > 0,
        "comparable_against_baseline": False,
        "worse_than_baseline": "UNCONFIRMED",
        "evidence_for_gt_086": "UNCONFIRMED",
        "provisional_verdict": "HOLD",
        "notes": notes,
    }

    out_json = "/kaggle/working/day2_evidence.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    out_md = "/kaggle/working/day2_evidence.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Day2 Evidence Report\n\n")
        for k in [
            "timestamp_utc",
            "source_of_truth",
            "baseline_sha",
            "experiment_scope",
            "one_variable_rule",
            "provisional_verdict",
            "comparable_against_baseline",
            "worse_than_baseline",
            "evidence_for_gt_086",
        ]:
            f.write(f"- {k}: {evidence[k]}\n")
        f.write("\n## submission_zip\n")
        f.write(f"- path: {evidence['submission_zip']['path']}\n")
        f.write(f"- exists: {evidence['submission_zip']['exists']}\n")
        f.write(f"- status: {evidence['submission_zip']['status']}\n")
        f.write(f"- size_bytes: {evidence['submission_zip']['size_bytes']}\n")
        f.write(f"- file_count: {evidence['submission_zip']['file_count']}\n")
        f.write("- file_list:\n")
        for item in evidence["submission_zip"]["file_list"]:
            f.write(f"  - {item}\n")

    return evidence


if __name__ == "__main__":
    adapter_dir = os.environ.get("WORKING_ADAPTER_DIR", "/kaggle/working/adapter")
    alignment = reconcile_serving_metadata(adapter_dir)
    collect_day2_evidence(
        baseline_sha="39f4bed90392567517b606d1301ae1c36a86a97c",
        notes=[
            "B1 serving alignment applied.",
            f"Aligned target_modules={alignment['target_modules_after']}",
            "Populate comparable_against_baseline/worse_than_baseline/evidence_for_gt_086 after Kaggle scoring logs are available.",
        ],
    )
