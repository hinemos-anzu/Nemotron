"""
Day2 Run1 target implementation for Kaggle notebook:
https://www.kaggle.com/code/hinemos/original-nemotron-asymmetric-svd-26041602

This file contains the Day2 evidence collection block to be appended at the end
of the notebook/script execution flow.
"""

import json
import os
import zipfile
from datetime import datetime, UTC


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
        f.write(f"- timestamp_utc: {evidence['timestamp_utc']}\n")
        f.write(f"- source_of_truth: {evidence['source_of_truth']}\n")
        f.write(f"- baseline_sha: {evidence['baseline_sha']}\n")
        f.write(f"- experiment_scope: {evidence['experiment_scope']}\n")
        f.write(f"- one_variable_rule: {evidence['one_variable_rule']}\n")
        f.write(f"- provisional_verdict: {evidence['provisional_verdict']}\n")
        f.write(f"- comparable_against_baseline: {evidence['comparable_against_baseline']}\n")
        f.write(f"- worse_than_baseline: {evidence['worse_than_baseline']}\n")
        f.write(f"- evidence_for_gt_086: {evidence['evidence_for_gt_086']}\n\n")
        f.write("## submission_zip\n")
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
    collect_day2_evidence(
        baseline_sha="39f4bed90392567517b606d1301ae1c36a86a97c",
        notes=[
            "Day2 Run1 flow-check implementation target.",
            "Populate comparable_against_baseline/worse_than_baseline/evidence_for_gt_086 after Kaggle scoring logs are available.",
        ],
    )
