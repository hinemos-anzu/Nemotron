#!/usr/bin/env python3
"""Run the cryptarithm diagnostics pipeline with Kaggle-friendly defaults.

The script is a thin orchestrator around the standalone cryptarithm tools. It is
intended for Kaggle notebooks / scripts where raw files usually live under
`/kaggle/input` and generated outputs should go to `/kaggle/working`.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cryptarithm_build_corpus_patch as patch_builder
import cryptarithm_generate_verified_cot as cot_generator
import cryptarithm_inventory as inventory
import cryptarithm_solver as solver
import cryptarithm_validate_corpus_patch as patch_validator

RAW_FILENAMES = ("problems.jsonl", "train.csv", "corpus.jsonl")
KAGGLE_OUTPUT_DIR = Path("/kaggle/working/cryptarithm")
LOCAL_OUTPUT_DIR = Path("reports/cryptarithm")


def existing_roots(paths: Sequence[str]) -> List[Path]:
    return [Path(path) for path in paths if Path(path).exists()]


def find_first_named(roots: Sequence[Path], filename: str) -> Path | None:
    for root in roots:
        direct = root / filename
        if direct.exists():
            return direct
        matches = sorted(path for path in root.rglob(filename) if path.is_file())
        if matches:
            return matches[0]
    return None


def discover_inputs(roots: Sequence[Path]) -> Dict[str, Path]:
    return {name: path for name in RAW_FILENAMES if (path := find_first_named(roots, name)) is not None}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def coverage_by_category(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "verified": 0})
    for row in rows:
        category = row.get("category", "unknown")
        summary[category]["total"] += 1
        if row.get("verified") == "true":
            summary[category]["verified"] += 1
    return dict(summary)


def write_summary(
    output: Path,
    input_paths: Dict[str, Path],
    inventory_path: Path,
    coverage_path: Path,
    cot_path: Path,
    failure_report_path: Path,
    patch_path: Path,
    patch_rows: int,
    skipped_patch_rows: int,
) -> None:
    inventory_rows = read_csv_rows(inventory_path)
    coverage_rows = read_csv_rows(coverage_path)
    category_counts = Counter(row.get("category", "unknown") for row in inventory_rows)
    failure_counts = Counter(row.get("failure_type") or "verified" for row in coverage_rows)
    category_coverage = coverage_by_category(coverage_rows)
    verified_cot_rows = count_jsonl(cot_path)

    lines = [
        "# Cryptarithm Kaggle Coverage Summary",
        "",
        "## Inputs",
        "",
    ]
    for name in RAW_FILENAMES:
        lines.append(f"- `{name}`: `{input_paths.get(name, 'not found')}`")
    lines.extend(
        [
            "",
            "## Generated outputs",
            "",
            f"- inventory: `{inventory_path}`",
            f"- coverage: `{coverage_path}`",
            f"- verified CoT: `{cot_path}` ({verified_cot_rows} rows)",
            f"- failure report: `{failure_report_path}`",
            f"- corpus patch: `{patch_path}` ({patch_rows} rows, skipped={skipped_patch_rows})",
            "",
            "## Inventory counts",
            "",
            "| category | count |",
            "|---|---:|",
        ]
    )
    for category, count in sorted(category_counts.items()):
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Solver coverage", "", "| category | total | verified | coverage |", "|---|---:|---:|---:|"])
    for category, counts in sorted(category_coverage.items()):
        total = counts["total"]
        verified = counts["verified"]
        coverage = (verified / total) if total else 0.0
        lines.append(f"| {category} | {total} | {verified} | {coverage:.2%} |")
    lines.extend(["", "## Failure buckets", "", "| failure_type | count |", "|---|---:|"])
    for failure_type, count in failure_counts.most_common():
        lines.append(f"| {failure_type} | {count} |")
    decision = "do not train yet"
    if verified_cot_rows >= 50:
        decision = "consider deduce-only patch review before training"
    if verified_cot_rows >= 200:
        decision = "enough rows to consider deduce/guess separated A/B planning"
    lines.extend(["", "## Decision", "", f"- Recommended next step: **{decision}**."])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> Dict[str, Path | int]:
    roots = existing_roots(args.input_dir)
    input_paths = discover_inputs(roots)
    if not input_paths:
        raise SystemExit(f"No input files named {', '.join(RAW_FILENAMES)} found under: {args.input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "cryptarithm_problem_inventory.csv"
    coverage_path = output_dir / "cryptarithm_solver_coverage.csv"
    cot_path = output_dir / "cryptarithm_generated_cot_sample.jsonl"
    failure_report_path = output_dir / "cryptarithm_failure_report.md"
    patch_path = output_dir / "cryptarithm_corpus_patch.jsonl"
    summary_path = output_dir / "cryptarithm_realdata_summary.md"

    records = inventory.load_records([input_paths[name] for name in RAW_FILENAMES if name in input_paths])
    corpus_path = input_paths.get("corpus.jsonl")
    corpus_ids = inventory.corpus_ids(corpus_path) if corpus_path else set()
    inventory_rows = inventory.build_inventory(records, corpus_ids)
    inventory.write_csv(inventory_rows, inventory_path)

    coverage_rows = solver.write_coverage(inventory_rows, coverage_path)
    verified_cot_rows = cot_generator.write_jsonl(coverage_rows, cot_path)
    cot_generator.write_report(coverage_rows, failure_report_path, verified_cot_rows, cot_path)

    patch_rows = 0
    skipped_patch_rows = 0
    if verified_cot_rows or not args.skip_empty_patch:
        patch_rows, skipped_patch_rows = patch_builder.build_patch(cot_path, patch_path, strict=args.strict_patch)
        if args.require_patch_rows:
            patch_validator.validate_patch(patch_path, require_rows=True)
    elif patch_path.exists():
        patch_path.unlink()

    write_summary(
        summary_path,
        input_paths,
        inventory_path,
        coverage_path,
        cot_path,
        failure_report_path,
        patch_path,
        patch_rows,
        skipped_patch_rows,
    )
    print(f"wrote Kaggle cryptarithm summary to {summary_path}")
    print(f"inventory_rows={len(inventory_rows)} coverage_rows={len(coverage_rows)} verified_cot_rows={verified_cot_rows} patch_rows={patch_rows}")
    return {
        "inventory_path": inventory_path,
        "coverage_path": coverage_path,
        "cot_path": cot_path,
        "failure_report_path": failure_report_path,
        "patch_path": patch_path,
        "summary_path": summary_path,
        "verified_cot_rows": verified_cot_rows,
        "patch_rows": patch_rows,
    }


def default_output_dir() -> str:
    return str(KAGGLE_OUTPUT_DIR if Path("/kaggle/working").exists() else LOCAL_OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        default=None,
        help="Directory to search recursively for problems.jsonl/train.csv/corpus.jsonl. Repeatable. Defaults to /kaggle/input and data/raw.",
    )
    parser.add_argument("--output-dir", default=default_output_dir(), help="Output directory for generated diagnostics.")
    parser.add_argument("--strict-patch", action="store_true", help="Fail if any verified-CoT source row is malformed while building the patch.")
    parser.add_argument("--require-patch-rows", action="store_true", help="Validate that the generated patch is non-empty.")
    parser.add_argument("--build-empty-patch", action="store_true", help="Create and validate an empty patch even when verified CoT count is zero.")
    args = parser.parse_args()
    args.skip_empty_patch = not args.build_empty_patch
    if args.input_dir is None:
        args.input_dir = ["/kaggle/input", "data/raw"]
    run_pipeline(args)


if __name__ == "__main__":
    main()
