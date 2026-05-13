#!/usr/bin/env python3
"""Generate solver-verified cryptarithm CoT samples and failure reports."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from cryptarithm_solver import parse_examples

DEFAULT_COVERAGE = Path("reports/cryptarithm/cryptarithm_solver_coverage.csv")
DEFAULT_JSONL = Path("reports/cryptarithm/cryptarithm_generated_cot_sample.jsonl")
DEFAULT_REPORT = Path("reports/cryptarithm/cryptarithm_failure_report.md")


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def reasoning_text(row: Dict[str, str]) -> str:
    category = row.get("category", "")
    rule = row.get("rule_type", "unknown")
    examples = parse_examples(row.get("examples", ""))
    lines: List[str] = []
    if category == "cryptarithm_guess":
        lines.append("We need to guess the hidden operation.")
        lines.append("Candidate rules: forward concatenation, reverse concatenation, swapped operands, reversed sides, and interleaving.")
        lines.append("Check each candidate against the examples:")
    else:
        lines.append("We need to infer the rule from the examples.")
    for idx, ex in enumerate(examples, 1):
        lines.append(f"Example {idx}: input = {ex.left}{ex.op}{ex.right}, output = {ex.output}.")
    lines.append(f"The consistent solver-verified rule is: {rule}.")
    lines.append(f"Apply the rule to the question: {row.get('question', '')}.")
    lines.append(f"The solver obtains {row.get('solver_answer', '')}, matching the labeled answer.")
    lines.append(f"Therefore, the answer is \\boxed{{{row.get('solver_answer', '')}}}.")
    return "\n".join(lines)


def write_jsonl(rows: Sequence[Dict[str, str]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row.get("verified") != "true":
                continue
            payload = {
                "problem_id": row.get("problem_id", ""),
                "category": row.get("category", ""),
                "rule_type": row.get("rule_type", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "solver_answer": row.get("solver_answer", ""),
                "verified": True,
                "reasoning_text": reasoning_text(row),
                "mask_policy": "final_answer_and_rule_application",
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_report(rows: Sequence[Dict[str, str]], output: Path, verified_count: int, jsonl_path: Path) -> None:
    by_category = defaultdict(list)
    for row in rows:
        by_category[row.get("category", "unknown")].append(row)
    failure_counts = Counter(row.get("failure_type") or "verified" for row in rows)
    unsolved_rules = Counter(row.get("solver_notes", "unknown") for row in rows if row.get("verified") != "true")
    mismatch_examples = [row for row in rows if row.get("verified") != "true"][:5]

    lines = [
        "# Cryptarithm Failure Report",
        "",
        "This report is generated without training, adapter surgery, rank compression, or submission packaging.",
        "",
        "## Coverage by category",
        "",
        "| category | total | verified | coverage |",
        "|---|---:|---:|---:|",
    ]
    for category in sorted(by_category):
        total = len(by_category[category])
        solved = sum(row.get("verified") == "true" for row in by_category[category])
        coverage = (solved / total) if total else 0.0
        lines.append(f"| {category} | {total} | {solved} | {coverage:.2%} |")
    lines.extend(
        [
            "",
            f"Verified CoT samples written: `{jsonl_path}` ({verified_count} rows).",
            "",
            "## Failure type counts",
            "",
            "| failure_type | count |",
            "|---|---:|",
        ]
    )
    for failure, count in failure_counts.most_common():
        lines.append(f"| {failure} | {count} |")
    lines.extend(["", "## Unsupported rule / mismatch notes", "", "| note | count |", "|---|---:|"])
    if unsolved_rules:
        for note, count in unsolved_rules.most_common(10):
            safe_note = note.replace("|", "\\|")
            lines.append(f"| {safe_note} | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(["", "## Answer mismatch examples", ""])
    if mismatch_examples:
        for row in mismatch_examples:
            lines.append(
                f"- `{row.get('problem_id', '')}` ({row.get('category', '')}): solver=`{row.get('solver_answer', '')}`, "
                f"answer=`{row.get('answer', '')}`, note={row.get('solver_notes', '')}"
            )
    else:
        lines.append("- No mismatches in the current inventory.")
    lines.extend(
        [
            "",
            "## Next templates to add",
            "",
            "1. Add explicit templates for any high-count unsupported notes above.",
            "2. Keep cryptarithm_deduce and cryptarithm_guess separated in downstream corpus patches.",
            "3. Proceed to a deduce-only SFT A/B only after verified coverage is large enough for a low-noise patch.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", default=str(DEFAULT_COVERAGE))
    parser.add_argument("--output", default=str(DEFAULT_JSONL))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    rows = read_rows(Path(args.coverage))
    count = write_jsonl(rows, Path(args.output))
    write_report(rows, Path(args.report), count, Path(args.output))
    print(f"wrote {count} verified CoT rows to {args.output}")
    print(f"wrote failure report to {args.report}")


if __name__ == "__main__":
    main()
