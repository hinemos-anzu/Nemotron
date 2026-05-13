#!/usr/bin/env python3
"""Build a JSONL corpus patch from solver-verified cryptarithm CoT samples.

This script only creates an additive patch file. It does not edit the existing
corpus, adapter files, training config, or submissions. Rows are written only
when the source CoT sample is explicitly solver-verified and contains the fields
needed for supervised fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_INPUT = Path("reports/cryptarithm/cryptarithm_generated_cot_sample.jsonl")
DEFAULT_OUTPUT = Path("reports/cryptarithm/cryptarithm_corpus_patch.jsonl")
CRYPTO_CATEGORIES = {"cryptarithm_deduce", "cryptarithm_guess"}


def sample_errors(sample: Dict[str, Any], line_no: int) -> List[str]:
    errors: List[str] = []
    if sample.get("verified") is not True:
        errors.append(f"line {line_no}: sample.verified must be true")
    if sample.get("category") not in CRYPTO_CATEGORIES:
        errors.append(f"line {line_no}: unexpected category {sample.get('category')!r}")
    if not sample.get("question"):
        errors.append(f"line {line_no}: missing question")
    if not sample.get("solver_answer"):
        errors.append(f"line {line_no}: missing solver_answer")
    if not sample.get("reasoning_text"):
        errors.append(f"line {line_no}: missing reasoning_text")
    return errors


def to_patch_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "problem_id": sample.get("problem_id", ""),
        "category": sample.get("category", ""),
        "messages": [
            {"role": "user", "content": sample.get("question", "")},
            {"role": "assistant", "content": sample.get("reasoning_text", "")},
        ],
        "answer": sample.get("solver_answer", ""),
        "metadata": {
            "source": "cryptarithm_solver_verified_cot",
            "rule_type": sample.get("rule_type", ""),
            "verified": True,
            "mask_policy": sample.get("mask_policy", "final_answer_and_rule_application"),
        },
    }


def build_patch(source: Path, output: Path, max_rows: int = 0, strict: bool = False) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0
    errors: List[str] = []
    if not source.exists():
        if strict:
            print(f"missing CoT source file: {source}", file=sys.stderr)
            raise SystemExit(1)
        return count, 0
    with output.open("w", encoding="utf-8") as out:
        with source.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if max_rows and count >= max_rows:
                    break
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    message = f"line {line_no}: invalid JSON: {exc}"
                    if strict:
                        errors.append(message)
                    else:
                        print(f"skipping {message}", file=sys.stderr)
                        skipped += 1
                    continue
                row_errors = sample_errors(sample, line_no)
                if row_errors:
                    if strict:
                        errors.extend(row_errors)
                    else:
                        print("skipping " + "; ".join(row_errors), file=sys.stderr)
                        skipped += 1
                    continue
                out.write(json.dumps(to_patch_record(sample), ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    return count, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap; 0 means all rows.")
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping malformed or unverified CoT samples.")
    args = parser.parse_args()

    count, skipped = build_patch(Path(args.input), Path(args.output), args.max_rows, args.strict)
    print(f"wrote {count} additive corpus patch rows to {args.output}; skipped={skipped}")


if __name__ == "__main__":
    main()
