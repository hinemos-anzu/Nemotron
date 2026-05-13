#!/usr/bin/env python3
"""Validate a solver-verified cryptarithm corpus patch JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_PATCH = Path("reports/cryptarithm/cryptarithm_corpus_patch.jsonl")
CRYPTO_CATEGORIES = {"cryptarithm_deduce", "cryptarithm_guess"}


def validate_record(row: Dict[str, Any], line_no: int) -> List[str]:
    errors: List[str] = []
    if row.get("category") not in CRYPTO_CATEGORIES:
        errors.append(f"line {line_no}: unexpected category {row.get('category')!r}")
    if not row.get("answer"):
        errors.append(f"line {line_no}: missing answer")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        errors.append(f"line {line_no}: expected two chat messages")
    else:
        expected_roles = ["user", "assistant"]
        for idx, (message, expected_role) in enumerate(zip(messages, expected_roles), 1):
            if not isinstance(message, dict):
                errors.append(f"line {line_no}: message {idx} must be an object")
                continue
            if message.get("role") != expected_role:
                errors.append(f"line {line_no}: message {idx} role must be {expected_role!r}")
            if not message.get("content"):
                errors.append(f"line {line_no}: message {idx} missing content")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        errors.append(f"line {line_no}: metadata must be an object")
    elif metadata.get("verified") is not True:
        errors.append(f"line {line_no}: metadata.verified must be true")
    return errors


def validate_patch(path: Path, require_rows: bool = False) -> int:
    errors: List[str] = []
    count = 0
    if not path.exists():
        errors.append(f"missing patch file: {path}")
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"line {line_no}: invalid JSON: {exc}")
                    continue
                errors.extend(validate_record(row, line_no))
    if require_rows and count == 0:
        errors.append("patch contains no rows")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", default=str(DEFAULT_PATCH))
    parser.add_argument("--require-rows", action="store_true", help="Fail when the patch is empty.")
    args = parser.parse_args()
    count = validate_patch(Path(args.patch), args.require_rows)
    print(f"validated {count} corpus patch rows in {args.patch}")


if __name__ == "__main__":
    main()
