#!/usr/bin/env python3
"""Build an inventory of cryptarithm_deduce / cryptarithm_guess problems.

The script is intentionally data-layout tolerant: it scans common challenge files
(`problems.jsonl`, `train.csv`, `corpus.jsonl`) when they exist and normalizes
records into a diagnostic CSV. It never mutates adapters, submissions, or source
corpora.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

CRYPTO_CATEGORIES = {"cryptarithm_deduce", "cryptarithm_guess"}
DEFAULT_INPUTS = ("problems.jsonl", "train.csv", "corpus.jsonl")
DEFAULT_OUTPUT = Path("reports/cryptarithm/cryptarithm_problem_inventory.csv")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if isinstance(obj, dict):
                obj["_source_path"] = str(path)
                yield obj


def read_csv(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["_source_path"] = str(path)
            yield dict(row)


def load_records(paths: Sequence[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            yield from read_jsonl(path)
        elif suffix == ".csv":
            yield from read_csv(path)


def first_present(record: Dict[str, Any], names: Sequence[str], default: str = "") -> str:
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            return str(value)
    return default


def detect_category(record: Dict[str, Any]) -> str:
    for key in ("category", "task", "problem_type", "type", "source_category"):
        value = str(record.get(key, ""))
        if value in CRYPTO_CATEGORIES:
            return value
    blob = json.dumps(record, ensure_ascii=False).lower()
    for category in CRYPTO_CATEGORIES:
        if category in blob:
            return category
    return ""


def normalize_examples(raw: str) -> str:
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True)


def operator_symbols(text: str) -> str:
    # Capture non-alphanumeric operator-like separators between token runs.
    ops: Set[str] = set()
    for match in re.finditer(r"[A-Za-z0-9]+\s*([^A-Za-z0-9\s]{1,4})\s*[A-Za-z0-9]+", text):
        ops.add(match.group(1))
    return " ".join(sorted(ops))


def length_summary(question: str, examples: str, answer: str) -> str:
    return json.dumps(
        {
            "question_chars": len(question),
            "examples_chars": len(examples),
            "answer_chars": len(answer),
        },
        sort_keys=True,
    )


def build_inventory(records: Iterable[Dict[str, Any]], corpus_problem_ids: Set[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for idx, record in enumerate(records):
        category = detect_category(record)
        if category not in CRYPTO_CATEGORIES:
            continue
        problem_id = first_present(record, ("problem_id", "id", "uid", "sample_id"), f"row_{idx}")
        question = first_present(record, ("question", "prompt", "input", "problem", "messages"))
        answer = first_present(record, ("answer", "target", "output", "solution", "label"))
        examples = normalize_examples(first_present(record, ("examples", "shots", "demonstrations", "few_shot", "metadata")))
        source_path = first_present(record, ("_source_path",))
        combined_text = "\n".join([question, examples, answer])
        rows.append(
            {
                "problem_id": problem_id,
                "category": category,
                "question": question,
                "answer": answer,
                "examples": examples,
                "input_output_length": length_summary(question, examples, answer),
                "operator_symbols": operator_symbols(combined_text),
                "current_corpus_included": str(problem_id in corpus_problem_ids).lower(),
                "source_path": source_path,
            }
        )
    return rows


def corpus_ids(path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not path.exists():
        return ids
    for record in read_jsonl(path):
        problem_id = first_present(record, ("problem_id", "id", "uid", "sample_id"))
        if problem_id:
            ids.add(problem_id)
    return ids


def write_csv(rows: List[Dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "problem_id",
        "category",
        "question",
        "answer",
        "examples",
        "input_output_length",
        "operator_symbols",
        "current_corpus_included",
        "source_path",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", default=list(DEFAULT_INPUTS), help="Input JSONL/CSV files to scan.")
    parser.add_argument("--corpus", default="corpus.jsonl", help="Corpus JSONL used to mark current inclusion.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output inventory CSV path.")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    rows = build_inventory(load_records(input_paths), corpus_ids(Path(args.corpus)))
    write_csv(rows, Path(args.output))
    print(f"wrote {len(rows)} cryptarithm rows to {args.output}")


if __name__ == "__main__":
    main()
