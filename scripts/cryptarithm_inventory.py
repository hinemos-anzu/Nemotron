"""
cryptarithm_inventory.py

Reads problems.jsonl / train.csv / corpus.jsonl from DATA_DIR,
filters for cryptarithm_deduce and cryptarithm_guess, and writes
reports/cryptarithm/cryptarithm_inventory.csv with per-problem metadata.

When real data files are absent the script falls back to built-in
synthetic examples so it can run standalone for structure validation.

Output columns:
  problem_id, category, question, answer, examples_count,
  question_len, answer_len, operator_symbols, in_corpus,
  source_file
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Paths (override via env vars)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT))
OUTPUT_DIR = REPO_ROOT / "reports" / "cryptarithm"
OUTPUT_PATH = OUTPUT_DIR / "cryptarithm_inventory.csv"

CRYPTARITHM_CATEGORIES = {"cryptarithm_deduce", "cryptarithm_guess"}
_OP_RE = re.compile(r"[+\-*/★⊕⊖⊗×÷@#$^|~&]+")

# ---------------------------------------------------------------------------
# Synthetic fallback data (used when real files are absent)
# ---------------------------------------------------------------------------

SYNTHETIC_PROBLEMS: List[Dict] = [
    # ---- cryptarithm_deduce ----
    {
        "id": "syn_ded_001", "category": "cryptarithm_deduce",
        "question": "3 + 5 = 35\n7 + 2 = 72\n4 + 8 = ?",
        "answer": "48",
        "examples": [{"input": "3 + 5", "output": "35"}, {"input": "7 + 2", "output": "72"}],
    },
    {
        "id": "syn_ded_002", "category": "cryptarithm_deduce",
        "question": "3 + 5 = 53\n7 + 2 = 27\n4 + 8 = ?",
        "answer": "84",
        "examples": [{"input": "3 + 5", "output": "53"}, {"input": "7 + 2", "output": "27"}],
    },
    {
        "id": "syn_ded_003", "category": "cryptarithm_deduce",
        "question": "12 + 34 = 2134\n56 + 78 = 6578\n90 + 11 = ?",
        "answer": "0911",
        "examples": [{"input": "12 + 34", "output": "2134"}, {"input": "56 + 78", "output": "6578"}],
    },
    {
        "id": "syn_ded_004", "category": "cryptarithm_deduce",
        "question": "12 + 34 = 1243\n56 + 78 = 5687\n90 + 11 = ?",
        "answer": "9011",
        "examples": [{"input": "12 + 34", "output": "1243"}, {"input": "56 + 78", "output": "5687"}],
    },
    {
        "id": "syn_ded_005", "category": "cryptarithm_deduce",
        "question": "123 + 456 = 321654\n789 + 012 = 987210\n345 + 678 = ?",
        "answer": "543876",
        "examples": [{"input": "123 + 456", "output": "321654"}, {"input": "789 + 012", "output": "987210"}],
    },
    {
        "id": "syn_ded_006", "category": "cryptarithm_deduce",
        "question": "AB + CD = ACBD\nEF + GH = EGFH\nIJ + KL = ?",
        "answer": "IKJL",
        "examples": [{"input": "AB + CD", "output": "ACBD"}, {"input": "EF + GH", "output": "EGFH"}],
    },
    {
        "id": "syn_ded_007", "category": "cryptarithm_deduce",
        "question": "AB + CD = CADB\nEF + GH = GEHF\nIJ + KL = ?",
        "answer": "KILJ",
        "examples": [{"input": "AB + CD", "output": "CADB"}, {"input": "EF + GH", "output": "GEHF"}],
    },
    {
        "id": "syn_ded_008", "category": "cryptarithm_deduce",
        "question": "12 ★ 34 = 1234\n56 ★ 78 = 5678\n99 ★ 11 = ?",
        "answer": "9911",
        "examples": [{"input": "12 ★ 34", "output": "1234"}, {"input": "56 ★ 78", "output": "5678"}],
    },
    {
        "id": "syn_ded_009", "category": "cryptarithm_deduce",
        "question": "HELLO + WORLD = HELLOWORLD\nFOO + BAR = FOOBAR\nABC + DEF = ?",
        "answer": "ABCDEF",
        "examples": [
            {"input": "HELLO + WORLD", "output": "HELLOWORLD"},
            {"input": "FOO + BAR", "output": "FOOBAR"},
        ],
    },
    {
        "id": "syn_ded_010", "category": "cryptarithm_deduce",
        "question": "1 + 2 = 99\n3 + 4 = 88\n5 + 6 = ?",
        "answer": "77",
        "examples": [{"input": "1 + 2", "output": "99"}, {"input": "3 + 4", "output": "88"}],
    },
    # ---- cryptarithm_guess ----
    {
        "id": "syn_gue_001", "category": "cryptarithm_guess",
        "question": "Given: 3 + 5 = 35, 7 + 2 = 72. What rule produces these outputs?",
        "answer": "forward_concat",
        "examples": [{"input": "3 + 5", "output": "35"}, {"input": "7 + 2", "output": "72"}],
    },
    {
        "id": "syn_gue_002", "category": "cryptarithm_guess",
        "question": "Given: 3 + 5 = 53, 7 + 2 = 27. What rule produces these outputs?",
        "answer": "reverse_concat",
        "examples": [{"input": "3 + 5", "output": "53"}, {"input": "7 + 2", "output": "27"}],
    },
    {
        "id": "syn_gue_003", "category": "cryptarithm_guess",
        "question": "Given: AB + CD = ACBD, EF + GH = EGFH. What rule produces these outputs?",
        "answer": "interleave_lr",
        "examples": [{"input": "AB + CD", "output": "ACBD"}, {"input": "EF + GH", "output": "EGFH"}],
    },
    {
        "id": "syn_gue_004", "category": "cryptarithm_guess",
        "question": "Given: 12 + 34 = 2143, 56 + 78 = 6587. What rule produces these outputs?",
        "answer": "reverse_both",
        "examples": [{"input": "12 + 34", "output": "2143"}, {"input": "56 + 78", "output": "6587"}],
    },
    {
        "id": "syn_gue_005", "category": "cryptarithm_guess",
        "question": "Given: 12 + 34 = 2134, 56 + 78 = 6578. What rule produces these outputs?",
        "answer": "reverse_left",
        "examples": [{"input": "12 + 34", "output": "2134"}, {"input": "56 + 78", "output": "6578"}],
    },
    {
        "id": "syn_gue_006", "category": "cryptarithm_guess",
        "question": "Given: 12 + 34 = 1243, 56 + 78 = 5687. What rule produces these outputs?",
        "answer": "reverse_right",
        "examples": [{"input": "12 + 34", "output": "1243"}, {"input": "56 + 78", "output": "5687"}],
    },
    {
        "id": "syn_gue_007", "category": "cryptarithm_guess",
        "question": "Given: AB + CD = CADB, EF + GH = GEHF. What rule produces these outputs?",
        "answer": "interleave_rl",
        "examples": [{"input": "AB + CD", "output": "CADB"}, {"input": "EF + GH", "output": "GEHF"}],
    },
    {
        "id": "syn_gue_008", "category": "cryptarithm_guess",
        "question": "Given: 123 + 456 = 321654, 789 + 012 = 987210. What rule produces these outputs?",
        "answer": "reverse_both",
        "examples": [{"input": "123 + 456", "output": "321654"}, {"input": "789 + 012", "output": "987210"}],
    },
    {
        "id": "syn_gue_009", "category": "cryptarithm_guess",
        "question": "Given: 1 + 2 = 99, 3 + 4 = 88. No standard string rule applies. Name it.",
        "answer": "unknown_operator_fallback",
        "examples": [{"input": "1 + 2", "output": "99"}, {"input": "3 + 4", "output": "88"}],
    },
    {
        "id": "syn_gue_010", "category": "cryptarithm_guess",
        "question": "Given: 12 ★ 34 = 1234, 56 ★ 78 = 5678. What rule does ★ follow?",
        "answer": "forward_concat",
        "examples": [{"input": "12 ★ 34", "output": "1234"}, {"input": "56 ★ 78", "output": "5678"}],
    },
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_csv(path: Path) -> List[Dict]:
    import csv as _csv
    records = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = _csv.DictReader(fh)
        records = list(reader)
    return records


def _field(rec: Dict, candidates: List[str], default="") -> str:
    for c in candidates:
        if c in rec and rec[c] is not None:
            return str(rec[c])
    return default


def load_all_records(data_dir: Path) -> tuple[List[Dict], Set[str]]:
    """
    Load records from problems.jsonl, train.csv, corpus.jsonl.
    Returns (all_records, corpus_ids).
    Falls back to SYNTHETIC_PROBLEMS when no files are found.
    """
    records: List[Dict] = []
    corpus_ids: Set[str] = set()
    found_any = False

    search_paths = {
        "problems.jsonl": ["problems.jsonl", "data/problems.jsonl", "input/problems.jsonl"],
        "train.csv": ["train.csv", "data/train.csv", "input/train.csv"],
        "corpus.jsonl": ["corpus.jsonl", "data/corpus.jsonl", "input/corpus.jsonl"],
    }

    for fname, candidates in search_paths.items():
        for rel in candidates:
            p = data_dir / rel
            if p.exists():
                found_any = True
                if p.suffix == ".jsonl":
                    loaded = _load_jsonl(p)
                else:
                    loaded = _load_csv(p)
                print(f"  [load] {p}  → {len(loaded)} records", flush=True)
                if "corpus" in fname:
                    for r in loaded:
                        rid = _field(r, ["id", "problem_id", "ID"])
                        if rid:
                            corpus_ids.add(rid)
                else:
                    records.extend(loaded)
                break

    if not found_any:
        print(
            "  [warn] No data files found under DATA_DIR. "
            f"Using {len(SYNTHETIC_PROBLEMS)} built-in synthetic examples.",
            flush=True,
        )
        records = SYNTHETIC_PROBLEMS
        # Mark no corpus entries for synthetic data
    else:
        # corpus.jsonl entries are also training candidates — add to records
        pass

    return records, corpus_ids


# ---------------------------------------------------------------------------
# Inventory extraction
# ---------------------------------------------------------------------------

def _extract_operators(text: str) -> str:
    ops = sorted(set(_OP_RE.findall(text)))
    return "|".join(ops) if ops else ""


def _count_examples(rec: Dict) -> int:
    ex = rec.get("examples")
    if isinstance(ex, list):
        return len(ex)
    if isinstance(ex, str):
        try:
            parsed = json.loads(ex)
            if isinstance(parsed, list):
                return len(parsed)
        except Exception:
            pass
    # Fall back: count "=" occurrences in question (rough heuristic)
    q = _field(rec, ["question", "input", "prompt"])
    return max(0, q.count("=") - 1)


def build_inventory(records: List[Dict], corpus_ids: Set[str]) -> List[Dict]:
    rows = []
    for rec in records:
        cat = _field(rec, ["category", "task_type", "type"])
        if cat not in CRYPTARITHM_CATEGORIES:
            continue

        pid = _field(rec, ["id", "problem_id", "ID"], default=f"unk_{len(rows)}")
        question = _field(rec, ["question", "input", "prompt"])
        answer = _field(rec, ["answer", "output", "label", "target"])
        src = rec.get("_source_file", "unknown")

        rows.append({
            "problem_id": pid,
            "category": cat,
            "question": question.replace("\n", " | "),
            "answer": answer,
            "examples_count": _count_examples(rec),
            "question_len": len(question),
            "answer_len": len(answer),
            "operator_symbols": _extract_operators(question),
            "in_corpus": pid in corpus_ids,
            "source_file": src,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "problem_id", "category", "question", "answer",
    "examples_count", "question_len", "answer_len",
    "operator_symbols", "in_corpus", "source_file",
]


def main() -> None:
    print("=== cryptarithm_inventory ===", flush=True)
    print(f"DATA_DIR : {DATA_DIR}", flush=True)
    print(f"OUTPUT   : {OUTPUT_PATH}", flush=True)

    records, corpus_ids = load_all_records(DATA_DIR)

    # Tag source file
    for r in records:
        if "_source_file" not in r:
            r["_source_file"] = r.get("__source__", "synthetic_or_unknown")

    inventory = build_inventory(records, corpus_ids)

    deduce = [r for r in inventory if r["category"] == "cryptarithm_deduce"]
    guess = [r for r in inventory if r["category"] == "cryptarithm_guess"]
    print(f"\n  cryptarithm_deduce : {len(deduce)}", flush=True)
    print(f"  cryptarithm_guess  : {len(guess)}", flush=True)
    print(f"  total              : {len(inventory)}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(inventory)

    print(f"\n  Saved → {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
