#!/usr/bin/env python3
"""Phase 3 Step 1: Build a category map from the validation problem set.

Reads problems.jsonl (or train.csv) and classifies each problem into:
  cryptarithm / bit_manipulation / numeral_conversion / cipher /
  equation / unit_conversion / rule_induction / arithmetic / logic / other

Also assigns subcategories for the three primary bottleneck groups.
Does NOT modify any adapter, config, or submission artefact.

Usage:
    python phase3_build_category_map.py \
        --input /kaggle/input/problems.jsonl \
        --output phase3_analysis/category_map.csv \
        --labeled-output phase3_analysis/validation_set_labeled.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Category classification rules (applied in priority order)
# ---------------------------------------------------------------------------

CRYPTARITHM_CATEGORIES = {"cryptarithm_deduce", "cryptarithm_guess"}

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "bit_manipulation": [
        r"\bXOR\b", r"\bAND\b", r"\bOR\b", r"\bNOT\b",
        r"\bshift\b", r"<<", r">>", r"\b0b[01]+\b", r"\bbitwise\b",
        r"\bbinary\b.*\boperation\b", r"\bbit\b",
    ],
    "numeral_conversion": [
        r"\bbinary\b", r"\bdecimal\b", r"\bhexadecimal\b", r"\bhex\b",
        r"\boctal\b", r"\bbase[ -]\d+\b", r"\broman numeral\b",
        r"\bconvert\b.*\bbase\b", r"\b0x[0-9a-fA-F]+\b", r"\b0b[01]+\b",
        r"\bconvert\b.*\bnumber\b",
    ],
    "cipher": [
        r"\bcipher\b", r"\bencode\b", r"\bdecode\b",
        r"\bencrypt\b", r"\bdecrypt\b", r"\bcaesar\b",
        r"\bROT\d+\b", r"\bshift cipher\b",
    ],
    "equation": [
        r"\bsolve\b.*\bfor\b", r"\bfind\b.*\bx\b", r"\bequation\b",
        r"\balgebra\b", r"\bvariable\b",
    ],
    "unit_conversion": [
        r"\bkilometer\b", r"\bmiles?\b", r"\bkilogram\b", r"\bpounds?\b",
        r"\bcelsius\b", r"\bfahrenheit\b", r"\bliter\b", r"\bgallon\b",
        r"\bmeter\b", r"\bfeet\b", r"\binch\b", r"\bcm\b", r"\bmph\b",
        r"\bkph\b",
    ],
    "rule_induction": [
        r"\bpattern\b", r"\bsequence\b", r"\binfer.*rule\b",
        r"\bnext.*element\b",
    ],
    "logic": [
        r"\bimplies\b", r"\bif.*then\b", r"\bpropositional\b",
        r"\btruth table\b", r"\bboolean\b", r"\bde morgan\b",
    ],
    "arithmetic": [
        r"\d+\s*[\+\-\*\/]\s*\d+", r"\bcalculate\b", r"\bcompute\b",
        r"\bsum\b", r"\bproduct\b", r"\bdifference\b",
    ],
}

CRYPTARITHM_SUBCATEGORY_RULES: List[Tuple[str, str]] = [
    ("alphametic_addition", r"[\+]|plus"),
    ("alphametic_subtraction", r"[\-]|minus"),
    ("alphametic_multiplication", r"[\*×]|times|mul"),
    ("digit_assignment", r"\bassign\b|\bdigit\b"),
    ("carry_reasoning", r"\bcarry\b"),
    ("leading_zero_constraint", r"leading zero|no leading"),
    ("string_transform", r"[A-Z]{2,}\s*\?\s*[A-Z]{2,}"),  # pure string xform
]

BIT_SUBCATEGORY_RULES: List[Tuple[str, str]] = [
    ("xor", r"\bXOR\b|⊕"),
    ("and", r"\bAND\b|(?<![a-z])&(?![a-z])"),
    ("or", r"\bOR\b|(?<![a-z])\|(?![a-z])"),
    ("shift_left", r"<<|\bshl\b|\bshift left\b"),
    ("shift_right", r">>|\bshr\b|\bshift right\b"),
    ("mask", r"\bmask\b|\bbitmask\b"),
    ("signed_unsigned", r"\bsigned\b|\bunsigned\b|\btwo.?s complement\b"),
    ("binary_arithmetic", r"\bbinary addition\b|\bbinary subtraction\b"),
]

NUMERAL_SUBCATEGORY_RULES: List[Tuple[str, str]] = [
    ("binary_to_decimal", r"binary to decimal|0b[01]+ to \d|bin.*dec"),
    ("decimal_to_binary", r"decimal to binary|\d+ to binary|dec.*bin"),
    ("hex_to_decimal", r"hex(?:adecimal)? to decimal|0x[0-9a-fA-F]+ to \d"),
    ("decimal_to_hex", r"decimal to hex|\d+ to hex(?:adecimal)?"),
    ("roman_numeral", r"roman numeral|XLIV|MMXX"),
    ("base_n_conversion", r"base[ -]\d+|convert.*base"),
]


def _match_any(patterns: List[str], text: str) -> Optional[str]:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return p
    return None


def classify_subcategory(category: str, text: str) -> str:
    rules: List[Tuple[str, str]] = []
    if category == "cryptarithm":
        rules = CRYPTARITHM_SUBCATEGORY_RULES
    elif category == "bit_manipulation":
        rules = BIT_SUBCATEGORY_RULES
    elif category == "numeral_conversion":
        rules = NUMERAL_SUBCATEGORY_RULES
    else:
        return "n/a"

    for subcategory, pattern in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return subcategory
    return "unknown"


def classify_problem(record: Dict[str, Any]) -> Dict[str, Any]:
    """Classify one problem record. Returns classification metadata."""
    raw_category = str(record.get("category", record.get("task", ""))).lower().strip()
    question = str(record.get("question", record.get("prompt", record.get("input", "")))).strip()
    examples_raw = record.get("examples", "")
    examples_text = json.dumps(examples_raw) if not isinstance(examples_raw, str) else examples_raw
    combined = f"{raw_category} {question} {examples_text}"

    matched_keywords: List[str] = []
    confidence = "low"
    rule = "fallback_keyword"
    manual_review = False

    # Priority 1: explicit category field for cryptarithm
    if raw_category in CRYPTARITHM_CATEGORIES:
        category = "cryptarithm"
        confidence = "high"
        rule = "explicit_category_field"
        subcategory = classify_subcategory("cryptarithm", combined)
        return {
            "category": category,
            "subcategory": subcategory,
            "confidence": confidence,
            "rule": rule,
            "matched_keywords": raw_category,
            "manual_review_required": False,
        }

    # Priority 2: keyword matching for other categories
    matched_categories: List[Tuple[str, str]] = []
    for cat, patterns in CATEGORY_KEYWORDS.items():
        m = _match_any(patterns, combined)
        if m:
            matched_categories.append((cat, m))

    if len(matched_categories) == 0:
        category = "other"
        subcategory = "unknown"
        confidence = "low"
        rule = "no_match_fallback"
        manual_review = True
    elif len(matched_categories) == 1:
        category, kw = matched_categories[0]
        subcategory = classify_subcategory(category, combined)
        matched_keywords = [kw]
        confidence = "medium"
        rule = "single_keyword_match"
    else:
        # Multiple matches — use priority order
        priority = list(CATEGORY_KEYWORDS.keys())
        matched_categories.sort(key=lambda x: priority.index(x[0]) if x[0] in priority else 99)
        category, kw = matched_categories[0]
        subcategory = classify_subcategory(category, combined)
        matched_keywords = [m[1] for m in matched_categories]
        confidence = "medium" if len(matched_categories) == 2 else "low"
        rule = "multi_keyword_priority"
        if len(matched_categories) >= 3:
            manual_review = True

    return {
        "category": category,
        "subcategory": subcategory,
        "confidence": confidence,
        "rule": rule,
        "matched_keywords": "|".join(matched_keywords),
        "manual_review_required": manual_review,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_csv(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def load_problems(paths: List[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            print(f"[WARN] {p} not found, skipping")
            continue
        if p.suffix == ".jsonl":
            records.extend(read_jsonl(p))
        elif p.suffix == ".csv":
            records.extend(read_csv(p))
        else:
            print(f"[WARN] Unsupported format {p.suffix}, skipping {p}")
    return records


def get_problem_id(record: Dict[str, Any], idx: int) -> str:
    for key in ("problem_id", "id", "uid", "sample_id"):
        v = record.get(key)
        if v:
            return str(v)
    return f"row_{idx}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_category_map(
    problems: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (category_map_rows, validation_labeled_rows)."""
    category_rows: List[Dict[str, Any]] = []
    labeled_rows: List[Dict[str, Any]] = []

    for idx, record in enumerate(problems):
        pid = get_problem_id(record, idx)
        clf = classify_problem(record)

        question = str(record.get("question", record.get("prompt", record.get("input", "")))).strip()
        answer = str(record.get("answer", record.get("target", record.get("output", "")))).strip()

        category_rows.append({
            "problem_id": pid,
            "category": clf["category"],
            "subcategory": clf["subcategory"],
            "confidence": clf["confidence"],
            "rule": clf["rule"],
            "matched_keywords": clf["matched_keywords"],
            "manual_review_required": clf["manual_review_required"],
        })

        labeled_rows.append({
            "problem_id": pid,
            "category": clf["category"],
            "subcategory": clf["subcategory"],
            "confidence": clf["confidence"],
            "manual_review_required": clf["manual_review_required"],
            "question": question[:300],  # truncate for readability
            "gold_answer": answer,
        })

    return category_rows, labeled_rows


def write_csv_rows(rows: List[Dict[str, Any]], path: Path, fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="+",
        default=["problems.jsonl", "train.csv"],
        help="Input problem file(s) (.jsonl or .csv)",
    )
    parser.add_argument(
        "--output", default="phase3_analysis/category_map.csv",
        help="Output category_map.csv path",
    )
    parser.add_argument(
        "--labeled-output", default="phase3_analysis/validation_set_labeled.csv",
        help="Output validation_set_labeled.csv path",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    problems = load_problems(input_paths)

    if not problems:
        print("[ERROR] No problems loaded. Check --input paths.")
        return

    category_rows, labeled_rows = build_category_map(problems)

    cat_fields = [
        "problem_id", "category", "subcategory",
        "confidence", "rule", "matched_keywords", "manual_review_required",
    ]
    write_csv_rows(category_rows, Path(args.output), cat_fields)

    labeled_fields = [
        "problem_id", "category", "subcategory",
        "confidence", "manual_review_required", "question", "gold_answer",
    ]
    write_csv_rows(labeled_rows, Path(args.labeled_output), labeled_fields)

    # Summary
    from collections import Counter
    cat_counts = Counter(r["category"] for r in category_rows)
    print(f"Loaded {len(problems)} problems.")
    print("Category distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(len(problems), 1)
        print(f"  {cat:25s} {count:5d}  ({pct:.1f}%)")
    manual_count = sum(1 for r in category_rows if r["manual_review_required"])
    print(f"Manual review required: {manual_count}")
    print(f"Wrote category_map    -> {args.output}")
    print(f"Wrote validation_set  -> {args.labeled_output}")


if __name__ == "__main__":
    main()
