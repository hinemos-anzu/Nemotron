#!/usr/bin/env python3
"""Phase 3 Step 6: Classify bit_manipulation and numeral_conversion failures.

Produces:
  - failure_cases_bit_manipulation.csv
  - failure_cases_numeral_conversion.csv
  - Appends to failure_type_summary.csv

bit_manipulation failure_type labels:
  base_conversion_error  – wrong binary/hex conversion as sub-step
  xor_error              – wrong XOR semantics/application
  and_error              – wrong AND
  or_error               – wrong OR
  shift_error            – wrong shift direction or amount
  mask_error             – wrong mask application
  signed_unsigned_error  – confusion between signed and unsigned representation
  endian_error           – byte order confusion
  arithmetic_error       – arithmetic mistake in bit calculation
  final_parse_error      – correct reasoning but wrong extraction
  answer_format_error    – correct value, wrong format
  unknown                – cannot determine

numeral_conversion failure_type labels:
  binary_decimal_error   – wrong positional value calculation
  decimal_binary_error   – wrong binary encoding
  hex_decimal_error      – wrong hex-to-decimal mapping
  decimal_hex_error      – wrong decimal-to-hex mapping
  base_n_place_value_error – wrong place values in arbitrary base
  roman_numeral_error    – wrong Roman numeral rules
  digit_order_error      – correct digits, wrong order
  final_parse_error      – correct reasoning, wrong extraction
  answer_format_error    – correct value, wrong format
  unknown                – cannot determine

SAFETY CONTRACT: Read-only analysis. No adapter or training data modified.

Usage:
    python phase3_classify_bit_numeral_failures.py \
        --predictions phase3_analysis/golden_validation_predictions.jsonl \
        --logprob     phase3_analysis/min_logprob_summary.csv \
        --output-bit  phase3_analysis/failure_cases_bit_manipulation.csv \
        --output-numeral phase3_analysis/failure_cases_numeral_conversion.csv \
        --failure-type-summary phase3_analysis/failure_type_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOW_LOGPROB_CUTOFF = -2.0


# ---------------------------------------------------------------------------
# Bit manipulation failure classifiers
# ---------------------------------------------------------------------------

def classify_bit_failure(rec: Dict[str, Any], lp_row: Dict[str, Any]) -> Tuple[str, str]:
    raw = rec.get("raw_output", "")
    gold = rec.get("gold_answer", "")
    pred = rec.get("pred_answer", "")
    subcategory = rec.get("subcategory", "unknown")

    if not rec.get("is_correct", False):
        # Check parse error first
        if gold and re.search(re.escape(gold), raw, re.IGNORECASE) and not re.search(r"\\boxed\{", raw):
            return "answer_format_error", "correct value in output but missing \\boxed{} format"

        if gold and re.search(re.escape(gold), raw, re.IGNORECASE) and not pred:
            return "final_parse_error", "gold answer in reasoning but extraction failed"

        # Subcategory-driven classification
        if subcategory == "xor":
            # XOR error: model used AND or OR instead
            if re.search(r"\bAND\b|\bOR\b", raw) and not re.search(r"\bXOR\b", raw):
                return "xor_error", "model substituted AND/OR for XOR"
            return "xor_error", "XOR result is incorrect"

        if subcategory in {"and", "or"}:
            if re.search(r"\bXOR\b", raw):
                return f"{subcategory}_error", f"model applied XOR instead of {subcategory.upper()}"
            return f"{subcategory}_error", f"{subcategory.upper()} bit result is incorrect"

        if subcategory in {"shift_left", "shift_right"}:
            if re.search(r"shift", raw, re.IGNORECASE):
                return "shift_error", "shift direction or amount is wrong"
            return "shift_error", "shift operation result is incorrect"

        if subcategory == "mask":
            return "mask_error", "mask application result is incorrect"

        if subcategory == "signed_unsigned":
            if re.search(r"two.?s complement|negative", raw, re.IGNORECASE):
                return "signed_unsigned_error", "signed/unsigned interpretation error"
            return "signed_unsigned_error", "signed vs unsigned confusion"

        if subcategory == "binary_arithmetic":
            return "arithmetic_error", "binary arithmetic calculation error"

        # Generic fallback for bit
        if re.search(r"0b[01]+|binary", raw, re.IGNORECASE):
            return "base_conversion_error", "binary representation error in sub-step"

    return "unknown", "could not determine failure type from output"


# ---------------------------------------------------------------------------
# Numeral conversion failure classifiers
# ---------------------------------------------------------------------------

def classify_numeral_failure(rec: Dict[str, Any], lp_row: Dict[str, Any]) -> Tuple[str, str]:
    raw = rec.get("raw_output", "")
    gold = rec.get("gold_answer", "")
    pred = rec.get("pred_answer", "")
    subcategory = rec.get("subcategory", "unknown")

    if not rec.get("is_correct", False):
        # Format / parse checks first
        if gold and re.search(re.escape(gold), raw, re.IGNORECASE) and not re.search(r"\\boxed\{", raw):
            return "answer_format_error", "correct value in output but missing \\boxed{} format"

        if gold and re.search(re.escape(gold), raw, re.IGNORECASE) and not pred:
            return "final_parse_error", "gold answer in reasoning but extraction failed"

        # Check digit order error: same digits, wrong order
        if pred and gold:
            pred_clean = re.sub(r"[^0-9a-fA-F]", "", pred.upper())
            gold_clean = re.sub(r"[^0-9a-fA-F]", "", gold.upper())
            if len(pred_clean) == len(gold_clean) and sorted(pred_clean) == sorted(gold_clean):
                return "digit_order_error", "correct digits but wrong order"

        if subcategory == "binary_to_decimal":
            return "binary_decimal_error", "binary-to-decimal place value calculation error"

        if subcategory == "decimal_to_binary":
            return "decimal_binary_error", "decimal-to-binary encoding error"

        if subcategory == "hex_to_decimal":
            return "hex_decimal_error", "hex digit value mapping error"

        if subcategory == "decimal_to_hex":
            return "decimal_hex_error", "decimal-to-hex conversion error"

        if subcategory == "roman_numeral":
            return "roman_numeral_error", "Roman numeral rule application error"

        if subcategory == "base_n_conversion":
            return "base_n_place_value_error", "place value error in base-N conversion"

        # Generic: look for binary or hex patterns
        if re.search(r"0b[01]+", raw):
            return "binary_decimal_error", "binary representation detected — conversion error"
        if re.search(r"0x[0-9a-fA-F]+", raw):
            return "hex_decimal_error", "hex representation detected — conversion error"

    return "unknown", "could not determine failure type from output"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def compute_example_priority_bit_numeral(
    failure_type: str,
    solver_check: bool,
    synthetic_possible: bool,
    ans_min_lp: Optional[float],
    category: str,
) -> int:
    high_value_bit = {
        "xor_error", "shift_error", "and_error", "or_error",
        "base_conversion_error", "signed_unsigned_error",
    }
    high_value_num = {
        "binary_decimal_error", "decimal_binary_error",
        "hex_decimal_error", "decimal_hex_error", "base_n_place_value_error",
    }

    if category == "bit_manipulation" and failure_type in high_value_bit:
        base = 5 if solver_check else 4
    elif category == "numeral_conversion" and failure_type in high_value_num:
        base = 5 if solver_check else 4
    elif failure_type in {"answer_format_error", "final_parse_error"}:
        base = 4
    elif failure_type in {"digit_order_error", "mask_error"}:
        base = 4
    elif failure_type == "unknown":
        base = 2
    else:
        base = 3

    if synthetic_possible:
        base = min(5, base + 1)

    if ans_min_lp is not None and ans_min_lp < -3.0:
        base = min(5, base + 1)

    return base


def recommended_template_bit(failure_type: str) -> str:
    t = {
        "xor_error": "bitwise_xor_step_cot",
        "and_error": "bitwise_and_step_cot",
        "or_error": "bitwise_or_step_cot",
        "shift_error": "shift_direction_explicit_cot",
        "mask_error": "bitmask_application_cot",
        "signed_unsigned_error": "twos_complement_cot",
        "endian_error": "byte_order_explicit_cot",
        "base_conversion_error": "binary_to_decimal_substep_cot",
        "arithmetic_error": "step_verify_cot",
        "final_parse_error": "boxed_answer_strict_cot",
        "answer_format_error": "format_check_cot",
        "unknown": "general_bit_diagnostic_cot",
    }
    return t.get(failure_type, "general_bit_diagnostic_cot")


def recommended_template_numeral(failure_type: str) -> str:
    t = {
        "binary_decimal_error": "positional_binary_cot",
        "decimal_binary_error": "division_remainder_binary_cot",
        "hex_decimal_error": "hex_digit_table_cot",
        "decimal_hex_error": "division_remainder_hex_cot",
        "base_n_place_value_error": "positional_base_n_cot",
        "roman_numeral_error": "roman_rule_explicit_cot",
        "digit_order_error": "left_to_right_verify_cot",
        "final_parse_error": "boxed_answer_strict_cot",
        "answer_format_error": "format_check_cot",
        "unknown": "general_numeral_diagnostic_cot",
    }
    return t.get(failure_type, "general_numeral_diagnostic_cot")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    if not path.exists():
        return recs
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_logprob(path: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            result[row["problem_id"]] = row
    return result


FAILURE_CSV_FIELDS = [
    "problem_id", "category", "subcategory",
    "question", "gold_answer", "pred_answer",
    "is_correct", "min_logprob", "answer_min_logprob",
    "failure_type", "failure_reason",
    "solver_check_possible", "synthetic_generation_possible",
    "recommended_template", "example_priority",
]


def write_failure_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FAILURE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_failure_type_summary(rows: List[Dict[str, Any]], path: Path, category: str) -> None:
    counts = Counter(r["failure_type"] for r in rows)
    total = max(len(rows), 1)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "failure_type", "count", "pct"])
        for ft, count in sorted(counts.items(), key=lambda x: -x[1]):
            writer.writerow({
                "category": category,
                "failure_type": ft,
                "count": count,
                "pct": round(100 * count / total, 1),
            })


# ---------------------------------------------------------------------------
# Process one category
# ---------------------------------------------------------------------------

def process_category(
    category: str,
    preds: List[Dict[str, Any]],
    logprob_map: Dict[str, Dict[str, Any]],
    classifier,
    template_fn,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in preds:
        if rec.get("is_correct", False):
            continue  # only wrong predictions for bit/numeral
        pid = rec["problem_id"]
        lp_row = logprob_map.get(pid, {})

        ans_min_lp_str = lp_row.get("answer_min_logprob")
        ans_min_lp = float(ans_min_lp_str) if ans_min_lp_str else None
        min_lp_str = lp_row.get("min_logprob")
        min_lp = float(min_lp_str) if min_lp_str else None

        failure_type, failure_reason = classifier(rec, lp_row)

        solver_check = category == "numeral_conversion" or (
            category == "bit_manipulation" and
            rec.get("subcategory") in {"xor", "and", "or", "shift_left", "shift_right", "binary_arithmetic"}
        )
        synthetic_possible = failure_type not in {"unknown", "hallucinated_rule"}

        priority = compute_example_priority_bit_numeral(
            failure_type, solver_check, synthetic_possible, ans_min_lp, category
        )

        rows.append({
            "problem_id": pid,
            "category": category,
            "subcategory": rec.get("subcategory", "unknown"),
            "question": rec.get("question", "")[:300],
            "gold_answer": rec.get("gold_answer", ""),
            "pred_answer": rec.get("pred_answer", ""),
            "is_correct": rec.get("is_correct", False),
            "min_logprob": min_lp if min_lp is not None else "",
            "answer_min_logprob": ans_min_lp if ans_min_lp is not None else "",
            "failure_type": failure_type,
            "failure_reason": failure_reason,
            "solver_check_possible": solver_check,
            "synthetic_generation_possible": synthetic_possible,
            "recommended_template": template_fn(failure_type),
            "example_priority": priority,
        })

    rows.sort(key=lambda r: (-r["example_priority"], r["failure_type"]))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="phase3_analysis/golden_validation_predictions.jsonl")
    parser.add_argument("--logprob", default="phase3_analysis/min_logprob_summary.csv")
    parser.add_argument("--output-bit", default="phase3_analysis/failure_cases_bit_manipulation.csv")
    parser.add_argument("--output-numeral", default="phase3_analysis/failure_cases_numeral_conversion.csv")
    parser.add_argument("--failure-type-summary", default="phase3_analysis/failure_type_summary.csv")
    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions))
    logprob_map = load_logprob(Path(args.logprob))

    bit_preds = [r for r in predictions if r.get("category") == "bit_manipulation"]
    num_preds = [r for r in predictions if r.get("category") == "numeral_conversion"]

    print(f"bit_manipulation predictions: {len(bit_preds)}")
    print(f"numeral_conversion predictions: {len(num_preds)}")

    bit_rows = process_category(
        "bit_manipulation", bit_preds, logprob_map,
        classify_bit_failure, recommended_template_bit,
    )
    num_rows = process_category(
        "numeral_conversion", num_preds, logprob_map,
        classify_numeral_failure, recommended_template_numeral,
    )

    write_failure_csv(bit_rows, Path(args.output_bit))
    write_failure_csv(num_rows, Path(args.output_numeral))

    # Append to failure_type_summary (file may already have cryptarithm rows)
    summary_path = Path(args.failure_type_summary)
    append_failure_type_summary(bit_rows, summary_path, "bit_manipulation")
    append_failure_type_summary(num_rows, summary_path, "numeral_conversion")

    for category, rows in [("bit_manipulation", bit_rows), ("numeral_conversion", num_rows)]:
        counts = Counter(r["failure_type"] for r in rows)
        print(f"\n{category} failure types ({len(rows)} wrong cases):")
        for ft, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / max(len(rows), 1)
            print(f"  {ft:35s} {count:4d} ({pct:.1f}%)")

    print(f"\nWrote bit    -> {args.output_bit}")
    print(f"Wrote numeral -> {args.output_numeral}")


if __name__ == "__main__":
    main()
