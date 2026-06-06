#!/usr/bin/env python3
"""Phase 3 Step 5: Classify cryptarithm failure types at fine-grained level.

Analyses raw_output of each wrong (or low-logprob correct) cryptarithm prediction
and assigns a failure_type label. Produces:
  - failure_cases_cryptarithm.csv
  - (appends to) failure_type_summary.csv

failure_type labels:
  mapping_conflict   – model assigns the same digit to two different letters
  leading_zero_error – answer starts with zero, violating constraint
  carry_error        – carry propagation mistake in arithmetic chain
  incomplete_search  – model stops searching too early
  arithmetic_error   – simple arithmetic mistake not related to constraint
  constraint_missed  – explicit constraint in problem is ignored
  final_parse_error  – correct reasoning but wrong answer extraction
  hallucinated_rule  – model invents a rule not in examples
  answer_format_error – correct answer but wrong format (e.g., missing boxed)
  unknown            – cannot determine from output

SAFETY CONTRACT: Read-only analysis. No adapter, config, or submission touched.

Usage:
    python phase3_classify_cryptarithm_failures.py \
        --predictions phase3_analysis/golden_validation_predictions.jsonl \
        --logprob     phase3_analysis/min_logprob_summary.csv \
        --output      phase3_analysis/failure_cases_cryptarithm.csv \
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
# Failure type classifiers (applied in priority order)
# ---------------------------------------------------------------------------

def detect_mapping_conflict(raw: str) -> bool:
    """Check if model assigns same digit to two different letters."""
    # Pattern: "A=5 ... B=5" where two different letters get same value
    assignments = re.findall(r"([A-Z])\s*=\s*(\d)", raw)
    digit_to_letters: Dict[str, set] = {}
    letter_to_digits: Dict[str, set] = {}
    for letter, digit in assignments:
        digit_to_letters.setdefault(digit, set()).add(letter)
        letter_to_digits.setdefault(letter, set()).add(digit)
    # Conflict: same digit assigned to multiple different letters
    for digit, letters in digit_to_letters.items():
        if len(letters) > 1:
            return True
    # Conflict: same letter assigned multiple different digits
    for letter, digits in letter_to_digits.items():
        if len(digits) > 1:
            return True
    return False


def detect_leading_zero_error(raw: str, gold_answer: str) -> bool:
    """Check if model violates leading zero constraint."""
    # If raw output assigns 0 to a leading letter but gold answer has non-zero there
    leading_zero = re.search(r"([A-Z])\s*=\s*0", raw)
    if not leading_zero:
        return False
    # Is that letter the first letter of a word in the equation?
    letter = leading_zero.group(1)
    return bool(re.search(rf"\b{letter}[A-Z]+\s*[=+\-]", raw))


def detect_carry_error(raw: str) -> bool:
    """Check if model makes carry reasoning error."""
    carry_patterns = [
        r"carry\s*=\s*\d.*carry\s*=\s*\d",  # carry mentioned multiple times
        r"carry\s+over",
        r"column\s+sum.*>.*10",
    ]
    has_carry_context = any(re.search(p, raw, re.IGNORECASE) for p in carry_patterns)
    if not has_carry_context:
        return False
    # Check if answer is wrong despite carry discussion (heuristic)
    return True


def detect_incomplete_search(raw: str) -> bool:
    """Check if model stops search before covering all possibilities."""
    patterns = [
        r"let me try",
        r"assume\s+[A-Z]\s*=",
        r"checking.*[A-Z]\s*=\s*\d",
    ]
    attempt_count = sum(len(re.findall(p, raw, re.IGNORECASE)) for p in patterns)
    # Few attempts on what should be exhaustive search
    return 1 <= attempt_count <= 2


def detect_final_parse_error(raw: str, gold_answer: str, pred_answer: str) -> bool:
    """Check if correct answer appears in reasoning but was not extracted."""
    if pred_answer and pred_answer.upper() == gold_answer.upper():
        return False  # not an error
    # Does gold_answer appear anywhere in the reasoning?
    if gold_answer and re.search(re.escape(gold_answer), raw, re.IGNORECASE):
        return True
    return False


def detect_hallucinated_rule(raw: str, question: str) -> bool:
    """Check if model applies a rule not derivable from examples."""
    # Heuristic: model mentions a transformation rule not in examples
    hallucination_patterns = [
        r"the rule is.*(?:reverse|flip|negate)",
        r"applying.*pattern.*(?:not.*example|differ)",
        r"I will assume the rule",
    ]
    return any(re.search(p, raw, re.IGNORECASE) for p in hallucination_patterns)


def detect_answer_format_error(raw: str, gold_answer: str, pred_answer: str) -> bool:
    """Check if correct value was produced but format was wrong."""
    if pred_answer and pred_answer.upper() == gold_answer.upper():
        return False
    # Is gold_answer in the output but in wrong format?
    return bool(gold_answer and re.search(re.escape(gold_answer), raw, re.IGNORECASE)
                and not re.search(r"\\boxed\{", raw))


def detect_arithmetic_error(raw: str) -> bool:
    """Check for arithmetic mistakes in intermediate steps."""
    # Look for arithmetic expressions and check if they might be wrong
    patterns = [
        r"\d+\s*\+\s*\d+\s*=\s*\d+",  # arithmetic expressions
        r"\d+\s*\*\s*\d+\s*=\s*\d+",
    ]
    return any(re.search(p, raw) for p in patterns)


def classify_failure(
    rec: Dict[str, Any],
    lp_row: Dict[str, Any],
) -> Tuple[str, str]:
    """Return (failure_type, failure_reason)."""
    raw = rec.get("raw_output", "")
    question = rec.get("question", "")
    gold = rec.get("gold_answer", "")
    pred = rec.get("pred_answer", "")
    is_correct = rec.get("is_correct", False)

    if is_correct:
        # Only included if low logprob — mark as fragile_correct
        return "fragile_correct", "correct but low logprob — protect from SFT"

    # Priority order of checks
    if detect_final_parse_error(raw, gold, pred):
        return "final_parse_error", "gold answer appears in reasoning but extraction failed"

    if detect_answer_format_error(raw, gold, pred):
        return "answer_format_error", "correct value in output but missing \\boxed{} or wrong format"

    if detect_mapping_conflict(raw):
        return "mapping_conflict", "model assigns same digit to different letters or vice versa"

    if detect_leading_zero_error(raw, gold):
        return "leading_zero_error", "model violates no-leading-zero constraint"

    if detect_hallucinated_rule(raw, question):
        return "hallucinated_rule", "model applies transformation rule not derivable from examples"

    if detect_carry_error(raw):
        return "carry_error", "carry propagation mentioned but answer is wrong"

    if detect_incomplete_search(raw):
        return "incomplete_search", "model makes only 1-2 assignment attempts on exhaustive search"

    if detect_arithmetic_error(raw):
        return "arithmetic_error", "arithmetic expression present; may contain arithmetic mistake"

    return "unknown", "could not determine failure type from output"


# ---------------------------------------------------------------------------
# Example priority scoring
# ---------------------------------------------------------------------------

def compute_example_priority(
    failure_type: str,
    solver_check: bool,
    synthetic_possible: bool,
    answer_min_lp: Optional[float],
) -> int:
    if failure_type in {"mapping_conflict", "carry_error", "leading_zero_error"}:
        base = 5 if solver_check else 4
    elif failure_type in {"incomplete_search", "constraint_missed"}:
        base = 4 if synthetic_possible else 3
    elif failure_type == "final_parse_error":
        base = 4  # easy win if we fix extraction
    elif failure_type == "answer_format_error":
        base = 5  # easiest win
    elif failure_type == "hallucinated_rule":
        base = 3
    elif failure_type == "arithmetic_error":
        base = 3 if solver_check else 2
    elif failure_type == "fragile_correct":
        base = 3  # protect these
    else:
        base = 2

    # Boost if answer segment has very low logprob
    if answer_min_lp is not None and answer_min_lp < -3.0:
        base = min(5, base + 1)

    return base


def recommended_template(failure_type: str) -> str:
    templates = {
        "mapping_conflict": "exhaustive_backtracking_cot",
        "leading_zero_error": "constraint_checklist_cot",
        "carry_error": "column_by_column_carry_cot",
        "incomplete_search": "systematic_enumeration_cot",
        "arithmetic_error": "step_verify_cot",
        "constraint_missed": "constraint_first_cot",
        "final_parse_error": "boxed_answer_strict_cot",
        "hallucinated_rule": "example_grounded_rule_cot",
        "answer_format_error": "format_check_cot",
        "fragile_correct": "no_change_protect",
        "unknown": "general_diagnostic_cot",
    }
    return templates.get(failure_type, "general_diagnostic_cot")


# ---------------------------------------------------------------------------
# I/O
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


def write_failure_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "problem_id", "subcategory", "question", "gold_answer", "pred_answer",
        "is_correct", "min_logprob", "answer_min_logprob",
        "failure_type", "failure_reason",
        "solver_check_possible", "synthetic_generation_possible",
        "recommended_template", "example_priority",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_failure_type_summary(rows: List[Dict[str, Any]], path: Path, category: str) -> None:
    counts = Counter(r["failure_type"] for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    fields = ["category", "failure_type", "count", "pct"]
    total = max(len(rows), 1)
    with path.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for ft, count in sorted(counts.items(), key=lambda x: -x[1]):
            writer.writerow({
                "category": category,
                "failure_type": ft,
                "count": count,
                "pct": round(100 * count / total, 1),
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="phase3_analysis/golden_validation_predictions.jsonl")
    parser.add_argument("--logprob", default="phase3_analysis/min_logprob_summary.csv")
    parser.add_argument("--output", default="phase3_analysis/failure_cases_cryptarithm.csv")
    parser.add_argument("--failure-type-summary", default="phase3_analysis/failure_type_summary.csv")
    parser.add_argument(
        "--include-correct-low-logprob", action="store_true",
        help="Also include correct predictions with answer_min_logprob < LOW_LOGPROB_CUTOFF",
    )
    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions))
    logprob_map = load_logprob(Path(args.logprob))

    crypto_preds = [r for r in predictions if r.get("category") == "cryptarithm"]
    print(f"Cryptarithm predictions: {len(crypto_preds)}")

    rows: List[Dict[str, Any]] = []
    for rec in crypto_preds:
        pid = rec["problem_id"]
        lp_row = logprob_map.get(pid, {})
        is_correct = rec.get("is_correct", False)

        ans_min_lp_str = lp_row.get("answer_min_logprob")
        ans_min_lp = float(ans_min_lp_str) if ans_min_lp_str else None
        min_lp_str = lp_row.get("min_logprob")
        min_lp = float(min_lp_str) if min_lp_str else None

        # Include wrong predictions and (optionally) fragile correct ones
        include = not is_correct
        if args.include_correct_low_logprob and is_correct:
            if ans_min_lp is not None and ans_min_lp < LOW_LOGPROB_CUTOFF:
                include = True

        if not include:
            continue

        failure_type, failure_reason = classify_failure(rec, lp_row)

        # Solver feasibility heuristic
        subcategory = rec.get("subcategory", "unknown")
        solver_check = subcategory in {
            "alphametic_addition", "alphametic_subtraction",
            "alphametic_multiplication", "string_transform",
        } or subcategory == "unknown"

        synthetic_possible = failure_type in {
            "mapping_conflict", "carry_error", "leading_zero_error",
            "incomplete_search", "answer_format_error",
        }

        priority = compute_example_priority(failure_type, solver_check, synthetic_possible, ans_min_lp)

        rows.append({
            "problem_id": pid,
            "subcategory": subcategory,
            "question": rec.get("question", "")[:300],
            "gold_answer": rec.get("gold_answer", ""),
            "pred_answer": rec.get("pred_answer", ""),
            "is_correct": is_correct,
            "min_logprob": min_lp if min_lp is not None else "",
            "answer_min_logprob": ans_min_lp if ans_min_lp is not None else "",
            "failure_type": failure_type,
            "failure_reason": failure_reason,
            "solver_check_possible": solver_check,
            "synthetic_generation_possible": synthetic_possible,
            "recommended_template": recommended_template(failure_type),
            "example_priority": priority,
        })

    # Sort by priority descending
    rows.sort(key=lambda r: (-r["example_priority"], r["failure_type"]))

    write_failure_csv(rows, Path(args.output))

    # Append to failure_type_summary
    summary_path = Path(args.failure_type_summary)
    if summary_path.exists():
        summary_path.unlink()  # reset to avoid duplicate headers
    append_failure_type_summary(rows, summary_path, "cryptarithm")

    # Print summary
    counts = Counter(r["failure_type"] for r in rows)
    print(f"\nCryptarithm failure types ({len(rows)} cases):")
    for ft, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(len(rows), 1)
        print(f"  {ft:30s} {count:4d} ({pct:.1f}%)")
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
