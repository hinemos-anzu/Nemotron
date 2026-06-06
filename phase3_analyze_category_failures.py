#!/usr/bin/env python3
"""Phase 3 Step 4: Aggregate failures by category and produce priority scores.

Reads:
  - golden_validation_predictions.jsonl
  - min_logprob_summary.csv

Writes:
  - category_failure_summary.csv

Priority score logic (1–5):
  5 = large failure count + low accuracy + low logprob (most improvable)
  4 = moderate failure count or low accuracy
  3 = small count or mixed signals
  2 = few failures or high accuracy
  1 = near-perfect or tiny count

SAFETY CONTRACT: Read-only analysis script. No adapter or training data modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOW_LOGPROB_CUTOFF = -2.0  # threshold for "low confidence" classification


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_logprob_summary(path: Path) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            result[row["problem_id"]] = row
    return result


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------

def compute_priority(
    n: int,
    accuracy: float,
    n_wrong: int,
    n_wrong_low_conf: int,
    category: str,
) -> Tuple[int, str]:
    """Return (priority_score, priority_reason)."""
    reasons: List[str] = []

    # Solver-verifiable categories get a bonus
    solver_bonus = category in {"cryptarithm", "bit_manipulation", "numeral_conversion",
                                "arithmetic", "equation"}

    score = 0

    # Rule 1: volume
    if n >= 100:
        score += 2
        reasons.append("high_volume")
    elif n >= 30:
        score += 1
        reasons.append("moderate_volume")

    # Rule 2: failure rate
    if accuracy < 0.70:
        score += 2
        reasons.append("low_accuracy")
    elif accuracy < 0.85:
        score += 1
        reasons.append("moderate_accuracy_gap")

    # Rule 3: wrong + low-confidence concentration
    if n_wrong > 0:
        low_conf_ratio = n_wrong_low_conf / n_wrong
        if low_conf_ratio >= 0.5:
            score += 1
            reasons.append("wrong_low_conf_concentrated")

    # Rule 4: solver/synthetic feasibility bonus
    if solver_bonus:
        score += 1
        reasons.append("solver_verifiable")

    priority = max(1, min(5, score))
    return priority, "+".join(reasons) if reasons else "no_clear_priority_signal"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    predictions: List[Dict[str, Any]],
    logprob_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Group by (category, subcategory)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in predictions:
        key = (rec.get("category", "other"), rec.get("subcategory", "unknown"))
        groups[key].append(rec)

    rows: List[Dict[str, Any]] = []
    for (cat, sub), grp in sorted(groups.items()):
        n = len(grp)
        correct = sum(1 for r in grp if r.get("is_correct", False))
        accuracy = correct / max(n, 1)

        # Min logprob from logprob_map
        min_lps: List[float] = []
        ans_min_lps: List[float] = []
        n_wrong_low_conf = 0
        n_wrong_high_conf = 0
        n_correct_low_conf = 0

        for rec in grp:
            pid = rec["problem_id"]
            lp_row = logprob_map.get(pid, {})
            is_correct = rec.get("is_correct", False)

            min_lp_str = lp_row.get("min_logprob") or lp_row.get("answer_min_logprob")
            min_lp = float(min_lp_str) if min_lp_str else None

            ans_lp_str = lp_row.get("answer_min_logprob")
            ans_lp = float(ans_lp_str) if ans_lp_str else None

            if min_lp is not None:
                min_lps.append(min_lp)
            if ans_lp is not None:
                ans_min_lps.append(ans_lp)

            if not is_correct:
                if min_lp is not None and min_lp < LOW_LOGPROB_CUTOFF:
                    n_wrong_low_conf += 1
                elif min_lp is not None:
                    n_wrong_high_conf += 1
                else:
                    # No logprob data — count as unknown, assign to wrong_high_conf bucket
                    n_wrong_high_conf += 1
            else:
                if min_lp is not None and min_lp < LOW_LOGPROB_CUTOFF:
                    n_correct_low_conf += 1

        avg_min_lp = sum(min_lps) / len(min_lps) if min_lps else None
        avg_ans_min_lp = sum(ans_min_lps) / len(ans_min_lps) if ans_min_lps else None

        priority, priority_reason = compute_priority(
            n=n,
            accuracy=accuracy,
            n_wrong=n - correct,
            n_wrong_low_conf=n_wrong_low_conf,
            category=cat,
        )

        rows.append({
            "category": cat,
            "subcategory": sub,
            "n": n,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "avg_min_logprob": round(avg_min_lp, 4) if avg_min_lp is not None else "",
            "avg_answer_min_logprob": round(avg_ans_min_lp, 4) if avg_ans_min_lp is not None else "",
            "n_wrong_low_conf": n_wrong_low_conf,
            "n_wrong_high_conf": n_wrong_high_conf,
            "n_correct_low_conf": n_correct_low_conf,
            "priority_score": priority,
            "priority_reason": priority_reason,
        })

    # Sort by priority desc, then accuracy asc
    rows.sort(key=lambda r: (-r["priority_score"], r["accuracy"]))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="phase3_analysis/golden_validation_predictions.jsonl",
    )
    parser.add_argument(
        "--logprob",
        default="phase3_analysis/min_logprob_summary.csv",
    )
    parser.add_argument(
        "--output",
        default="phase3_analysis/category_failure_summary.csv",
    )
    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions))
    logprob_map = load_logprob_summary(Path(args.logprob))

    if not predictions:
        print("[WARN] No predictions loaded. Output will be empty.")

    rows = aggregate(predictions, logprob_map)

    fields = [
        "category", "subcategory", "n", "correct", "accuracy",
        "avg_min_logprob", "avg_answer_min_logprob",
        "n_wrong_low_conf", "n_wrong_high_conf", "n_correct_low_conf",
        "priority_score", "priority_reason",
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print("\nTop priority categories:")
    for row in rows[:10]:
        print(f"  [{row['priority_score']}] {row['category']}/{row['subcategory']}: "
              f"n={row['n']} acc={row['accuracy']} reason={row['priority_reason']}")


if __name__ == "__main__":
    main()
