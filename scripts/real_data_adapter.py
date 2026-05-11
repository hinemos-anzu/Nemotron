"""
real_data_adapter.py

Adapts real Kaggle competition train.csv (id, prompt, answer)
to the format expected by cryptarithm_solver pipeline.

Key differences vs synthetic data:
  - Field name: 'prompt' (not 'question')
  - No 'category' column → detected from prompt text
  - Test case format: "Now, determine the result for: X" (not "= ?")
  - Target category: 'equation' (≈ cryptarithm in internal naming)

Usage:
  python scripts/real_data_adapter.py --data /kaggle/input/<comp>/train.csv
  python scripts/real_data_adapter.py  # reads DATA_CSV env var
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from cryptarithm_solver import solve_batch, ALL_RULES, parse_examples
from category_solvers import solve_by_category

REPO_ROOT = _SCRIPTS.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "cryptarithm"

# ---------------------------------------------------------------------------
# Category detection from prompt text
# ---------------------------------------------------------------------------

def detect_category(prompt: str) -> str:
    p = prompt.lower()
    if "gravitational constant" in p or "d = 0.5*g" in p or "falling distance" in p:
        return "gravity"
    if "bit" in p and ("binary" in p or "xor" in p or "rotation" in p or "bitwise" in p):
        return "bit_manipulation"
    if "cipher" in p or ("encrypt" in p and "decrypt" in p):
        return "cipher"
    if "numeral system" in p or "roman" in p:
        return "numeral"
    if "unit conversion" in p or ("convert" in p and "becomes" in p):
        return "unit_conversion"
    if "transformation rules" in p or "determine the result for" in p:
        return "equation"
    return "other"


# ---------------------------------------------------------------------------
# Prompt normalization: convert real format → solver-compatible format
# ---------------------------------------------------------------------------

def normalize_prompt(prompt: str) -> str:
    """
    Convert real prompt to a format the solver can parse.

    Real format:
      <example_lines with "X OP Y = Z">
      Now, determine the result for: A OP B

    Solver expects:
      X OP Y = Z
      A OP B = ?
    """
    # Find test-case line
    m = re.search(
        r"Now,?\s*determine the result for:?\s*(.+?)(?:\n|$)",
        prompt,
        re.IGNORECASE,
    )
    if not m:
        # Already has "= ?" or no test case detectable
        return prompt

    test_input = m.group(1).strip()

    # Extract example lines (lines with " = " before the Now... line)
    before_now = prompt[: m.start()]
    ex_lines = [
        l.strip()
        for l in before_now.split("\n")
        if " = " in l and len(l.strip()) > 2
    ]

    normalized = "\n".join(ex_lines) + f"\n{test_input} = ?"
    return normalized


# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(dict(row))
    return records


def filter_target_category(records: List[Dict], target: str = "equation") -> List[Dict]:
    out = []
    for r in records:
        cat = detect_category(r.get("prompt", ""))
        r["category"] = cat
        if cat == target:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

def run(data_path: Path, target_cat: str = "equation") -> None:
    print(f"Loading: {data_path}", flush=True)
    records = load_csv(data_path)
    print(f"  Total rows: {len(records)}", flush=True)

    # Detect categories
    all_cats: Dict[str, int] = {}
    for r in records:
        c = detect_category(r.get("prompt", ""))
        r["category"] = c
        all_cats[c] = all_cats.get(c, 0) + 1

    print("\nCategory distribution:")
    for cat, cnt in sorted(all_cats.items(), key=lambda x: -x[1]):
        marker = " ← TARGET" if cat == target_cat else ""
        print(f"  {cat:30s}: {cnt}{marker}")

    # Filter to target category
    targets = [r for r in records if r["category"] == target_cat]
    print(f"\nTarget '{target_cat}' rows: {len(targets)}", flush=True)

    # Normalize prompts for solver
    for r in targets:
        r["question"] = normalize_prompt(r.get("prompt", ""))
        r["problem_id"] = r.get("id", "")
        r.setdefault("examples", "")

    # For equation: use cryptarithm solve_batch (concat rules)
    # For all other categories: use category-specific solver
    print("Running solver...", flush=True)
    solved = []
    if target_cat == "equation":
        solved = solve_batch(targets, question_key="question", answer_key="answer")
    else:
        for r in targets:
            prompt = r.get("prompt", "")
            expected = r.get("answer", "")
            predicted = solve_by_category(target_cat, prompt, expected)
            is_correct = predicted is not None and str(predicted).strip() == str(expected).strip()
            solved.append({
                **r,
                "solver_rule": target_cat + "_solver",
                "solver_predicted": predicted,
                "solver_correct": is_correct,
                "solver_parse_ok": predicted is not None,
                "solver_explanation": f"predicted={predicted}; expected={expected}",
            })

    correct = sum(1 for r in solved if r.get("solver_correct"))
    parse_ok = sum(1 for r in solved if r.get("solver_parse_ok"))
    print(f"  parse_ok   : {parse_ok}/{len(solved)}", flush=True)
    print(f"  correct    : {correct}/{len(solved)}", flush=True)

    # Rule distribution
    rule_counts: Dict[str, int] = {}
    rule_ok: Dict[str, int] = {}
    for r in solved:
        rule = r.get("solver_rule", "unknown")
        rule_counts[rule] = rule_counts.get(rule, 0) + 1
        if r.get("solver_correct"):
            rule_ok[rule] = rule_ok.get(rule, 0) + 1

    print("\nRule distribution on real data:")
    for rule, cnt in sorted(rule_counts.items(), key=lambda x: -x[1]):
        ok = rule_ok.get(rule, 0)
        print(f"  {rule:35s}: {cnt:4d}  correct={ok}")

    # Save coverage CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cov_path = OUTPUT_DIR / "real_data_solver_coverage.csv"
    with cov_path.open("w", newline="", encoding="utf-8") as fh:
        cols = ["problem_id", "category", "solver_rule", "solver_predicted",
                "solver_correct", "solver_parse_ok", "answer", "solver_explanation"]
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in solved:
            writer.writerow({c: r.get(c, "") for c in cols})
    print(f"\n  Saved → {cov_path}", flush=True)

    # Save verified CoT
    verified = [r for r in solved if r.get("solver_correct")]
    cot_path = OUTPUT_DIR / "real_data_verified_cot.jsonl"
    with cot_path.open("w", encoding="utf-8") as fh:
        for r in verified:
            fh.write(json.dumps({
                "problem_id": r.get("problem_id", ""),
                "category": r.get("category", ""),
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "rule": r.get("solver_rule", ""),
                "verified": True,
            }, ensure_ascii=False) + "\n")
    print(f"  Saved CoT → {cot_path}  ({len(verified)} records)", flush=True)

    # Summary
    print(f"\n=== Summary ===")
    print(f"  target_category : {target_cat}")
    print(f"  total_problems  : {len(targets)}")
    print(f"  parse_ok        : {parse_ok}")
    print(f"  solver_correct  : {correct}  ({correct/len(targets)*100:.1f}%)")
    print(f"  verified_cot    : {len(verified)}")

    top_unresolved = [
        (r, cnt - rule_ok.get(r, 0))
        for r, cnt in sorted(rule_counts.items(), key=lambda x: -(x[1] - rule_ok.get(x[0], 0)))
        if (cnt - rule_ok.get(r, 0)) > 0
    ]
    if top_unresolved:
        print("\n  Top unresolved rules:")
        for rule, miss in top_unresolved[:5]:
            print(f"    {rule:35s} unresolved={miss}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run solver on real Kaggle train.csv")
    parser.add_argument(
        "--data",
        default=os.environ.get("DATA_CSV", ""),
        help="Path to train.csv (or set DATA_CSV env var)",
    )
    parser.add_argument(
        "--category",
        default="equation",
        help="Target category to analyze (default: equation)",
    )
    args = parser.parse_args()

    if not args.data:
        print("ERROR: specify --data /path/to/train.csv or set DATA_CSV env var", file=sys.stderr)
        sys.exit(1)

    run(Path(args.data), target_cat=args.category)
