"""
run_all_categories.py

Runs all category solvers against train.csv in one pass and writes:
  reports/cryptarithm/all_categories_cot.jsonl   - combined verified CoT
  reports/cryptarithm/all_categories_coverage.csv - per-problem solver results
  reports/cryptarithm/all_categories_summary.md  - coverage table

CoT format per line:
  {
    "problem_id": ...,
    "category": ...,
    "prompt": ...,        <- original prompt (training input)
    "answer": ...,        <- ground-truth answer (training output)
    "cot": ...,           <- reasoning trace
    "rule": ...,
    "verified": true
  }

Usage:
    python scripts/run_all_categories.py \\
        --data /kaggle/input/.../train.csv

    # or set env var
    DATA_CSV=/path/to/train.csv python scripts/run_all_categories.py
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

from cryptarithm_solver import solve_batch
from category_solvers import solve_by_category
from real_data_adapter import (
    detect_category,
    normalize_prompt,
    _is_correct,
    _NUMERIC_CATS,
)

REPO_ROOT = _SCRIPTS.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "cryptarithm"
ALL_COT_PATH   = OUTPUT_DIR / "all_categories_cot.jsonl"
ALL_COV_PATH   = OUTPUT_DIR / "all_categories_coverage.csv"
SUMMARY_PATH   = OUTPUT_DIR / "all_categories_summary.md"

TARGET_CATS = [
    "gravity", "unit_conversion", "numeral",
    "cipher", "bit_manipulation", "equation",
]

# ---------------------------------------------------------------------------
# CoT builders per category
# ---------------------------------------------------------------------------

def _build_cot_gravity(prompt: str, answer: str) -> str:
    import re as _re
    ex = _re.findall(r"t\s*=\s*([\d.]+)\s*s[^,\n]*[,\s]+distance\s*=\s*([\d.]+)", prompt, _re.IGNORECASE)
    num = sum(float(d) * float(t)**2 for t, d in ex)
    den = sum(float(t)**4 for t, _ in ex)
    g = 2.0 * num / den if den else 0
    m = _re.search(r"for\s+t\s*=\s*([\d.]+)\s*s", prompt[prompt.lower().rfind("now"):], _re.IGNORECASE)
    t_test = float(m.group(1)) if m else 0
    lines = [
        "Step 1: Extract (t, distance) examples from the prompt.",
    ]
    for t, d in ex[:4]:
        lines.append(f"  t={t}s, d={d}m  →  g = 2×{d}/{float(t)**2:.4f} = {2*float(d)/float(t)**2:.4f}")
    lines.append(f"Step 2: Fit gravitational constant via least-squares.")
    lines.append(f"  g = 2×Σ(d·t²) / Σ(t⁴) = {g:.6f}")
    lines.append(f"Step 3: Compute falling distance for t = {t_test}s.")
    lines.append(f"  d = 0.5 × {g:.4f} × {t_test}² = 0.5 × {g:.4f} × {t_test**2:.4f}")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def _build_cot_unit_conversion(prompt: str, answer: str) -> str:
    import re as _re
    ex = _re.findall(r"([\d.]+)\s*\w*\s+becomes\s+([\d.]+)", prompt, _re.IGNORECASE)
    factors = [float(o)/float(i) for i, o in ex if float(i) > 0]
    factor = sum(factors) / len(factors) if factors else 0
    m = _re.search(r"convert.*?:\s*([\d.]+)", prompt[prompt.lower().rfind("now"):], _re.IGNORECASE)
    if not m:
        m = _re.search(r"convert.*?([\d.]+)\s*\w*\s*$", prompt[prompt.lower().rfind("now"):], _re.IGNORECASE)
    test_val = float(m.group(1)) if m else 0
    lines = ["Step 1: Extract (input, output) conversion examples."]
    for i, o in ex[:4]:
        lines.append(f"  {i} → {o}  factor = {float(o)/float(i):.6f}")
    lines.append(f"Step 2: Average factor = {factor:.6f}")
    lines.append(f"Step 3: Apply to test value {test_val}: {factor:.6f} × {test_val} = {factor*test_val:.4f}")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def _build_cot_numeral(prompt: str, answer: str) -> str:
    import re as _re
    ex = _re.findall(r"(\d+)\s*->\s*([A-Z]+)", prompt)
    lines = ["Step 1: Verify examples match Roman numeral system."]
    for n, r in ex[:4]:
        lines.append(f"  {n} → {r}")
    lines.append("Step 2: All examples confirm standard Roman numeral encoding.")
    m = _re.search(r"write the number\s+(\d+)", prompt, _re.IGNORECASE)
    if m:
        lines.append(f"Step 3: Convert {m.group(1)} to Roman numerals.")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def _build_cot_cipher(prompt: str, answer: str) -> str:
    import re as _re
    lines = ["Step 1: Extract character mapping from cipher examples."]
    now_idx = prompt.lower().rfind("now")
    ex_text = prompt[:now_idx] if now_idx >= 0 else prompt
    pairs = _re.findall(r"^(.+?)\s*->\s*(.+)$", ex_text, _re.MULTILINE)
    # Show a few mappings
    shown = 0
    for src, tgt in pairs[:2]:
        for sc, tc in zip(src.replace(" ", ""), tgt.replace(" ", "")):
            if shown < 6:
                lines.append(f"  cipher '{sc}' → plain '{tc}'")
                shown += 1
    lines.append("Step 2: Apply substitution table to test input.")
    m = _re.search(r":\s*(.+?)$", prompt[now_idx:].strip(), _re.DOTALL)
    if m:
        lines.append(f"  Input:  {m.group(1).strip()}")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def _build_cot_bit(prompt: str, answer: str, rule: str) -> str:
    import re as _re
    ex = _re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    lines = [f"Step 1: Observe input→output bit patterns."]
    for a, b in ex[:4]:
        lines.append(f"  {a} → {b}")
    lines.append(f"Step 2: Identified rule: {rule}")
    m = _re.search(r"determine the output for:?\s*([01]{8})", prompt, _re.IGNORECASE)
    if m:
        lines.append(f"Step 3: Apply rule to {m.group(1)}.")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def _build_cot_equation(prompt: str, answer: str, rule: str) -> str:
    normalized = normalize_prompt(prompt)
    lines = [
        "Step 1: Observe input→output transformation examples.",
        "Step 2: Identify the rule from examples.",
        f"  Rule identified: {rule}",
        "Step 3: Apply rule to test case.",
        f"Answer: {answer}",
    ]
    return "\n".join(lines)


def _build_cot(category: str, prompt: str, answer: str, rule: str) -> str:
    try:
        if category == "gravity":
            return _build_cot_gravity(prompt, answer)
        elif category == "unit_conversion":
            return _build_cot_unit_conversion(prompt, answer)
        elif category == "numeral":
            return _build_cot_numeral(prompt, answer)
        elif category == "cipher":
            return _build_cot_cipher(prompt, answer)
        elif category == "bit_manipulation":
            return _build_cot_bit(prompt, answer, rule)
        elif category == "equation":
            return _build_cot_equation(prompt, answer, rule)
    except Exception:
        pass
    return f"Rule: {rule}\nAnswer: {answer}"


# ---------------------------------------------------------------------------
# Main solver loop
# ---------------------------------------------------------------------------

_COV_COLS = [
    "problem_id", "category", "solver_rule",
    "solver_predicted", "solver_correct", "answer",
]


def run_all(data_path: Path) -> None:
    print(f"Loading: {data_path}", flush=True)
    rows: List[Dict] = []
    with data_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  Total rows: {len(rows)}", flush=True)

    # Attach category
    for r in rows:
        r["_cat"] = detect_category(r.get("prompt", ""))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cot_records: List[Dict] = []
    cov_records: List[Dict] = []
    stats: Dict[str, Dict] = {}

    for cat in TARGET_CATS:
        targets = [r for r in rows if r["_cat"] == cat]
        if not targets:
            continue

        correct_n = 0
        for r in targets:
            prompt   = r.get("prompt", "")
            expected = r.get("answer", "")
            pid      = r.get("id", "")

            if cat == "equation":
                # Use cryptarithm concat solver
                normalized = normalize_prompt(prompt)
                result = solve_batch(
                    [{"question": normalized, "answer": expected,
                      "category": cat, "problem_id": pid}],
                    question_key="question", answer_key="answer",
                )[0]
                predicted = result.get("solver_predicted")
                rule      = result.get("solver_rule", "unknown")
            else:
                predicted = solve_by_category(cat, prompt, expected)
                rule = cat + "_solver"

            correct = _is_correct(predicted, expected, cat)
            if correct:
                correct_n += 1

            cov_records.append({
                "problem_id":      pid,
                "category":        cat,
                "solver_rule":     rule,
                "solver_predicted": predicted,
                "solver_correct":  correct,
                "answer":          expected,
            })

            if correct:
                cot_text = _build_cot(cat, prompt, expected, rule)
                cot_records.append({
                    "problem_id": pid,
                    "category":   cat,
                    "prompt":     prompt,
                    "answer":     expected,
                    "cot":        cot_text,
                    "rule":       rule,
                    "verified":   True,
                })

        stats[cat] = {
            "total":   len(targets),
            "correct": correct_n,
            "pct":     correct_n / len(targets) * 100 if targets else 0,
        }
        print(f"  {cat:20s}: {correct_n:4d}/{len(targets):4d}  ({correct_n/len(targets)*100:.1f}%)", flush=True)

    # Write combined CoT JSONL
    with ALL_COT_PATH.open("w", encoding="utf-8") as fh:
        for rec in cot_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n  CoT JSONL → {ALL_COT_PATH}  ({len(cot_records)} records)", flush=True)

    # Write coverage CSV
    with ALL_COV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cov_records)
    print(f"  Coverage  → {ALL_COV_PATH}", flush=True)

    # Write summary markdown
    total_all = sum(v["total"] for v in stats.values())
    correct_all = sum(v["correct"] for v in stats.values())
    md_lines = [
        "# All-Category Solver Coverage Summary",
        "",
        "| Category | Total | Correct | Coverage |",
        "|---|---|---|---|",
    ]
    for cat in TARGET_CATS:
        if cat not in stats:
            continue
        s = stats[cat]
        md_lines.append(f"| {cat} | {s['total']} | {s['correct']} | {s['pct']:.1f}% |")
    md_lines += [
        f"| **TOTAL** | **{total_all}** | **{correct_all}** | **{correct_all/total_all*100:.1f}%** |",
        "",
        f"Verified CoT records: **{len(cot_records)}**",
        f"Saved to: `{ALL_COT_PATH.name}`",
    ]
    SUMMARY_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Summary   → {SUMMARY_PATH}", flush=True)
    print(f"\nTotal verified CoT: {len(cot_records)} / {total_all}  ({correct_all/total_all*100:.1f}%)", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.environ.get("DATA_CSV", ""),
                        help="Path to train.csv")
    args = parser.parse_args()
    if not args.data:
        print("ERROR: --data or DATA_CSV required", file=sys.stderr)
        sys.exit(1)
    run_all(Path(args.data))
