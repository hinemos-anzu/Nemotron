"""
cryptarithm_generate_verified_cot.py

For every cryptarithm problem where the deterministic solver matches the
ground-truth answer, generate a structured Chain-of-Thought (CoT) teaching
example and save it to reports/cryptarithm/cryptarithm_generated_cot_sample.jsonl.

Also writes:
  reports/cryptarithm/cryptarithm_solver_coverage.csv
  reports/cryptarithm/cryptarithm_failure_report.md

Reads data via the same loader as cryptarithm_inventory.py.
Falls back to synthetic examples when no data files are present.

CoT format (per line in JSONL):
  {
    "problem_id": ...,
    "category": ...,
    "question": ...,
    "answer": ...,
    "rule": ...,
    "cot": "Step 1: ...\nStep 2: ...\nAnswer: ...",
    "verified": true
  }
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Import sibling modules
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from cryptarithm_solver import (
    ALL_RULES,
    Example,
    SolveResult,
    parse_examples,
    parse_test_case,
    identify_rule,
    solve,
    solve_guess,
    solve_batch,
)
from cryptarithm_inventory import (
    CRYPTARITHM_CATEGORIES,
    DATA_DIR,
    SYNTHETIC_PROBLEMS,
    load_all_records,
    build_inventory,
    _field,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "reports" / "cryptarithm"
COVERAGE_CSV = OUTPUT_DIR / "cryptarithm_solver_coverage.csv"
COT_JSONL = OUTPUT_DIR / "cryptarithm_generated_cot_sample.jsonl"
FAILURE_MD = OUTPUT_DIR / "cryptarithm_failure_report.md"

# ---------------------------------------------------------------------------
# CoT template builders (one per rule class)
# ---------------------------------------------------------------------------

_RULE_DESC: Dict[str, str] = {
    "forward_concat":            "Concatenate left string followed by right string.",
    "reverse_concat":            "Concatenate right string followed by left string.",
    "swap_left_right":           "Swap operands: output is right then left.",
    "reverse_left":              "Reverse the left operand, then append the right operand.",
    "reverse_right":             "Append the right operand reversed after the left operand.",
    "reverse_both":              "Reverse both operands: rev(left) + rev(right).",
    "interleave_lr":             "Interleave characters starting from the left operand.",
    "interleave_rl":             "Interleave characters starting from the right operand.",
    "operator_conditioned_rule": "Apply a sub-rule determined by the operator symbol.",
    "unknown_operator_fallback": "No matching deterministic rule found.",
}


def _fmt_interleave(left: str, right: str, lr: bool) -> str:
    """Show interleave trace: A[0]B[0]A[1]B[1]... or B[0]A[0]..."""
    a, b = (left, right) if lr else (right, left)
    an, bn = ("L", "R") if lr else ("R", "L")
    parts = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            parts.append(f"{an}[{i}]={a[i]}")
        if i < len(b):
            parts.append(f"{bn}[{i}]={b[i]}")
    return ", ".join(parts)


def build_cot(
    problem_id: str,
    category: str,
    question: str,
    answer: str,
    rule_name: str,
    predicted: str,
    examples: List[Example],
    test_left: str,
    test_op: str,
    test_right: str,
) -> str:
    """
    Build a human-readable CoT string for one verified problem.
    Each step is a single semantic action.
    """
    lines: List[str] = []

    # --- Step 1: read examples ---
    lines.append("Step 1: Observe the input-output examples.")
    for i, ex in enumerate(examples, 1):
        lines.append(f"  Example {i}: {ex.left} {ex.op} {ex.right} = {ex.result}")

    # --- Step 2: identify the rule ---
    lines.append(f"Step 2: Identify the rule.")
    lines.append(f"  Checking rule '{rule_name}': {_RULE_DESC.get(rule_name, '')}")

    # Rule-specific verification trace
    for i, ex in enumerate(examples, 1):
        fn = ALL_RULES.get(rule_name)
        if fn:
            try:
                chk = fn(ex.left, ex.op, ex.right)
            except Exception:
                chk = "error"
        else:
            chk = "n/a"
        match = "✓" if chk == ex.result else "✗"
        lines.append(f"    Example {i}: apply({ex.left}, {ex.op}, {ex.right}) → {chk}  {match}")

    lines.append(f"  All examples match rule '{rule_name}'. Rule confirmed.")

    # --- Step 3: apply the rule to the test case ---
    lines.append(f"Step 3: Apply rule '{rule_name}' to the test case.")
    lines.append(f"  Test input: {test_left} {test_op} {test_right} = ?")

    if rule_name == "forward_concat":
        lines.append(f"  Concatenate left '{test_left}' then right '{test_right}'.")
        lines.append(f"  Result: '{test_left}' + '{test_right}' = '{predicted}'")

    elif rule_name in ("reverse_concat", "swap_left_right"):
        lines.append(f"  Concatenate right '{test_right}' then left '{test_left}'.")
        lines.append(f"  Result: '{test_right}' + '{test_left}' = '{predicted}'")

    elif rule_name == "reverse_left":
        rev_l = test_left[::-1]
        lines.append(f"  Reverse left: '{test_left}' → '{rev_l}'.")
        lines.append(f"  Append right: '{rev_l}' + '{test_right}' = '{predicted}'")

    elif rule_name == "reverse_right":
        rev_r = test_right[::-1]
        lines.append(f"  Keep left: '{test_left}'.")
        lines.append(f"  Reverse right: '{test_right}' → '{rev_r}'.")
        lines.append(f"  Concatenate: '{test_left}' + '{rev_r}' = '{predicted}'")

    elif rule_name == "reverse_both":
        rev_l = test_left[::-1]
        rev_r = test_right[::-1]
        lines.append(f"  Reverse left: '{test_left}' → '{rev_l}'.")
        lines.append(f"  Reverse right: '{test_right}' → '{rev_r}'.")
        lines.append(f"  Concatenate: '{rev_l}' + '{rev_r}' = '{predicted}'")

    elif rule_name == "interleave_lr":
        trace = _fmt_interleave(test_left, test_right, lr=True)
        lines.append(f"  Interleave (left-first): {trace}")
        lines.append(f"  Result: '{predicted}'")

    elif rule_name == "interleave_rl":
        trace = _fmt_interleave(test_left, test_right, lr=False)
        lines.append(f"  Interleave (right-first): {trace}")
        lines.append(f"  Result: '{predicted}'")

    elif rule_name == "operator_conditioned_rule":
        lines.append(f"  Operator '{test_op}' maps to sub-rule. Result: '{predicted}'")

    else:
        lines.append(f"  Applied rule → '{predicted}'")

    # --- Final answer ---
    lines.append(f"Answer: {predicted}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Solver coverage runner
# ---------------------------------------------------------------------------

COVERAGE_COLS = [
    "problem_id", "category", "solver_rule", "solver_predicted",
    "solver_correct", "solver_parse_ok", "answer", "solver_explanation",
]


def run_solver_coverage(records: List[Dict]) -> List[Dict]:
    """Apply solve() to every cryptarithm record. Return augmented records."""
    crypto = [r for r in records if _field(r, ["category", "task_type", "type"]) in CRYPTARITHM_CATEGORIES]
    return solve_batch(crypto)


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def analyse_failures(solved: List[Dict]) -> Dict:
    total_deduce = sum(1 for r in solved if r.get("category") == "cryptarithm_deduce")
    total_guess = sum(1 for r in solved if r.get("category") == "cryptarithm_guess")

    correct_deduce = sum(
        1 for r in solved
        if r.get("category") == "cryptarithm_deduce" and r.get("solver_correct")
    )
    correct_guess = sum(
        1 for r in solved
        if r.get("category") == "cryptarithm_guess" and r.get("solver_correct")
    )

    rule_counts: Dict[str, int] = defaultdict(int)
    rule_correct: Dict[str, int] = defaultdict(int)
    for r in solved:
        rule = r.get("solver_rule", "unknown")
        rule_counts[rule] += 1
        if r.get("solver_correct"):
            rule_correct[rule] += 1

    failures = [r for r in solved if not r.get("solver_correct")]
    parse_fails = [r for r in solved if not r.get("solver_parse_ok")]
    answer_mismatches = [
        r for r in solved
        if r.get("solver_parse_ok") and not r.get("solver_correct") and r.get("solver_predicted")
    ]

    return {
        "total_deduce": total_deduce,
        "total_guess": total_guess,
        "correct_deduce": correct_deduce,
        "correct_guess": correct_guess,
        "rule_counts": dict(rule_counts),
        "rule_correct": dict(rule_correct),
        "failures": failures,
        "parse_fails": parse_fails,
        "answer_mismatches": answer_mismatches,
    }


# ---------------------------------------------------------------------------
# CoT generation
# ---------------------------------------------------------------------------

def build_cot_guess(rule_name: str, examples: List[Example], predicted: str) -> str:
    """CoT for cryptarithm_guess: show verification of rule against examples."""
    lines = [
        "Step 1: Observe the input-output examples.",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"  Example {i}: {ex.left} {ex.op} {ex.right} = {ex.result}")

    lines.append("Step 2: Test candidate rules against all examples.")
    fn = ALL_RULES.get(rule_name)
    for i, ex in enumerate(examples, 1):
        chk = fn(ex.left, ex.op, ex.right) if fn else "n/a"
        match = "✓" if chk == ex.result else "✗"
        lines.append(f"  Example {i}: {rule_name}({ex.left}, {ex.op}, {ex.right}) → {chk}  {match}")

    lines.append(f"Step 3: All examples match '{rule_name}'.")
    desc = _RULE_DESC.get(rule_name, "")
    if desc:
        lines.append(f"  Description: {desc}")
    lines.append(f"Answer: {predicted}")
    return "\n".join(lines)


def generate_verified_cot_records(solved: List[Dict]) -> List[Dict]:
    """Return CoT dicts only for problems where solver_correct is True."""
    cot_records = []
    for rec in solved:
        if not rec.get("solver_correct"):
            continue
        question = _field(rec, ["question", "input", "prompt"])
        answer = _field(rec, ["answer", "output", "label", "target"])
        rule_name = rec.get("solver_rule", "unknown")
        predicted = rec.get("solver_predicted", "")
        category = rec.get("category", "")

        examples = parse_examples(question)

        if category == "cryptarithm_guess":
            cot_text = build_cot_guess(rule_name, examples, predicted)
        else:
            test_case = parse_test_case(question)
            if test_case is None:
                cot_text = (
                    f"Step 1: Observe examples.\n"
                    f"Step 2: No test case found; rule identified from context: '{rule_name}'.\n"
                    f"Answer: {predicted}"
                )
            else:
                tl, top, tr = test_case
                cot_text = build_cot(
                    problem_id=_field(rec, ["id", "problem_id", "ID"]),
                    category=category,
                    question=question,
                    answer=answer,
                    rule_name=rule_name,
                    predicted=predicted,
                    examples=examples,
                    test_left=tl,
                    test_op=top,
                    test_right=tr,
                )

        cot_records.append({
            "problem_id": _field(rec, ["id", "problem_id", "ID"]),
            "category": category,
            "question": question,
            "answer": answer,
            "rule": rule_name,
            "cot": cot_text,
            "verified": True,
        })
    return cot_records


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_coverage_csv(solved: List[Dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with COVERAGE_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COVERAGE_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in solved:
            writer.writerow({col: r.get(col, "") for col in COVERAGE_COLS})
    print(f"  Saved coverage CSV → {COVERAGE_CSV}", flush=True)


def write_cot_jsonl(cot_records: List[Dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with COT_JSONL.open("w", encoding="utf-8") as fh:
        for rec in cot_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved CoT JSONL   → {COT_JSONL}  ({len(cot_records)} records)", flush=True)


def write_failure_report(analysis: Dict, total_solved: int, cot_count: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    td = analysis["total_deduce"]
    tg = analysis["total_guess"]
    cd = analysis["correct_deduce"]
    cg = analysis["correct_guess"]

    rule_counts = analysis["rule_counts"]
    rule_correct = analysis["rule_correct"]
    failures = analysis["failures"]
    parse_fails = analysis["parse_fails"]
    answer_mismatches = analysis["answer_mismatches"]

    # Rule coverage table
    rule_rows = []
    for rule, cnt in sorted(rule_counts.items(), key=lambda x: -x[1]):
        ok = rule_correct.get(rule, 0)
        rule_rows.append((rule, cnt, ok, f"{ok/cnt*100:.0f}%" if cnt else "n/a"))

    # Top unresolved rules
    unresolved = [
        (rule, cnt)
        for rule, cnt in sorted(rule_counts.items(), key=lambda x: -x[1])
        if rule_correct.get(rule, 0) < cnt
    ]

    # Suggest next templates
    needs_template = [
        rule for rule, cnt in unresolved
        if rule == "unknown_operator_fallback" or rule_correct.get(rule, 0) == 0
    ]

    lines = [
        "# Cryptarithm Solver Failure Report",
        "",
        "## 1. Problem count by category",
        "",
        f"| Category | Total | Solver correct | Coverage |",
        f"|---|---|---|---|",
        f"| cryptarithm_deduce | {td} | {cd} | {cd/td*100:.1f}% |" if td else "| cryptarithm_deduce | 0 | 0 | n/a |",
        f"| cryptarithm_guess  | {tg} | {cg} | {cg/tg*100:.1f}% |" if tg else "| cryptarithm_guess  | 0 | 0 | n/a |",
        f"| **Total** | **{td+tg}** | **{cd+cg}** | **{(cd+cg)/(td+tg)*100:.1f}%** |" if (td+tg) else "| **Total** | **0** | **0** | **n/a** |",
        "",
        "## 2. Rule coverage breakdown",
        "",
        "| Rule | Problems matched | Correct | Accuracy |",
        "|---|---|---|---|",
    ]
    for rule, cnt, ok, acc in rule_rows:
        lines.append(f"| {rule} | {cnt} | {ok} | {acc} |")

    lines += [
        "",
        "## 3. Parse failures",
        "",
        f"Total parse failures: {len(parse_fails)}",
        "",
    ]
    for r in parse_fails[:5]:
        pid = _field(r, ["id", "problem_id", "ID"])
        lines.append(f"- `{pid}`: {r.get('solver_explanation', '')[:120]}")

    lines += [
        "",
        "## 4. Answer mismatch examples (solver parsed but predicted wrong)",
        "",
    ]
    if answer_mismatches:
        lines.append("| problem_id | category | predicted | expected | rule |")
        lines.append("|---|---|---|---|---|")
        for r in answer_mismatches[:10]:
            pid = _field(r, ["id", "problem_id", "ID"])
            cat = r.get("category", "")
            pred = r.get("solver_predicted", "")
            exp = _field(r, ["answer", "output", "label", "target"])
            rule = r.get("solver_rule", "")
            lines.append(f"| {pid} | {cat} | {pred} | {exp} | {rule} |")
    else:
        lines.append("_No answer mismatches (all parsed problems solved correctly or fell through to unknown)._")

    lines += [
        "",
        "## 5. Unresolved rules (not matching any example set)",
        "",
    ]
    if unresolved:
        for rule, cnt in unresolved:
            ok = rule_correct.get(rule, 0)
            lines.append(f"- **{rule}**: {cnt} problems, {ok} correct, {cnt-ok} unresolved")
    else:
        lines.append("_All rules matched at least one problem correctly._")

    lines += [
        "",
        "## 6. Next templates to add",
        "",
    ]
    suggested = [
        ("digit_sum_concat", "Combine digit sums of left and right"),
        ("mod_concat", "Apply modulo then concatenate"),
        ("length_conditioned", "Rule changes based on string length of operands"),
        ("operator_symbol_map", "Map operator to a named operation (e.g. ★→XOR)"),
        ("numeric_arithmetic_fallback", "Evaluate as standard arithmetic before string rules"),
    ]
    if needs_template:
        lines.append(
            f"The following rules returned no matches and likely need new templates: "
            f"`{'`, `'.join(needs_template)}`."
        )
        lines.append("")
    lines.append("Suggested additions based on failure patterns:")
    lines.append("")
    for name, desc in suggested:
        lines.append(f"- **{name}**: {desc}")

    lines += [
        "",
        "## 7. Verified CoT count",
        "",
        f"Verified CoT records generated: **{cot_count}**",
        f"(saved to `reports/cryptarithm/cryptarithm_generated_cot_sample.jsonl`)",
        "",
        "## 8. Recommendation: proceed to training?",
        "",
    ]

    total = td + tg
    coverage = (cd + cg) / total if total else 0
    if coverage >= 0.70:
        lines.append(
            f"**YES** — Solver coverage is {coverage*100:.1f}% (≥70%). "
            "Sufficient verified CoT data to proceed with SFT data preparation. "
            "Recommend: run `train_sft.py` on `cryptarithm_generated_cot_sample.jsonl` "
            "after Planner review."
        )
    elif coverage >= 0.40:
        lines.append(
            f"**CONDITIONAL** — Solver coverage is {coverage*100:.1f}% (40-70%). "
            "Acceptable for a first training run but add missing templates first. "
            "Address top unresolved rules before scaling up."
        )
    else:
        lines.append(
            f"**NO** — Solver coverage is {coverage*100:.1f}% (<40%). "
            "Too many problems are unverified. Expand solver rules before training."
        )

    FAILURE_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved failure report → {FAILURE_MD}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== cryptarithm_generate_verified_cot ===", flush=True)
    print(f"DATA_DIR : {DATA_DIR}", flush=True)

    records, corpus_ids = load_all_records(DATA_DIR)

    # Tag source
    for r in records:
        if "_source_file" not in r:
            r["_source_file"] = r.get("__source__", "synthetic_or_unknown")

    print("\n[1] Running solver coverage...", flush=True)
    solved = run_solver_coverage(records)

    print(f"  Total cryptarithm problems : {len(solved)}", flush=True)
    correct = sum(1 for r in solved if r.get("solver_correct"))
    print(f"  Solver correct             : {correct}", flush=True)

    print("\n[2] Writing solver coverage CSV...", flush=True)
    write_coverage_csv(solved)

    print("\n[3] Generating verified CoT records...", flush=True)
    cot_records = generate_verified_cot_records(solved)
    write_cot_jsonl(cot_records)

    print("\n[4] Analysing failures and writing report...", flush=True)
    analysis = analyse_failures(solved)
    write_failure_report(analysis, total_solved=len(solved), cot_count=len(cot_records))

    # Summary to stdout
    td = analysis["total_deduce"]
    tg = analysis["total_guess"]
    cd = analysis["correct_deduce"]
    cg = analysis["correct_guess"]

    print("\n=== Summary ===", flush=True)
    print(f"  cryptarithm_deduce : {cd}/{td} solved", flush=True)
    print(f"  cryptarithm_guess  : {cg}/{tg} solved", flush=True)
    print(f"  verified CoT       : {len(cot_records)}", flush=True)

    rule_counts = analysis["rule_counts"]
    rule_correct = analysis["rule_correct"]
    top_unresolved = [
        (r, cnt - rule_correct.get(r, 0))
        for r, cnt in sorted(rule_counts.items(), key=lambda x: -(x[1] - rule_correct.get(x[0], 0)))
        if (cnt - rule_correct.get(r, 0)) > 0
    ]
    if top_unresolved:
        print("\n  Top unresolved rules:", flush=True)
        for rule, miss in top_unresolved[:5]:
            print(f"    {rule:35s} unresolved={miss}", flush=True)


if __name__ == "__main__":
    main()
