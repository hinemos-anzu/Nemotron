"""
cryptarithm_solver.py

Deterministic rule-based solver for cryptarithm_deduce and cryptarithm_guess.
Each rule maps (left: str, op: str, right: str) -> predicted_result: str.

Rules:
  forward_concat        : left + right   (string concat, left first)
  reverse_concat        : right + left   (string concat, right first)
  swap_left_right       : right + left   (alias; meaningful when op context matters)
  reverse_left          : rev(left) + right
  reverse_right         : left + rev(right)
  reverse_both          : rev(left) + rev(right)
  interleave_lr         : interleave chars left-first
  interleave_rl         : interleave chars right-first
  operator_conditioned_rule : select sub-rule based on operator token
  unknown_operator_fallback : exhaustive scan; returns None if nothing matches
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Example:
    left: str
    op: str
    right: str
    result: str


@dataclass
class SolveResult:
    rule_name: str
    predicted: Optional[str]
    is_correct: bool
    explanation: str
    parse_ok: bool = True


# ---------------------------------------------------------------------------
# Low-level rule functions
# ---------------------------------------------------------------------------

def _s(x: str) -> str:
    return str(x).strip()


def forward_concat(left: str, op: str, right: str) -> str:
    return _s(left) + _s(right)


def reverse_concat(left: str, op: str, right: str) -> str:
    return _s(right) + _s(left)


def swap_left_right(left: str, op: str, right: str) -> str:
    return _s(right) + _s(left)


def reverse_left(left: str, op: str, right: str) -> str:
    return _s(left)[::-1] + _s(right)


def reverse_right(left: str, op: str, right: str) -> str:
    return _s(left) + _s(right)[::-1]


def reverse_both(left: str, op: str, right: str) -> str:
    return _s(left)[::-1] + _s(right)[::-1]


def interleave_lr(left: str, op: str, right: str) -> str:
    l, r = _s(left), _s(right)
    out = []
    for i in range(max(len(l), len(r))):
        if i < len(l):
            out.append(l[i])
        if i < len(r):
            out.append(r[i])
    return "".join(out)


def interleave_rl(left: str, op: str, right: str) -> str:
    l, r = _s(left), _s(right)
    out = []
    for i in range(max(len(l), len(r))):
        if i < len(r):
            out.append(r[i])
        if i < len(l):
            out.append(l[i])
    return "".join(out)


# operator → sub-rule mapping used by operator_conditioned_rule
_OP_RULE_MAP: Dict[str, str] = {
    "+": "forward_concat",
    "★": "forward_concat",
    "⊕": "forward_concat",
    "-": "reverse_concat",
    "⊖": "reverse_concat",
    "*": "interleave_lr",
    "×": "interleave_lr",
    "⊗": "interleave_lr",
    "/": "reverse_both",
    "÷": "reverse_both",
}


def operator_conditioned_rule(left: str, op: str, right: str) -> str:
    sub = _OP_RULE_MAP.get(op.strip(), "forward_concat")
    return ALL_RULES[sub](left, op, right)


# Exhaustive fallback — returns None; caller handles
def _unknown_operator_fallback(left: str, op: str, right: str) -> Optional[str]:
    return None


# ---------------------------------------------------------------------------
# Guess-type solver (answer = rule name)
# ---------------------------------------------------------------------------

def solve_guess(question: str, expected_answer: Optional[str] = None) -> SolveResult:
    """
    For cryptarithm_guess: parse examples, identify rule, predicted = rule_name.
    is_correct when identified rule_name == normalized expected_answer.
    """
    examples = parse_examples(question)
    if not examples:
        return SolveResult(
            rule_name="unknown_operator_fallback",
            predicted=None,
            is_correct=False,
            explanation="parse_error: no examples found in guess question.",
            parse_ok=False,
        )

    rule_name = identify_rule(examples)
    predicted = rule_name

    expected_str = _s(expected_answer).lower() if expected_answer else None
    is_correct = predicted is not None and predicted == expected_str

    expl = (
        f"examples_parsed={len(examples)}; "
        f"rule_identified={rule_name}; "
        f"expected={expected_str}; "
        f"match={'YES' if is_correct else 'NO'}"
    )
    return SolveResult(
        rule_name=rule_name,
        predicted=predicted,
        is_correct=is_correct,
        explanation=expl,
        parse_ok=True,
    )


# Ordered rule registry (order matters for tie-breaking)
ALL_RULES: Dict[str, callable] = {
    "forward_concat": forward_concat,
    "reverse_concat": reverse_concat,
    "swap_left_right": swap_left_right,
    "reverse_left": reverse_left,
    "reverse_right": reverse_right,
    "reverse_both": reverse_both,
    "interleave_lr": interleave_lr,
    "interleave_rl": interleave_rl,
    "operator_conditioned_rule": operator_conditioned_rule,
    "unknown_operator_fallback": _unknown_operator_fallback,
}

# Rules tried during identification (exclude the meta rules)
_CANDIDATE_RULES = [
    "forward_concat",
    "reverse_concat",
    "swap_left_right",
    "reverse_left",
    "reverse_right",
    "reverse_both",
    "interleave_lr",
    "interleave_rl",
    "operator_conditioned_rule",
]


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

# Operator charset (single or multi-char tokens)
_OP_PAT = r"[+\-*/★⊕⊖⊗×÷@#$^|~&]+"

# Matches "left OP right = result"
_EXAMPLE_RE = re.compile(
    r"([A-Za-z0-9]+)\s*(" + _OP_PAT + r")\s*([A-Za-z0-9]+)\s*=\s*([A-Za-z0-9]+)"
)

# Matches "left OP right = ?" (test case)
_TEST_RE = re.compile(
    r"([A-Za-z0-9]+)\s*(" + _OP_PAT + r")\s*([A-Za-z0-9]+)\s*=\s*\?"
)


def parse_examples(text: str) -> List[Example]:
    return [
        Example(left=m.group(1), op=m.group(2), right=m.group(3), result=m.group(4))
        for m in _EXAMPLE_RE.finditer(text)
    ]


def parse_test_case(text: str) -> Optional[Tuple[str, str, str]]:
    m = _TEST_RE.search(text)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def identify_rule(examples: List[Example]) -> str:
    """Return the first rule name that reproduces every example, else 'unknown_operator_fallback'."""
    if not examples:
        return "unknown_operator_fallback"
    for rule_name in _CANDIDATE_RULES:
        fn = ALL_RULES[rule_name]
        if all(_safe_apply(fn, ex.left, ex.op, ex.right) == ex.result for ex in examples):
            return rule_name
    return "unknown_operator_fallback"


def _safe_apply(fn, left, op, right) -> Optional[str]:
    try:
        return fn(left, op, right)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public solver entry point
# ---------------------------------------------------------------------------

def solve(question: str, expected_answer: Optional[str] = None) -> SolveResult:
    """
    Parse examples from question text, identify rule, predict test-case answer.

    Args:
        question: Full problem text including examples and '= ?' test line.
        expected_answer: Ground truth for correctness check (optional).

    Returns:
        SolveResult with rule_name, predicted, is_correct, explanation.
    """
    examples = parse_examples(question)
    test_case = parse_test_case(question)

    if not examples:
        return SolveResult(
            rule_name="unknown_operator_fallback",
            predicted=None,
            is_correct=False,
            explanation="parse_error: no examples extracted from question text.",
            parse_ok=False,
        )

    rule_name = identify_rule(examples)

    if test_case is None:
        return SolveResult(
            rule_name=rule_name,
            predicted=None,
            is_correct=False,
            explanation=f"parse_error: no test case (= ?) found. Rule identified from examples: {rule_name}.",
            parse_ok=False,
        )

    left, op, right = test_case

    if rule_name == "unknown_operator_fallback":
        return SolveResult(
            rule_name="unknown_operator_fallback",
            predicted=None,
            is_correct=False,
            explanation=(
                f"No rule matched all {len(examples)} example(s). "
                f"Test case: {left} {op} {right} = ?"
            ),
            parse_ok=True,
        )

    fn = ALL_RULES[rule_name]
    predicted = _safe_apply(fn, left, op, right)
    expected_str = _s(expected_answer) if expected_answer is not None else None
    is_correct = predicted is not None and predicted == expected_str

    expl_parts = [
        f"examples_parsed={len(examples)}",
        f"rule={rule_name}",
        f"input=({left} {op} {right})",
        f"predicted={predicted}",
    ]
    if expected_str is not None:
        expl_parts.append(f"expected={expected_str}")
        expl_parts.append(f"match={'YES' if is_correct else 'NO'}")

    return SolveResult(
        rule_name=rule_name,
        predicted=predicted,
        is_correct=is_correct,
        explanation="; ".join(expl_parts),
        parse_ok=True,
    )


# ---------------------------------------------------------------------------
# Bulk solve helper
# ---------------------------------------------------------------------------

def solve_batch(
    records: List[Dict],
    question_key: str = "question",
    answer_key: str = "answer",
    category_key: str = "category",
) -> List[Dict]:
    """
    Apply solve() or solve_guess() to a list of record dicts.
    cryptarithm_guess records use solve_guess(); all others use solve().
    Returns each record augmented with solver_* fields.
    """
    out = []
    for rec in records:
        q = rec.get(question_key, "")
        a = rec.get(answer_key)
        cat = rec.get(category_key, "")
        if cat == "cryptarithm_guess":
            result = solve_guess(q, expected_answer=a)
        else:
            result = solve(q, expected_answer=a)
        out.append({
            **rec,
            "solver_rule": result.rule_name,
            "solver_predicted": result.predicted,
            "solver_correct": result.is_correct,
            "solver_parse_ok": result.parse_ok,
            "solver_explanation": result.explanation,
        })
    return out


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _SMOKE = [
        {
            "id": "smoke_forward",
            "category": "cryptarithm_deduce",
            "question": "3 + 5 = 35\n7 + 2 = 72\n4 + 8 = ?",
            "answer": "48",
        },
        {
            "id": "smoke_reverse",
            "category": "cryptarithm_deduce",
            "question": "3 + 5 = 53\n7 + 2 = 27\n4 + 8 = ?",
            "answer": "84",
        },
        {
            "id": "smoke_rev_left",
            "category": "cryptarithm_deduce",
            "question": "123 + 45 = 32145\n67 + 89 = 7689\n12 + 34 = ?",
            "answer": "2134",
        },
        {
            "id": "smoke_interleave",
            "category": "cryptarithm_deduce",
            "question": "AB + CD = ACBD\nEF + GH = EGFH\nIJ + KL = ?",
            "answer": "IKJL",
        },
        {
            "id": "smoke_unknown",
            "category": "cryptarithm_deduce",
            "question": "1 + 2 = 99\n3 + 4 = 88\n5 + 6 = ?",
            "answer": "77",
        },
    ]
    results = solve_batch(_SMOKE)
    for r in results:
        status = "OK" if r["solver_correct"] else ("MISS" if r["solver_predicted"] else "FAIL")
        print(f"[{status}] {r['id']:20s} rule={r['solver_rule']:30s} pred={r['solver_predicted']}")
