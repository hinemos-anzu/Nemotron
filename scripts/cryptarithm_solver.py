#!/usr/bin/env python3
"""Rule-based cryptarithm solver and coverage reporter.

Supported rule candidates:
- forward_concat
- reverse_concat
- reverse_left
- reverse_right
- reverse_both
- interleave_lr
- interleave_rl
- operator_conditioned_rule
- unknown_operator_fallback (reported as unsupported; never verified without examples)

The solver learns candidate string-transform rules from examples and applies the
same verified rule to the question. It is conservative: a record is marked solved
only when at least one example supports a rule and the solver answer exactly
matches the normalized provided answer.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_INVENTORY = Path("reports/cryptarithm/cryptarithm_problem_inventory.csv")
DEFAULT_COVERAGE = Path("reports/cryptarithm/cryptarithm_solver_coverage.csv")
TOKEN = r"[A-Za-z0-9]+"
EXPR_RE = re.compile(rf"(?P<left>{TOKEN})\s*(?P<op>[^A-Za-z0-9\s]{{1,4}})\s*(?P<right>{TOKEN})")


@dataclass(frozen=True)
class Example:
    left: str
    op: str
    right: str
    output: str


@dataclass(frozen=True)
class ParsedQuestion:
    left: str
    op: str
    right: str


@dataclass(frozen=True)
class SolverResult:
    rule_type: str
    solver_answer: str
    verified: bool
    confidence: str
    failure_type: str
    matched_examples: int
    total_examples: int
    notes: str


def interleave(left: str, right: str, start_left: bool = True) -> str:
    first, second = (left, right) if start_left else (right, left)
    chars: List[str] = []
    for idx in range(max(len(first), len(second))):
        if idx < len(first):
            chars.append(first[idx])
        if idx < len(second):
            chars.append(second[idx])
    return "".join(chars)


def rule_functions() -> Dict[str, Callable[[str, str], str]]:
    return {
        "forward_concat": lambda l, r: l + r,
        "reverse_concat": lambda l, r: r + l,
        "reverse_left": lambda l, r: l[::-1] + r,
        "reverse_right": lambda l, r: l + r[::-1],
        "reverse_both": lambda l, r: l[::-1] + r[::-1],
        "interleave_lr": lambda l, r: interleave(l, r, True),
        "interleave_rl": lambda l, r: interleave(l, r, False),
        "unknown_operator_fallback": lambda l, r: l + r,
    }


def clean_answer(text: str) -> str:
    text = str(text or "").strip()
    boxed = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed.group(1).strip()
    else:
        therefore = re.search(r"(?:answer|therefore|so)\s*(?:is|=|:)\s*([A-Za-z0-9]+)\b", text, re.IGNORECASE)
        if therefore:
            text = therefore.group(1)
        else:
            quoted = re.search(r"['\"]([A-Za-z0-9]+)['\"]\s*$", text)
            if quoted:
                text = quoted.group(1)
    return re.sub(r"[^A-Za-z0-9]", "", text.strip().strip("."))


def parse_question(text: str) -> Optional[ParsedQuestion]:
    match = EXPR_RE.search(str(text or ""))
    if not match:
        return None
    return ParsedQuestion(match.group("left"), match.group("op"), match.group("right"))


def parse_examples(raw: str) -> List[Example]:
    examples: List[Example] = []
    if not raw:
        return examples
    candidates: List[object]
    try:
        candidates = [json.loads(raw)]
    except Exception:
        try:
            candidates = [ast.literal_eval(raw)]
        except Exception:
            candidates = [raw]

    def visit(obj: object) -> None:
        if isinstance(obj, dict):
            inp = obj.get("input") or obj.get("question") or obj.get("prompt") or obj.get("x")
            out = obj.get("output") or obj.get("answer") or obj.get("target") or obj.get("y")
            if inp is not None and out is not None:
                parsed = parse_question(str(inp))
                if parsed:
                    examples.append(Example(parsed.left, parsed.op, parsed.right, clean_answer(str(out))))
            for value in obj.values():
                if isinstance(value, (dict, list, tuple)):
                    visit(value)
        elif isinstance(obj, (list, tuple)):
            if len(obj) == 2 and not isinstance(obj[0], (dict, list, tuple)):
                parsed = parse_question(str(obj[0]))
                if parsed:
                    examples.append(Example(parsed.left, parsed.op, parsed.right, clean_answer(str(obj[1]))))
            else:
                for value in obj:
                    visit(value)
        elif isinstance(obj, str):
            # Match forms such as "AB?CD -> ABCD" or "AB?CD = CDAB".
            for match in re.finditer(rf"({TOKEN}\s*[^A-Za-z0-9\s]{{1,4}}\s*{TOKEN})\s*(?:->|=>|=)\s*({TOKEN})", obj):
                parsed = parse_question(match.group(1))
                if parsed:
                    examples.append(Example(parsed.left, parsed.op, parsed.right, clean_answer(match.group(2))))

    for candidate in candidates:
        visit(candidate)
    # Deduplicate while preserving order.
    seen = set()
    unique: List[Example] = []
    for ex in examples:
        key = (ex.left, ex.op, ex.right, ex.output)
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def infer_rule(examples: Sequence[Example], question_op: str) -> Tuple[Optional[str], str, int]:
    funcs = rule_functions()
    if not examples:
        return None, "no examples; refusing fallback verification", 0

    valid: List[str] = []
    for name, fn in funcs.items():
        if name == "unknown_operator_fallback":
            continue
        matches = sum(fn(ex.left, ex.right) == ex.output for ex in examples)
        if matches == len(examples):
            valid.append(name)
    if valid:
        return valid[0], f"rule matches all {len(examples)} examples", len(examples)

    # Operator-conditioned inference: learn a rule only from examples sharing the question operator.
    same_op = [ex for ex in examples if ex.op == question_op]
    if same_op:
        for name, fn in funcs.items():
            if name == "unknown_operator_fallback":
                continue
            if all(fn(ex.left, ex.right) == ex.output for ex in same_op):
                return (
                    f"operator_conditioned_rule:{question_op}:{name}",
                    f"operator-conditioned rule matches {len(same_op)} examples with op {question_op}",
                    len(same_op),
                )
    best_name = ""
    best_matches = -1
    for name, fn in funcs.items():
        if name == "unknown_operator_fallback":
            continue
        matches = sum(fn(ex.left, ex.right) == ex.output for ex in examples)
        if matches > best_matches:
            best_name = name
            best_matches = matches
    return None, f"no rule matched all examples; best={best_name} matched {best_matches}/{len(examples)}", max(best_matches, 0)


def apply_rule(rule_type: str, left: str, right: str) -> str:
    base = rule_type.split(":")[-1]
    return rule_functions().get(base, rule_functions()["unknown_operator_fallback"])(left, right)


def solve_record(row: Dict[str, str]) -> SolverResult:
    question = parse_question(row.get("question", ""))
    answer = clean_answer(row.get("answer", ""))
    examples = parse_examples(row.get("examples", ""))
    if question is None:
        return SolverResult("", "", False, "none", "extraction failure", 0, len(examples), "could not parse question expression")
    rule_type, notes, matched = infer_rule(examples, question.op)
    if not rule_type:
        failure = "guess failure" if not examples else "wrong operator rule"
        return SolverResult("", "", False, "low", failure, matched, len(examples), notes)
    solver_answer = apply_rule(rule_type, question.left, question.right)
    verified = bool(answer) and solver_answer == answer
    failure_type = "" if verified else ("answer format failure" if not answer else "answer mismatch")
    confidence = "high" if verified and matched == len(examples) else ("medium" if solver_answer else "low")
    return SolverResult(rule_type, solver_answer, verified, confidence, failure_type, matched, len(examples), notes)


def read_inventory(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_coverage(rows: Sequence[Dict[str, str]], output: Path) -> List[Dict[str, str]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict[str, str]] = []
    for row in rows:
        result = solve_record(row)
        out = dict(row)
        out.update(
            {
                "rule_type": result.rule_type,
                "normalized_answer": clean_answer(row.get("answer", "")),
                "solver_answer": result.solver_answer,
                "verified": str(result.verified).lower(),
                "confidence": result.confidence,
                "failure_type": result.failure_type,
                "matched_examples": str(result.matched_examples),
                "total_examples": str(result.total_examples),
                "solver_notes": result.notes,
            }
        )
        out_rows.append(out)
    fields = list(rows[0].keys()) if rows else ["problem_id", "category", "question", "answer", "examples"]
    for field in ["rule_type", "normalized_answer", "solver_answer", "verified", "confidence", "failure_type", "matched_examples", "total_examples", "solver_notes"]:
        if field not in fields:
            fields.append(field)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--output", default=str(DEFAULT_COVERAGE))
    args = parser.parse_args()
    rows = read_inventory(Path(args.inventory))
    out_rows = write_coverage(rows, Path(args.output))
    solved = sum(row.get("verified") == "true" for row in out_rows)
    print(f"wrote coverage for {len(out_rows)} rows to {args.output}; verified={solved}")


if __name__ == "__main__":
    main()
