#!/usr/bin/env python3
"""Phase 3 Token-Budget / Adapter-Path Comparison (Tasks 4-7).

SAFETY CONTRACT (carried over verbatim from the task spec; see
phase3_config_audit/phase3_vs_golden_config_diff.md for the full Stage-1
diff and the three mismatch verdicts this script consumes):
  - Does NOT train or run SFT.
  - Does NOT write/overwrite adapter_model.safetensors or adapter_config.json.
  - Does NOT change rank maps, target_modules, dtype, or LoRA structure.
  - Does NOT create submission.zip or submit to Kaggle.
  - Does NOT modify phase3_run_golden_validation.py's prompt template, parser,
    chat-template wrapping, or quantization logic. The prompt_harness_mismatch
    and quantization_mismatch verdicts from Stage 1 are carried into Section 6
    of the report as caveats and are intentionally left unchanged here.
  - The ONLY things that vary across conditions A / B / (C) are:
      * GOLDEN_GENERATION_CONFIG["max_new_tokens"]  (via --max-new-tokens)
      * the adapter load path                       (via --adapter; per
        Task 4 this is fixed to the B3 adapter for ALL of A/B/C, so it is
        NOT itself a variable across conditions A/B/C - only vs. the
        ORIGINAL baseline run, which used the old adapter)
    output_scores / return_dict_in_generate remain False/unset (logprob OFF)
    in every condition - identical to phase3_run_golden_validation.py's
    defaults, so this is not a variable either.

What this does:
  Task 4: (no code change) phase3_run_golden_validation.py already accepts
          --adapter. This script passes --adapter <B3 adapter dir> (default
          /kaggle/working/golden_b3_adapter, produced by
          phase3_adapter_artifact_audit.py) for every condition below.
  Task 5: build a representative subset of ~6-8 problems from a REAL
          golden_validation_predictions.jsonl (the original ~20% / n=20-30
          baseline run, OLD adapter, max_new_tokens=2048) and write
          phase3_token_budget_subset/subset_cases.csv:
            - 3x bit_manipulation,    finish_reason in {length,error,unknown}
            - 2x other categories,    finish_reason in {length,error,unknown}
            - 1-2x numeral_conversion, is_correct == True  ("correct")
            - 1x any category,        parse_success == False ("parse_fail")
          ("timeout" in the task spec maps to finish_reason in
          {length, error, unknown, not_run, timeout} - i.e. "did not cleanly
          stop via EOS" - since the runner's finish_reason vocabulary has no
          literal "timeout" value.) If a bucket is short, this function backs
          off to progressively broader buckets and records every fallback in
          the selection log / report so the substitution is auditable.
  Task 6: run the subset under
            condition A: --max-new-tokens 2048  (Golden default, B3 adapter)
            condition B: --max-new-tokens 4096  (B3 adapter)
          via subprocess calls to phase3_run_golden_validation.py with
          --adapter <B3 adapter> --problem-ids-file subset_cases.csv.
  Task 7: for up to --max-c-cases (default 2) subset problems still
          finish_reason == "length" after condition B, optionally re-run just
          those with condition C: --max-new-tokens 6144
          (--run-condition-c / --no-run-condition-c, default on).

Outputs (under --output-dir, default phase3_token_budget_audit/):
  phase3_token_budget_subset/subset_cases.csv
  phase3_token_budget_audit/condition_A_2048/golden_validation_predictions.jsonl
  phase3_token_budget_audit/condition_B_4096/golden_validation_predictions.jsonl
  phase3_token_budget_audit/condition_C_6144/golden_validation_predictions.jsonl  (optional)
  phase3_token_budget_comparison.csv
  phase3_token_budget_report.md
  run_commands.md
  reproducibility_notes.md

Usage (Kaggle / RTX GPU session, AFTER phase3_adapter_artifact_audit.py has
extracted + PASSED its check on the B3 adapter):
  python phase3_token_budget_compare.py \\
      --baseline-predictions /kaggle/working/phase3_analysis/golden_validation_predictions.jsonl \\
      --stage1-status phase3_token_budget_audit/phase3_config_audit/STAGE1_STATUS.json \\
      --adapter /kaggle/working/golden_b3_adapter \\
      --problems problems.jsonl \\
      --category-map phase3_analysis/category_map.csv \\
      --output-dir phase3_token_budget_audit

Dry run / no-GPU (build subset + run_commands.md only - e.g. to sanity-check
selection logic before the GPU run; does not invoke the GPU runner):
  python phase3_token_budget_compare.py \\
      --baseline-predictions phase3_analysis/golden_validation_predictions.jsonl \\
      --skip-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ADAPTER = "/kaggle/working/golden_b3_adapter"
DEFAULT_RUNNER = "phase3_run_golden_validation.py"
DEFAULT_PROBLEMS = "problems.jsonl"
DEFAULT_CATEGORY_MAP = "phase3_analysis/category_map.csv"
DEFAULT_OUTPUT_DIR = "phase3_token_budget_audit"

# --subset-dir / --stage1-status default to these paths *relative to
# --output-dir* (computed in main()), matching phase3_adapter_artifact_audit.py's
# layout: <output-dir>/phase3_config_audit/STAGE1_STATUS.json (Stage 1 output)
# and <output-dir>/phase3_token_budget_subset/subset_cases.csv (Stage 2 output).
SUBSET_SUBDIR_NAME = "phase3_token_budget_subset"
STAGE1_STATUS_RELPATH = Path("phase3_config_audit") / "STAGE1_STATUS.json"

# finish_reason values that the task spec calls "timeout/length", i.e.
# "did not cleanly stop via EOS". phase3_run_golden_validation.py's
# finish_reason vocabulary is {"eos", "length", "error", "unknown"}; the
# schema-template predictions file additionally uses "not_run". "timeout" is
# included defensively in case a future runner emits it literally.
NOT_EOS_FINISH_REASONS = {"length", "error", "unknown", "not_run", "timeout"}

# Condition letter -> max_new_tokens override passed to the runner.
CONDITION_TOKEN_BUDGETS = {"A": 2048, "B": 4096, "C": 6144}
CONDITION_DIR_NAMES = {
    "A": "condition_A_2048",
    "B": "condition_B_4096",
    "C": "condition_C_6144",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL predictions file. Returns [] if the file does not exist."""
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_stage1_status(path: Path) -> Optional[Dict[str, Any]]:
    """Load phase3_config_audit/STAGE1_STATUS.json, or None if missing."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Task 5: representative subset selection
# ---------------------------------------------------------------------------

def select_subset(
    records: List[Dict[str, Any]],
    n_bit_manip_noneos: int = 3,
    n_other_noneos: int = 2,
    n_numeral_correct: int = 2,
    n_parse_fail: int = 1,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Select a representative subset of problems from real baseline predictions.

    Target composition (6-8 problems total):
      - n_bit_manip_noneos x bit_manipulation, finish_reason in NOT_EOS_FINISH_REASONS
      - n_other_noneos     x non-bit_manipulation, finish_reason in NOT_EOS_FINISH_REASONS
      - up to n_numeral_correct x numeral_conversion, is_correct == True
      - n_parse_fail       x any category, parse_success == False

    If a bucket has fewer candidates than requested, this function backfills
    from progressively broader buckets and records every fallback decision in
    the returned selection log (str list), so substitutions are auditable in
    the report rather than silent.

    Returns (selected_records, selection_log). Each selected record gains a
    "_selection_bucket" key describing which bucket (or fallback) it filled.
    """
    log: List[str] = []
    selected: List[Dict[str, Any]] = []
    selected_ids: set = set()

    def _pid(r: Dict[str, Any]) -> str:
        return str(r.get("problem_id", ""))

    def _take(pool: List[Dict[str, Any]], n: int, bucket_name: str) -> int:
        """Take up to n records from pool (skipping already-selected ids).
        Returns the number actually taken."""
        taken = 0
        for r in pool:
            if taken >= n:
                break
            pid = _pid(r)
            if pid in selected_ids:
                continue
            r = dict(r)
            r["_selection_bucket"] = bucket_name
            selected.append(r)
            selected_ids.add(pid)
            taken += 1
        return taken

    # --- Bucket 1: bit_manipulation, non-EOS finish_reason -----------------
    bit_noneos = [
        r for r in records
        if r.get("category") == "bit_manipulation"
        and r.get("finish_reason") in NOT_EOS_FINISH_REASONS
    ]
    n_taken = _take(bit_noneos, n_bit_manip_noneos, "bit_manipulation_noneos")
    if n_taken < n_bit_manip_noneos:
        log.append(
            f"bit_manipulation_noneos: found {len(bit_noneos)}, needed "
            f"{n_bit_manip_noneos}, took {n_taken}. Backfilling remaining "
            f"{n_bit_manip_noneos - n_taken} slot(s) from any-category "
            f"non-EOS cases."
        )
        fallback_pool = [
            r for r in records
            if r.get("finish_reason") in NOT_EOS_FINISH_REASONS
            and _pid(r) not in selected_ids
        ]
        extra = _take(fallback_pool, n_bit_manip_noneos - n_taken, "bit_manipulation_noneos_FALLBACK_any_category")
        if extra < (n_bit_manip_noneos - n_taken):
            log.append(
                f"  -> fallback also short: only {extra} additional "
                f"non-EOS case(s) available across all categories."
            )

    # --- Bucket 2: non-bit_manipulation, non-EOS finish_reason --------------
    other_noneos = [
        r for r in records
        if r.get("category") != "bit_manipulation"
        and r.get("finish_reason") in NOT_EOS_FINISH_REASONS
        and _pid(r) not in selected_ids
    ]
    n_taken = _take(other_noneos, n_other_noneos, "other_noneos")
    if n_taken < n_other_noneos:
        log.append(
            f"other_noneos: found {len(other_noneos)}, needed "
            f"{n_other_noneos}, took {n_taken}. Backfilling remaining "
            f"{n_other_noneos - n_taken} slot(s) from any incorrect case "
            f"(is_correct == False), any finish_reason."
        )
        fallback_pool = [
            r for r in records
            if r.get("is_correct") is False
            and _pid(r) not in selected_ids
        ]
        extra = _take(fallback_pool, n_other_noneos - n_taken, "other_noneos_FALLBACK_incorrect")
        if extra < (n_other_noneos - n_taken):
            log.append(
                f"  -> fallback also short: only {extra} additional "
                f"incorrect case(s) available."
            )

    # --- Bucket 3: numeral_conversion, correct ------------------------------
    numeral_correct = [
        r for r in records
        if r.get("category") == "numeral_conversion"
        and r.get("is_correct") is True
        and _pid(r) not in selected_ids
    ]
    n_taken = _take(numeral_correct, n_numeral_correct, "numeral_conversion_correct")
    if n_taken < n_numeral_correct:
        log.append(
            f"numeral_conversion_correct: found {len(numeral_correct)}, "
            f"wanted up to {n_numeral_correct}, took {n_taken}."
        )
        if n_taken == 0:
            fallback_pool = [
                r for r in records
                if r.get("is_correct") is True
                and _pid(r) not in selected_ids
            ]
            extra = _take(fallback_pool, 1, "numeral_conversion_correct_FALLBACK_any_correct")
            if extra:
                log.append(
                    "  -> fallback: took 1 correct case from a different "
                    "category instead (no numeral_conversion correct case "
                    "available)."
                )
            else:
                log.append(
                    "  -> no fallback available: zero is_correct == True "
                    "cases found in baseline predictions at all. Subset "
                    "will be smaller than the 6-8 target."
                )

    # --- Bucket 4: any category, parse_fail ---------------------------------
    parse_fail = [
        r for r in records
        if r.get("parse_success") is False
        and _pid(r) not in selected_ids
    ]
    n_taken = _take(parse_fail, n_parse_fail, "parse_fail")
    if n_taken < n_parse_fail:
        log.append(
            f"parse_fail: found {len(parse_fail)}, needed {n_parse_fail}, "
            f"took {n_taken}. No parse_success == False cases remain "
            f"(outside already-selected problems) - subset will be smaller "
            f"than the 6-8 target by {n_parse_fail - n_taken}."
        )

    log.append(f"TOTAL selected: {len(selected)} (target range: 6-8)")
    return selected, log


SUBSET_CSV_FIELDS = [
    "problem_id",
    "category",
    "subcategory",
    "selection_bucket",
    "gold_answer",
    "baseline_pred_answer",
    "baseline_is_correct",
    "baseline_parse_success",
    "baseline_parse_error_type",
    "baseline_finish_reason",
    "baseline_generation_token_count",
    "baseline_min_logprob",
    "baseline_answer_min_logprob",
]


def write_subset_csv(selected: List[Dict[str, Any]], path: Path) -> None:
    """Write subset_cases.csv. Only the 'problem_id' column is required by
    phase3_run_golden_validation.py's --problem-ids-file; the remaining
    columns capture baseline context for the report's case-level findings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUBSET_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in selected:
            writer.writerow({
                "problem_id": r.get("problem_id", ""),
                "category": r.get("category", ""),
                "subcategory": r.get("subcategory", ""),
                "selection_bucket": r.get("_selection_bucket", ""),
                "gold_answer": r.get("gold_answer", ""),
                "baseline_pred_answer": r.get("pred_answer", ""),
                "baseline_is_correct": r.get("is_correct", ""),
                "baseline_parse_success": r.get("parse_success", ""),
                "baseline_parse_error_type": r.get("parse_error_type", ""),
                "baseline_finish_reason": r.get("finish_reason", ""),
                "baseline_generation_token_count": r.get("generation_token_count", ""),
                "baseline_min_logprob": r.get("min_logprob", ""),
                "baseline_answer_min_logprob": r.get("answer_min_logprob", ""),
            })


# ---------------------------------------------------------------------------
# Task 6/7: orchestration (subprocess calls to phase3_run_golden_validation.py)
# ---------------------------------------------------------------------------

def build_run_command(
    python_exe: str,
    runner: str,
    adapter: str,
    problems: str,
    category_map: str,
    subset_csv: Path,
    max_new_tokens: int,
    output_dir: Path,
    model: Optional[str] = None,
    seed: int = 42,
) -> List[str]:
    """Build the exact subprocess argv for one condition.

    Per Task 6/7: --adapter is the B3 adapter for ALL conditions (Task 4),
    --problem-ids-file restricts to the fixed subset (Task 5), and
    --max-new-tokens is the ONLY generation-config override (Tasks 6/7).
    output_scores / return_dict_in_generate are never passed, so they remain
    False/unset (logprob OFF) - identical across baseline and A/B/C.
    """
    cmd = [
        python_exe, runner,
        "--adapter", adapter,
        "--problems", problems,
        "--category-map", category_map,
        "--problem-ids-file", str(subset_csv),
        "--max-new-tokens", str(max_new_tokens),
        "--output-dir", str(output_dir),
        "--seed", str(seed),
    ]
    if model:
        cmd += ["--model", model]
    return cmd


def run_condition(cmd: List[str], output_dir: Path, dry_run: bool) -> Dict[str, Any]:
    """Execute (or, if dry_run, skip) one condition's runner subprocess.

    The subprocess's stdout/stderr stream live to this script's stdout/stderr
    (no capture) so progress is visible during long Kaggle runs.

    In dry_run mode, no subprocess is launched; if golden_validation_predictions.jsonl
    already exists in output_dir from a prior real run, it is loaded so that
    aggregation/diagnosis can still proceed (e.g. on a second --skip-run pass
    after the GPU run already happened).
    """
    pred_path = output_dir / "golden_validation_predictions.jsonl"
    result: Dict[str, Any] = {
        "cmd": " ".join(cmd),
        "output_dir": str(output_dir),
        "predictions_path": str(pred_path),
        "executed": False,
        "returncode": None,
        "predictions": [],
    }
    if dry_run:
        result["predictions"] = load_jsonl(pred_path)
        if result["predictions"]:
            result["returncode"] = 0  # results pre-exist from an earlier real run
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd)
    result["executed"] = True
    result["returncode"] = proc.returncode
    result["predictions"] = load_jsonl(pred_path)
    return result


def select_condition_c_subset(
    condition_b_records: List[Dict[str, Any]],
    subset_problem_ids: List[str],
    max_cases: int,
) -> List[Dict[str, Any]]:
    """Task 7: from condition B's predictions, pick up to max_cases subset
    problems that are STILL finish_reason == "length" after 4096 tokens."""
    by_id = {r.get("problem_id"): r for r in condition_b_records}
    candidates = []
    for pid in subset_problem_ids:
        r = by_id.get(pid)
        if r is not None and r.get("finish_reason") == "length":
            candidates.append(r)
    return candidates[:max_cases]


# ---------------------------------------------------------------------------
# Aggregation: phase3_token_budget_comparison.csv
# ---------------------------------------------------------------------------

COMPARISON_CSV_FIELDS = [
    "problem_id", "category", "subcategory", "selection_bucket", "gold_answer",
    "condition", "max_new_tokens", "adapter",
    "is_correct", "parse_success", "parse_error_type", "finish_reason",
    "generation_token_count", "elapsed_seconds", "pred_answer",
    "min_logprob", "answer_min_logprob", "status",
]


def _extract_run_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "is_correct": rec.get("is_correct", ""),
        "parse_success": rec.get("parse_success", ""),
        "parse_error_type": rec.get("parse_error_type", ""),
        "finish_reason": rec.get("finish_reason", ""),
        "generation_token_count": rec.get("generation_token_count", ""),
        "elapsed_seconds": rec.get("elapsed_seconds", ""),
        "pred_answer": rec.get("pred_answer", ""),
        "min_logprob": rec.get("min_logprob", ""),
        "answer_min_logprob": rec.get("answer_min_logprob", ""),
    }


_EMPTY_RUN_FIELDS = {
    "is_correct": "", "parse_success": "", "parse_error_type": "",
    "finish_reason": "", "generation_token_count": "", "elapsed_seconds": "",
    "pred_answer": "", "min_logprob": "", "answer_min_logprob": "",
}


def build_comparison_rows(
    subset: List[Dict[str, Any]],
    baseline_records: List[Dict[str, Any]],
    condition_results: Dict[str, Dict[str, Any]],
    old_adapter_label: str,
    b3_adapter_label: str,
) -> List[Dict[str, Any]]:
    """One row per (subset problem, condition in {baseline, A, B, C}).

    'status' explains rows with empty run fields:
      - "ok": real result present
      - "condition_not_run": this script did not invoke this condition at all
        (e.g. --skip-run with no prior results, or condition C was disabled)
      - "not_in_condition_subset": condition C ran, but this problem was not
        among the <=max-c-cases escalated cases (Task 7 is selective)
    """
    baseline_by_id = {r.get("problem_id"): r for r in baseline_records}
    rows: List[Dict[str, Any]] = []

    for s in subset:
        pid = s.get("problem_id")
        common = {
            "problem_id": pid,
            "category": s.get("category", ""),
            "subcategory": s.get("subcategory", ""),
            "selection_bucket": s.get("_selection_bucket", s.get("selection_bucket", "")),
            "gold_answer": s.get("gold_answer", ""),
        }

        base = baseline_by_id.get(pid)
        if base is not None:
            rows.append({
                **common, **_extract_run_fields(base),
                "condition": "baseline",
                "max_new_tokens": base.get("generation_config", {}).get("max_new_tokens", 2048),
                "adapter": old_adapter_label,
                "status": "ok",
            })
        else:
            rows.append({
                **common, **_EMPTY_RUN_FIELDS,
                "condition": "baseline", "max_new_tokens": 2048,
                "adapter": old_adapter_label, "status": "missing_from_baseline_predictions",
            })

        for letter in ("A", "B", "C"):
            cr = condition_results.get(letter)
            row_base = {
                **common,
                "condition": letter,
                "max_new_tokens": CONDITION_TOKEN_BUDGETS[letter],
                "adapter": b3_adapter_label,
            }
            if cr is None:
                rows.append({**row_base, **_EMPTY_RUN_FIELDS, "status": "condition_not_run"})
                continue
            preds_by_id = {r.get("problem_id"): r for r in cr["predictions"]}
            rec = preds_by_id.get(pid)
            if rec is None:
                status = "condition_not_run" if not cr["predictions"] else "not_in_condition_subset"
                rows.append({**row_base, **_EMPTY_RUN_FIELDS, "status": status})
                continue
            rows.append({**row_base, **_extract_run_fields(rec), "status": "ok"})

    return rows


def write_comparison_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COMPARISON_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# Section 6 Diagnosis: 5 judgment rules -> pattern A/B/C/D
# ---------------------------------------------------------------------------

def _norm(s: Any) -> str:
    return str(s if s is not None else "").strip().lower()


def _is_failing(rec: Optional[Dict[str, Any]]) -> Optional[bool]:
    """True if rec represents a failing case (wrong / unparsed / non-EOS).
    None if rec is unavailable (condition not run for this problem)."""
    if rec is None:
        return None
    if rec.get("is_correct") is True:
        return False
    return True


def _gold_in_raw(rec: Dict[str, Any], gold_answer: str) -> bool:
    gold = _norm(gold_answer)
    if not gold:
        return False
    haystack = _norm(rec.get("raw_output", "")) + " " + _norm(rec.get("final_answer_text", ""))
    return gold in haystack


def diagnose(
    subset: List[Dict[str, Any]],
    baseline_records: List[Dict[str, Any]],
    condition_results: Dict[str, Dict[str, Any]],
    stage1_status: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply the 5 judgment rules and classify into pattern A/B/C/D.

    Rule 1 (artifact-fix signal, isolates Task-4 adapter swap): baseline
      (OLD adapter, 2048) is failing AND condition A (B3 adapter, 2048) is
      is_correct == True for the SAME problem. max_new_tokens is held
      constant (2048) on both sides, so any flip is attributable to the
      adapter path alone.

    Rule 2 (token-budget-fix signal, isolates Tasks 6/7 max_new_tokens):
      condition A (B3, 2048) is failing AND a higher-budget condition for the
      SAME problem (B at 4096, or C at 6144 if escalated) becomes
      is_correct == True, OR finish_reason flips length -> eos. Adapter is
      held constant (B3) on both sides, so any flip is attributable to
      max_new_tokens alone.

    Rule 3 (parser-issue signal): in any of A/B/C, finish_reason == "eos" but
      parse_success == False, OR is_correct == False while the gold answer
      string appears verbatim in raw_output/final_answer_text (the model
      produced the right answer but extract_answer/answers_match did not
      recognize it).

    Rule 4 (reasoning-failure signal): in any of A/B/C, finish_reason ==
      "eos", parse_success == True, is_correct == False, and the gold answer
      does NOT appear anywhere in raw_output (a genuinely wrong computation,
      not an extraction miss).

    Rule 5 (precedence -> overall pattern): patterns are not mutually
      exclusive at the per-case level (different subset cases can show
      different signals). Overall classification uses precedence
      A > B > C > D: if Rule 1 fires for >=1 case -> "A" (artifact mismatch
      primary); elif Rule 2 fires -> "B" (token budget primary); elif
      Rule 3 fires -> "C" (parser issue primary); elif Rule 4 fires, or no
      rule fires at all -> "D" (true reasoning failure / capability ceiling).
      This order reflects increasing remediation cost: A and B are pure
      configuration changes already within the allowed verification-time
      changes; C is a parser/regex fix (still out-of-scope per the
      prohibition list, but cheap/local); D is the only outcome implying a
      genuine model-capability gap.

    Returns "PENDING" for rule5_overall_pattern (with no rule evaluation) if
    condition A has not been run for any subset problem yet, since Rules 1-4
    all depend on at least condition A's results.
    """
    baseline_by_id = {r.get("problem_id"): r for r in baseline_records}
    preds_by_letter: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for letter in ("A", "B", "C"):
        cr = condition_results.get(letter)
        preds_by_letter[letter] = (
            {r.get("problem_id"): r for r in cr["predictions"]} if cr else {}
        )

    any_a_results = bool(preds_by_letter["A"])
    if not any_a_results:
        return {
            "rule1_artifact_fix": {"count": 0, "problem_ids": [], "detail": []},
            "rule2_token_budget_fix": {"count": 0, "problem_ids": [], "detail": []},
            "rule3_parser_issue": {"count": 0, "problem_ids": [], "detail": []},
            "rule4_reasoning_failure": {"count": 0, "problem_ids": [], "detail": []},
            "rule5_overall_pattern": "PENDING",
            "rule5_rationale": (
                "Condition A has not been run yet (no predictions found in "
                "condition_A_2048/golden_validation_predictions.jsonl). "
                "Run Task 6 (conditions A and B) and re-run this script "
                "(optionally with --skip-run if results already exist on "
                "disk) to compute the diagnosis."
            ),
            "stage1_caveats": _extract_stage1_caveats(stage1_status),
        }

    rule1: Dict[str, Any] = {"count": 0, "problem_ids": [], "detail": []}
    rule2: Dict[str, Any] = {"count": 0, "problem_ids": [], "detail": []}
    rule3: Dict[str, Any] = {"count": 0, "problem_ids": [], "detail": []}
    rule4: Dict[str, Any] = {"count": 0, "problem_ids": [], "detail": []}

    for s in subset:
        pid = s.get("problem_id")
        gold = str(s.get("gold_answer", ""))
        base = baseline_by_id.get(pid)
        a = preds_by_letter["A"].get(pid)
        b = preds_by_letter["B"].get(pid)
        c = preds_by_letter["C"].get(pid)

        # --- Rule 1: baseline (old adapter, 2048) failing -> A (B3, 2048) correct
        if base is not None and a is not None:
            if _is_failing(base) and a.get("is_correct") is True:
                rule1["count"] += 1
                rule1["problem_ids"].append(pid)
                rule1["detail"].append(
                    f"{pid}: baseline(old adapter,2048)=FAIL "
                    f"(is_correct={base.get('is_correct')}, "
                    f"finish_reason={base.get('finish_reason')}, "
                    f"parse_success={base.get('parse_success')}) -> "
                    f"A(B3 adapter,2048)=CORRECT"
                )

        # --- Rule 2: A (B3,2048) failing -> B (B3,4096) [or C] fixes it / resolves length
        for higher_letter, higher_rec, prev_rec, prev_label, hi_label in (
            ("B", b, a, "A(B3,2048)", "B(B3,4096)"),
            ("C", c, b, "B(B3,4096)", "C(B3,6144)"),
        ):
            if prev_rec is None or higher_rec is None:
                continue
            prev_failing = _is_failing(prev_rec)
            if not prev_failing:
                continue
            fixed_correct = higher_rec.get("is_correct") is True
            length_resolved = (
                _norm(prev_rec.get("finish_reason")) == "length"
                and _norm(higher_rec.get("finish_reason")) == "eos"
            )
            if fixed_correct or length_resolved:
                rule2["count"] += 1
                rule2["problem_ids"].append(pid)
                rule2["detail"].append(
                    f"{pid}: {prev_label}=FAIL "
                    f"(is_correct={prev_rec.get('is_correct')}, "
                    f"finish_reason={prev_rec.get('finish_reason')}) -> "
                    f"{hi_label}="
                    f"{'CORRECT' if fixed_correct else 'length->eos'} "
                    f"(is_correct={higher_rec.get('is_correct')}, "
                    f"finish_reason={higher_rec.get('finish_reason')})"
                )

        # --- Rule 3: parser-issue signal across A/B/C
        for letter, rec in (("A", a), ("B", b), ("C", c)):
            if rec is None:
                continue
            eos_but_unparsed = (
                _norm(rec.get("finish_reason")) == "eos"
                and rec.get("parse_success") is False
            )
            wrong_but_gold_present = (
                rec.get("is_correct") is False and _gold_in_raw(rec, gold)
            )
            if eos_but_unparsed or wrong_but_gold_present:
                rule3["count"] += 1
                rule3["problem_ids"].append(f"{pid}[{letter}]")
                reason = (
                    "finish_reason=eos but parse_success=False"
                    if eos_but_unparsed
                    else "is_correct=False but gold_answer text appears in raw_output "
                         "(extraction/normalization miss)"
                )
                rule3["detail"].append(f"{pid} condition {letter}: {reason}")

        # --- Rule 4: reasoning-failure signal across A/B/C
        for letter, rec in (("A", a), ("B", b), ("C", c)):
            if rec is None:
                continue
            if (
                _norm(rec.get("finish_reason")) == "eos"
                and rec.get("parse_success") is True
                and rec.get("is_correct") is False
                and not _gold_in_raw(rec, gold)
            ):
                rule4["count"] += 1
                rule4["problem_ids"].append(f"{pid}[{letter}]")
                rule4["detail"].append(
                    f"{pid} condition {letter}: finish_reason=eos, "
                    f"parse_success=True, is_correct=False, gold_answer "
                    f"not found in raw_output -> pred_answer="
                    f"{rec.get('pred_answer')!r} vs gold_answer={gold!r}"
                )

    # --- Rule 5: precedence A > B > C > D
    if rule1["count"] > 0:
        overall = "A"
        rationale = (
            f"Rule 1 fired for {rule1['count']} case(s): switching ONLY the "
            f"adapter path (old -> B3, max_new_tokens held at 2048) flipped "
            f"these from failing to is_correct=True. -> Pattern A "
            f"(artifact mismatch primary)."
        )
    elif rule2["count"] > 0:
        overall = "B"
        rationale = (
            f"Rule 1 fired for 0 cases (adapter swap alone did not fix any "
            f"subset problem at 2048 tokens), but Rule 2 fired for "
            f"{rule2['count']} case(s): with the adapter held constant (B3), "
            f"increasing max_new_tokens resolved a length-truncation or "
            f"flipped the case to is_correct=True. -> Pattern B "
            f"(token budget primary)."
        )
    elif rule3["count"] > 0:
        overall = "C"
        rationale = (
            f"Rules 1-2 fired for 0 cases (neither the adapter swap nor a "
            f"larger token budget fixed any subset problem), but Rule 3 "
            f"fired {rule3['count']} time(s): the model reaches EOS with the "
            f"correct answer present in its output, but "
            f"extract_answer/answers_match fails to recognize it. -> "
            f"Pattern C (parser issue primary)."
        )
    elif rule4["count"] > 0:
        overall = "D"
        rationale = (
            f"Rules 1-3 fired for 0 cases. Rule 4 fired for {rule4['count']} "
            f"case(s): the model reaches EOS, produces a parseable answer, "
            f"but the answer is simply wrong and the gold answer is not "
            f"present anywhere in the output (not an extraction miss). -> "
            f"Pattern D (true reasoning failure primary)."
        )
    else:
        overall = "D"
        rationale = (
            "Rules 1-4 all fired for 0 cases across the subset (no fixes "
            "from the adapter swap, no fixes from larger token budgets, no "
            "parser-recognizable near-misses, and no clean "
            "eos+parsed+wrong+gold-absent cases either - e.g. results may "
            "still be finish_reason=length/error even at the largest budget "
            "tried). Defaulting to Pattern D (true reasoning failure / "
            "capability ceiling) - manual review of "
            "phase3_token_budget_comparison.csv and the raw_output of each "
            "remaining failing case is recommended before drawing "
            "conclusions."
        )

    return {
        "rule1_artifact_fix": rule1,
        "rule2_token_budget_fix": rule2,
        "rule3_parser_issue": rule3,
        "rule4_reasoning_failure": rule4,
        "rule5_overall_pattern": overall,
        "rule5_rationale": rationale,
        "stage1_caveats": _extract_stage1_caveats(stage1_status),
    }


def _extract_stage1_caveats(stage1_status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if stage1_status is None:
        return {
            "available": False,
            "note": (
                "STAGE1_STATUS.json not found - run "
                "phase3_adapter_artifact_audit.py first. The artifact_mismatch, "
                "prompt_harness_mismatch, and quantization_mismatch verdicts "
                "from Stage 1 could not be loaded."
            ),
        }
    return {
        "available": True,
        "artifact_mismatch": stage1_status.get("artifact_mismatch"),
        "prompt_harness_mismatch": stage1_status.get("prompt_harness_mismatch"),
        "quantization_mismatch": stage1_status.get("quantization_mismatch"),
        "stage1_pass": stage1_status.get("pass"),
        "b3_adapter_dir": stage1_status.get("b3_adapter_dir"),
        "old_adapter_dir": stage1_status.get("old_adapter_dir"),
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

# Short, fixed summaries of the 3 Stage-1 verdicts. STAGE1_STATUS.json only
# stores the booleans; the full reasoning lives in
# phase3_config_audit/phase3_vs_golden_config_diff.md (written by
# phase3_adapter_artifact_audit.py). These one-line summaries mirror that
# document so Section 3 is self-contained even if the diff file isn't open.
MISMATCH_MEANINGS = {
    "artifact_mismatch": (
        "Phase3's ADAPTER_PATH points at the B3 SVD conversion's "
        "PRE-conversion INPUT adapter, not the POST-conversion OUTPUT "
        "(/kaggle/working/adapter) that was actually zipped into B3's "
        "submission.zip (Public LB 0.86). Task 4 (this script's --adapter) "
        "fixes this for conditions A/B/C by pointing at the extracted, "
        "verified B3 adapter."
    ),
    "prompt_harness_mismatch": (
        "The executed Golden-Baseline notebook (0.86 LB) wraps every problem "
        "in ChatML via apply_chat_template() "
        "(<|im_start|>user...<|im_end|><|im_start|>assistant) before "
        "tokenizing. phase3_run_golden_validation.py tokenizes the raw "
        "prompt text directly, with no chat-template wrapping. NOT changed "
        "by this script (would be a 2nd simultaneous variable change) - "
        "recorded as a caveat."
    ),
    "quantization_mismatch": (
        "phase3_run_golden_validation.py ALWAYS forces 4-bit NF4 "
        "quantization (hard requirement, tuned for Kaggle T4x2). The "
        "executed Golden-Baseline notebook only uses 4-bit if total GPU "
        "VRAM < 50GB; on the RTX Pro 5000 (>=50GB) that produced the 0.86 "
        "LB / 20% n=20-30 result, it ran in FULL bf16 with NO quantization. "
        "NOT changed by this script - recorded as a caveat. Held constant "
        "across baseline/A/B/C within THIS script, so it does not confound "
        "the A vs B vs C max_new_tokens comparison, but it DOES confound any "
        "comparison back to the original 0.86 LB run."
    ),
}


def _md_table(rows: List[Dict[str, Any]], columns: List[Tuple[str, str]]) -> List[str]:
    """columns: list of (row_key, header_text)."""
    headers = [h for _, h in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        cells = []
        for k, _ in columns:
            v = r.get(k, "")
            if v is None or v == "":
                v = "-"
            cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _condition_stats(comparison_rows: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    sub = [r for r in comparison_rows if r["condition"] == condition and r["status"] == "ok"]
    n = len(sub)
    n_correct = sum(1 for r in sub if r["is_correct"] is True)
    n_eos = sum(1 for r in sub if _norm(r["finish_reason"]) == "eos")
    n_length = sum(1 for r in sub if _norm(r["finish_reason"]) == "length")
    n_parsed = sum(1 for r in sub if r["parse_success"] is True)
    accuracy = (n_correct / n) if n else None
    return {
        "n": n, "n_correct": n_correct, "accuracy": accuracy,
        "n_eos": n_eos, "n_length": n_length, "n_parsed": n_parsed,
    }


def write_token_budget_report(
    path: Path,
    subset: List[Dict[str, Any]],
    selection_log: List[str],
    baseline_records: List[Dict[str, Any]],
    condition_results: Dict[str, Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
    diagnosis: Dict[str, Any],
    stage1_status: Optional[Dict[str, Any]],
    comparison_csv_path: Path,
    args: argparse.Namespace,
    old_adapter_label: str,
    b3_adapter_label: str,
) -> None:
    L: List[str] = []
    pattern = diagnosis["rule5_overall_pattern"]
    pattern_titles = {
        "A": "Pattern A - Artifact mismatch primary",
        "B": "Pattern B - Token budget primary",
        "C": "Pattern C - Parser issue primary",
        "D": "Pattern D - True reasoning failure primary",
        "PENDING": "PENDING - conditions A/B not yet run",
    }

    L.append("# Phase 3 Token Budget / Adapter Comparison Report")
    L.append("")
    L.append(
        "This report covers Tasks 4-7 of the Phase 3 audit: re-running a "
        "representative subset of the original ~20% (n=20-30) baseline run "
        "with (Task 4) the B3 Golden adapter and (Tasks 6-7) larger "
        "max_new_tokens budgets, to determine whether the adapter-path "
        "mismatch and/or the 2048-token budget were primary causes of the "
        "low score - as opposed to a parser issue or a true model-reasoning "
        "failure."
    )
    L.append("")

    # -----------------------------------------------------------------
    # 1. Summary
    # -----------------------------------------------------------------
    L.append("## 1. Summary")
    L.append("")
    L.append(f"- Subset size: **{len(subset)}** problem(s) (target: 6-8)")
    L.append(f"- Subset cases file: `{args.subset_csv if args.subset_csv else '(see run_commands.md)'}`")
    if stage1_status:
        L.append(
            f"- Stage 1 artifact check (Task 2, B3 adapter vs expected "
            f"values): **{'PASS' if stage1_status.get('golden_check', {}).get('pass') else 'FAIL'}**"
        )
    else:
        L.append("- Stage 1 status: **NOT FOUND** - run `phase3_adapter_artifact_audit.py` first.")
    L.append(
        f"- Known Stage-1 mismatches (all out-of-scope caveats, see Section 3): "
        f"artifact_mismatch=**{diagnosis['stage1_caveats'].get('artifact_mismatch', 'unknown')}**, "
        f"prompt_harness_mismatch=**{diagnosis['stage1_caveats'].get('prompt_harness_mismatch', 'unknown')}**, "
        f"quantization_mismatch=**{diagnosis['stage1_caveats'].get('quantization_mismatch', 'unknown')}**"
    )
    L.append("")
    L.append("### Accuracy on subset by condition")
    L.append("")
    acc_rows = []
    for cond, label, adapter_label, mnt in (
        ("baseline", "baseline (original run)", old_adapter_label, "2048 (orig.)"),
        ("A", "A", b3_adapter_label, "2048"),
        ("B", "B", b3_adapter_label, "4096"),
        ("C", "C", b3_adapter_label, "6144 (escalated subset only)"),
    ):
        st = _condition_stats(comparison_rows, cond)
        if st["n"] == 0:
            continue
        acc_rows.append({
            "condition": label,
            "adapter": "old" if adapter_label == old_adapter_label else "B3",
            "max_new_tokens": mnt,
            "n": st["n"],
            "n_correct": st["n_correct"],
            "accuracy": f"{st['accuracy']:.0%}" if st["accuracy"] is not None else "-",
            "n_eos": st["n_eos"],
            "n_length": st["n_length"],
            "n_parsed": st["n_parsed"],
        })
    L.extend(_md_table(acc_rows, [
        ("condition", "Condition"), ("adapter", "Adapter"), ("max_new_tokens", "max_new_tokens"),
        ("n", "n"), ("n_correct", "correct"), ("accuracy", "accuracy"),
        ("n_eos", "eos"), ("n_length", "length"), ("n_parsed", "parsed"),
    ]))
    L.append("")
    L.append(f"`old` adapter = `{old_adapter_label}`")
    L.append("")
    L.append(f"`B3` adapter = `{b3_adapter_label}`")
    L.append("")
    L.append(f"**Overall diagnosis: {pattern_titles.get(pattern, pattern)}**")
    L.append("")
    L.append(diagnosis["rule5_rationale"])
    L.append("")

    # -----------------------------------------------------------------
    # 2. Golden Adapter Artifact Audit
    # -----------------------------------------------------------------
    L.append("## 2. Golden Adapter Artifact Audit (Stage 1, Tasks 1-2)")
    L.append("")
    if stage1_status:
        gc = stage1_status.get("golden_check", {}) or {}
        oc = stage1_status.get("old_check") or {}
        L.append(f"- B3 adapter extracted to: `{stage1_status.get('b3_adapter_dir', '?')}`")
        L.append(f"- Task 2 expected-value check (B3 adapter): **{'PASS' if gc.get('pass') else 'FAIL'}**")
        if gc.get("failures"):
            for f in gc["failures"]:
                L.append(f"  - FAIL: {f}")
        L.append(f"- Old adapter audited: `{stage1_status.get('old_adapter_dir', '?')}`")
        if oc:
            L.append(f"- Old adapter check: **{'PASS' if oc.get('pass') else 'FAIL'}**"
                     f"{' (informational only - old adapter is expected to differ)' if not oc.get('pass') else ''}")
            if oc.get("failures"):
                for f in oc["failures"]:
                    L.append(f"  - {f}")
        else:
            L.append("- Old adapter check: skipped (`--skip-old-adapter-audit`)")
        L.append("")
        L.append(
            "See `phase3_config_audit/golden_b3_artifact_report.md` and "
            "`phase3_config_audit/old_adapter_artifact_report.md` for full "
            "per-tensor detail (tensor_count, prefix, target_modules, rank "
            "histogram, dtype, sha256)."
        )
    else:
        L.append(
            f"**STAGE1_STATUS.json not found** at `{args.stage1_status}`. Run "
            f"`phase3_adapter_artifact_audit.py` first - this script depends "
            f"on its output for the B3 adapter path (Task 4) and for the "
            f"3 mismatch verdicts referenced in Section 3 / Section 6."
        )
    L.append("")

    # -----------------------------------------------------------------
    # 3. Phase3 vs Golden Config Diff
    # -----------------------------------------------------------------
    L.append("## 3. Phase3 vs Golden Config Diff (Stage 1, Task 3)")
    L.append("")
    L.append(
        "Three static mismatches were identified by comparing "
        "`phase3_run_golden_validation.py` against the executed "
        "Golden-Baseline notebook that produced the 0.86 Public LB / "
        "~100th-place score. All three are BY-CONSTRUCTION / "
        "code-comparison findings - true regardless of this subset's run "
        "results - and all three are OUT OF SCOPE per the prohibition list "
        "(\"prompt改善やparser改善を同時に入れない\", \"複数変数を同時に変更しない\"). "
        "They are recorded here as caveats only; only Task 4 (adapter path) "
        "is exercised by conditions A/B/C below."
    )
    L.append("")
    cav = diagnosis["stage1_caveats"]
    diff_rows = []
    for key in ("artifact_mismatch", "prompt_harness_mismatch", "quantization_mismatch"):
        diff_rows.append({
            "verdict": key,
            "value": cav.get(key, "unknown"),
            "meaning": MISMATCH_MEANINGS[key],
        })
    L.extend(_md_table(diff_rows, [("verdict", "Verdict"), ("value", "Value"), ("meaning", "Meaning")]))
    L.append("")
    L.append(
        "See `phase3_config_audit/phase3_vs_golden_config_diff.md` for the "
        "full 28-row, 3-column "
        "(`phase3_baseline_script` / `phase3_executed_notebook` / "
        "`b3_golden_measured_or_unknown`) diff and the detailed reasoning "
        "behind each verdict."
    )
    L.append("")

    # -----------------------------------------------------------------
    # 4. Token Budget Comparison table
    # -----------------------------------------------------------------
    L.append("## 4. Token Budget Comparison")
    L.append("")
    L.append(f"Full data: `{comparison_csv_path}`. One row per (subset problem, condition).")
    L.append("")
    L.append(
        "`status` legend: `ok` = real result present; `condition_not_run` = "
        "this script did not invoke this condition (e.g. `--skip-run` "
        "before any GPU run, or condition C disabled); "
        "`not_in_condition_subset` = condition C ran but this problem was "
        "not among the escalated `finish_reason=length` cases (Task 7 is "
        "selective); `missing_from_baseline_predictions` = this problem_id "
        "was in the subset but not found in `--baseline-predictions`."
    )
    L.append("")
    adapter_short = {old_adapter_label: "old", b3_adapter_label: "B3"}
    table_rows = []
    for r in comparison_rows:
        table_rows.append({
            "problem_id": r["problem_id"],
            "category": r["category"],
            "condition": r["condition"],
            "adapter": adapter_short.get(r["adapter"], r["adapter"]),
            "max_new_tokens": r["max_new_tokens"],
            "is_correct": r["is_correct"],
            "parse_success": r["parse_success"],
            "finish_reason": r["finish_reason"],
            "generation_token_count": r["generation_token_count"],
            "pred_answer": r["pred_answer"],
            "gold_answer": r["gold_answer"],
            "status": r["status"],
        })
    L.extend(_md_table(table_rows, [
        ("problem_id", "problem_id"), ("category", "category"), ("condition", "cond"),
        ("adapter", "adapter"), ("max_new_tokens", "max_tok"),
        ("is_correct", "correct"), ("parse_success", "parsed"),
        ("finish_reason", "finish"), ("generation_token_count", "gen_tok"),
        ("pred_answer", "pred"), ("gold_answer", "gold"), ("status", "status"),
    ]))
    L.append("")

    # -----------------------------------------------------------------
    # 5. Case-level Findings
    # -----------------------------------------------------------------
    L.append("## 5. Case-level Findings")
    L.append("")
    rows_by_pid: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in comparison_rows:
        rows_by_pid.setdefault(r["problem_id"], {})[r["condition"]] = r

    for s in subset:
        pid = s.get("problem_id")
        bucket = s.get("_selection_bucket", s.get("selection_bucket", ""))
        L.append(f"### `{pid}` ({s.get('category', '?')}/{s.get('subcategory', '?')})")
        L.append("")
        L.append(f"- Selection bucket: `{bucket}`")
        L.append(f"- gold_answer: `{s.get('gold_answer', '')}`")
        cond_rows = rows_by_pid.get(pid, {})
        for cond, label in (
            ("baseline", "Baseline (old adapter, 2048)"),
            ("A", "A (B3 adapter, 2048)"),
            ("B", "B (B3 adapter, 4096)"),
            ("C", "C (B3 adapter, 6144)"),
        ):
            r = cond_rows.get(cond)
            if r is None:
                continue
            if r["status"] != "ok":
                L.append(f"- {label}: {r['status']}")
                continue
            L.append(
                f"- {label}: is_correct=`{r['is_correct']}`, "
                f"parse_success=`{r['parse_success']}`, "
                f"finish_reason=`{r['finish_reason']}`, "
                f"gen_tokens=`{r['generation_token_count']}`, "
                f"pred_answer=`{r['pred_answer']}`"
            )
        # Per-case rule hits
        hits = []
        for rule_key, rule_label in (
            ("rule1_artifact_fix", "Rule 1 (artifact-fix)"),
            ("rule2_token_budget_fix", "Rule 2 (token-budget-fix)"),
            ("rule3_parser_issue", "Rule 3 (parser issue)"),
            ("rule4_reasoning_failure", "Rule 4 (reasoning failure)"),
        ):
            ids = diagnosis[rule_key]["problem_ids"]
            if any(i == pid or i.startswith(f"{pid}[") for i in ids):
                hits.append(rule_label)
        if hits:
            L.append(f"- Rule hits: {', '.join(hits)}")
        L.append("")

    # -----------------------------------------------------------------
    # 6. Diagnosis (5 judgment rules)
    # -----------------------------------------------------------------
    L.append("## 6. Diagnosis")
    L.append("")
    if pattern == "PENDING":
        L.append(diagnosis["rule5_rationale"])
        L.append("")
    else:
        for rule_key, rule_title in (
            ("rule1_artifact_fix", "Rule 1 - Artifact-fix signal (baseline FAIL, old adapter, 2048 -> A CORRECT, B3 adapter, 2048)"),
            ("rule2_token_budget_fix", "Rule 2 - Token-budget-fix signal (B3 adapter constant; A FAIL -> B/C fixes or resolves length)"),
            ("rule3_parser_issue", "Rule 3 - Parser-issue signal (eos+unparsed, or gold text present but is_correct=False)"),
            ("rule4_reasoning_failure", "Rule 4 - Reasoning-failure signal (eos+parsed+wrong, gold not present anywhere in output)"),
        ):
            r = diagnosis[rule_key]
            L.append(f"**{rule_title}**")
            L.append("")
            L.append(f"- Count: {r['count']}")
            for d in r["detail"]:
                L.append(f"  - {d}")
            L.append("")

        L.append("**Rule 5 - Precedence (A > B > C > D) -> overall pattern**")
        L.append("")
        L.append(f"- Overall pattern: **{pattern_titles.get(pattern, pattern)}**")
        L.append(f"- Rationale: {diagnosis['rule5_rationale']}")
        L.append("")

    L.append(
        "**Caveats carried forward from Stage 1 (Section 3), regardless of "
        "the pattern selected above:**"
    )
    L.append("")
    for key in ("artifact_mismatch", "prompt_harness_mismatch", "quantization_mismatch"):
        L.append(f"- `{key}` = **{cav.get(key, 'unknown')}** - {MISMATCH_MEANINGS[key]}")
    L.append("")
    L.append(
        "Even if Pattern A/B/C is selected (i.e. a config-only fix resolves "
        "this subset), the `prompt_harness_mismatch` and "
        "`quantization_mismatch` differences vs. the original 0.86 LB run "
        "remain unaddressed. A full re-run with the B3 adapter at the "
        "chosen max_new_tokens is therefore necessary before drawing "
        "conclusions about the overall Phase 3 score - this subset "
        "comparison only isolates the EFFECT of the adapter path and "
        "max_new_tokens relative to each other, not the absolute score "
        "that would result on the real Kaggle/T4x2 + always-4-bit + "
        "no-ChatML configuration of `phase3_run_golden_validation.py`."
    )
    L.append("")

    # -----------------------------------------------------------------
    # 7. Recommendation
    # -----------------------------------------------------------------
    L.append("## 7. Recommendation")
    L.append("")
    recs = {
        "A": (
            "Adopt the B3 adapter (`{b3}`) as `phase3_run_golden_validation.py`'s "
            "default `--adapter` / `ADAPTER_PATH` for the next full "
            "n=20-30 validation run, keeping max_new_tokens=2048 (Golden "
            "default) and logprob OFF, exactly as in condition A. Re-run "
            "the FULL validation set (not just this subset) with this one "
            "change before considering any further change."
        ),
        "B": (
            "Increase `GOLDEN_GENERATION_CONFIG['max_new_tokens']` from 2048 "
            "toward {budget} for the categories shown in Rule 2's detail "
            "above (re-run the FULL validation set with the B3 adapter "
            "(`{b3}`) AND this single max_new_tokens change). If only "
            "condition C (6144) fixed the escalated cases, prefer 4096 "
            "first and re-measure before jumping to 6144, since 6144 "
            "roughly triples per-problem latency vs. 2048."
        ),
        "C": (
            "Do NOT change max_new_tokens or the adapter path based on this "
            "subset alone. Instead, file the specific `extract_answer` / "
            "`normalize_answer` / `answers_match` failure mode(s) listed "
            "under Rule 3 above as a parser bugfix ticket. Per the "
            "prohibition list (\"prompt改善やparser改善を同時に入れない\"), "
            "this fix must be applied and validated on its own (not bundled "
            "with the Task 4 adapter-path change), with a follow-up "
            "before/after comparison limited to the affected case(s)."
        ),
        "D": (
            "No configuration-only change (adapter path, max_new_tokens, "
            "logprob) recovers these subset case(s) with the B3 adapter at "
            "up to {budget} tokens. Treat this subset as evidence of a "
            "genuine model-capability gap on these "
            "categories/subcategories for now. Do not conclude this for the "
            "FULL validation set from this small subset alone - first "
            "re-run the FULL set with the B3 adapter (Pattern A's "
            "recommendation) since Stage 1's `artifact_mismatch=True` means "
            "the ORIGINAL 20% score was measured on the wrong adapter "
            "regardless of this subset's outcome."
        ),
        "PENDING": (
            "Run Task 6 (conditions A and B) - and Task 7 (condition C) if "
            "applicable - then re-run this script (with `--skip-run` if "
            "results already exist on disk) to compute Sections 4-7."
        ),
    }
    max_c = condition_results.get("C", {}).get("predictions")
    budget_used = "6144" if max_c else "4096"
    L.append(recs[pattern].format(b3=b3_adapter_label, budget=budget_used))
    L.append("")
    L.append(
        "**Regardless of the pattern above, this audit explicitly does NOT "
        "proceed to any of the following** (per the task's prohibition "
        "list):"
    )
    L.append("")
    for item in (
        "Additional training/fine-tuning on cryptarithm or bit-manipulation categories",
        "Concise-reasoning SFT (supervised fine-tuning to shorten outputs)",
        "Re-searching the SVD rank-compression map (RANK_MAP / B3_RANK_MAP)",
        "Adapter fusion / merging",
        "Creating a new submission.zip",
        "Submitting to Kaggle",
    ):
        L.append(f"- {item}")
    L.append("")
    L.append(
        "This script also makes NO changes to "
        "`adapter_model.safetensors`, `adapter_config.json`, rank maps, "
        "target_modules, dtype, or LoRA structure, and does not run more "
        "than one variable change per condition relative to "
        "`phase3_run_golden_validation.py`'s defaults (adapter path is "
        "fixed across A/B/C; only max_new_tokens varies between A/B/C)."
    )
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_run_commands_md(
    path: Path,
    args: argparse.Namespace,
    subset_csv: Path,
    commands_run: List[Dict[str, Any]],
    stage1_status: Optional[Dict[str, Any]],
    output_dir: Path,
) -> None:
    L: List[str] = []
    L.append("# Run Commands (Tasks 1-7)")
    L.append("")
    L.append(
        "Exact commands to reproduce this audit end-to-end. Stage 1 and "
        "Stage 2 are separate scripts; Stage 2 (this script) re-invokes "
        "`phase3_run_golden_validation.py` once per condition (A/B/[C])."
    )
    L.append("")

    L.append("## Stage 1 - Adapter artifact audit (Tasks 1-3)")
    L.append("")
    L.append("```bash")
    b3_dest = (stage1_status or {}).get("b3_adapter_dir", args.adapter)
    old_adapter = (stage1_status or {}).get("old_adapter_dir", "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20")
    L.append(
        "python3 phase3_adapter_artifact_audit.py \\\n"
        f"    --b3-dest {b3_dest} \\\n"
        f"    --old-adapter {old_adapter} \\\n"
        f"    --output-dir {output_dir}"
    )
    L.append("```")
    L.append("")
    L.append(
        f"Produces `{output_dir}/phase3_config_audit/STAGE1_STATUS.json` "
        f"(consumed by Stage 2 below) plus the artifact reports and config "
        f"diff referenced in Sections 2-3 of "
        f"`phase3_token_budget_report.md`. STOPS with exit code 1 if the "
        f"extracted B3 adapter fails its Task 2 expected-value check "
        f"(unless `--force` is given)."
    )
    L.append("")

    L.append("## Task 4 - Adapter path (no code change)")
    L.append("")
    L.append(
        f"`phase3_run_golden_validation.py` already accepts `--adapter`. "
        f"Stage 2 below passes `--adapter {b3_dest}` (the B3 adapter "
        f"extracted by Stage 1) for ALL of conditions A/B/C - this is the "
        f"ONLY change relative to the original baseline run, which used "
        f"`--adapter {old_adapter}`."
    )
    L.append("")

    L.append("## Stage 2 - Token budget comparison (Tasks 5-7)")
    L.append("")
    L.append("```bash")
    stage2_cmd = (
        "python3 phase3_token_budget_compare.py \\\n"
        f"    --baseline-predictions {args.baseline_predictions} \\\n"
        f"    --stage1-status {args.stage1_status} \\\n"
        f"    --adapter {args.adapter} \\\n"
        f"    --runner {args.runner} \\\n"
        f"    --problems {args.problems} \\\n"
        f"    --category-map {args.category_map} \\\n"
        f"    --output-dir {args.output_dir}"
    )
    if args.model:
        stage2_cmd += f" \\\n    --model {args.model}"
    if args.skip_run:
        stage2_cmd += " \\\n    --skip-run"
    if not args.run_condition_c:
        stage2_cmd += " \\\n    --no-run-condition-c"
    L.append(stage2_cmd)
    L.append("```")
    L.append("")
    L.append(
        f"This single command performs Task 5 (writes "
        f"`{subset_csv}` if it does not already exist - pass "
        f"`--rebuild-subset` to regenerate it from "
        f"`--baseline-predictions`), Task 6 (conditions A and B), Task 7 "
        f"(condition C for up to `--max-c-cases` "
        f"(default {args.max_c_cases}) cases still `finish_reason=length` "
        f"after condition B), and writes "
        f"`phase3_token_budget_comparison.csv`, "
        f"`phase3_token_budget_report.md`, this file, and "
        f"`reproducibility_notes.md` under `--output-dir`."
    )
    L.append("")
    L.append(
        "To re-run analysis only (no GPU work) once condition predictions "
        "already exist on disk, add `--skip-run`: results already present "
        "under `<output-dir>/condition_*/golden_validation_predictions.jsonl` "
        "are loaded and re-aggregated/re-diagnosed without re-invoking the "
        "GPU runner."
    )
    L.append("")

    L.append("## Per-condition runner invocations (as executed by Stage 2)")
    L.append("")
    for c in commands_run:
        status = "executed" if c["executed"] else "NOT executed (see note)"
        L.append(f"### {c['label']} - {status}")
        L.append("")
        L.append("```bash")
        L.append(c["cmd"])
        L.append("```")
        L.append("")
    L.append(
        "All three invocations differ from each other ONLY in "
        "`--max-new-tokens` and `--output-dir` (and, for condition C, "
        "`--problem-ids-file` points at the smaller escalation subset, "
        "`subset_cases_condition_c.csv`). `--adapter` is identical "
        "(B3 adapter) across A/B/C. `output_scores`/`return_dict_in_generate` "
        "are never passed by any of these commands, so logprob collection "
        "is OFF in every condition - identical to "
        "`phase3_run_golden_validation.py`'s own defaults."
    )
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_reproducibility_notes_md(
    path: Path,
    args: argparse.Namespace,
    stage1_status: Optional[Dict[str, Any]],
    condition_c_subset_csv: Path,
    condition_results: Dict[str, Dict[str, Any]],
) -> None:
    L: List[str] = []
    L.append("# Reproducibility Notes - Token Budget Comparison (Stage 2)")
    L.append("")
    L.append(
        "These notes are specific to `phase3_token_budget_compare.py` "
        "(Tasks 4-7). For the general Phase 3 reproducibility requirements "
        "(seed, determinism, hardware, offline packages), see "
        "`phase3_analysis/reproducibility_notes.md`."
    )
    L.append("")

    L.append("## What is held constant across baseline / A / B / C")
    L.append("")
    L.append(
        "- `temperature=0.0`, `do_sample=False`, `repetition_penalty=1.0`, "
        "stop tokens (`<|endoftext|>`, `<|im_end|>`), and `--seed 42` are "
        "identical across baseline and conditions A/B/C (GOLDEN_GENERATION_CONFIG "
        "is only overridden via `--max-new-tokens`, which changes ONE key)."
    )
    L.append(
        "- The adapter path is identical (B3 adapter) across A/B/C - it "
        "differs ONLY between baseline (old adapter) and A/B/C. This makes "
        "Rule 1 (Section 6) a clean isolation of the adapter-path effect at "
        "fixed max_new_tokens=2048."
    )
    L.append(
        "- `output_scores`/`return_dict_in_generate` are unset (False) in "
        "baseline and A/B/C alike - logprob collection is OFF everywhere in "
        "this comparison, so it cannot confound any of A/B/C vs baseline."
    )
    L.append(
        "- The prompt template, parser (`extract_answer` / "
        "`normalize_answer` / `answers_match`), category map, problem "
        "ordering, and quantization config of "
        "`phase3_run_golden_validation.py` are UNCHANGED and identical "
        "across baseline and A/B/C."
    )
    L.append("")

    L.append("## What is NOT comparable to the original 0.86 LB run")
    L.append("")
    cav = _extract_stage1_caveats(stage1_status)
    if cav.get("available"):
        L.append(
            f"- `prompt_harness_mismatch` = "
            f"**{cav.get('prompt_harness_mismatch')}**: the 0.86 LB run "
            f"applied ChatML chat-template wrapping; A/B/C do not. This "
            f"difference is present in baseline AND A/B/C alike "
            f"(phase3_run_golden_validation.py's harness is unchanged), so "
            f"it does NOT confound the within-script A/B/C/baseline "
            f"comparisons in Sections 4-6 - but it DOES mean none of "
            f"baseline/A/B/C is directly comparable to the 0.86 LB number."
        )
        L.append(
            f"- `quantization_mismatch` = "
            f"**{cav.get('quantization_mismatch')}**: the 0.86 LB run (on "
            f"an RTX Pro 5000, >=50GB VRAM) ran the base model in full "
            f"bf16/fp16 with NO quantization; "
            f"`phase3_run_golden_validation.py` ALWAYS forces 4-bit NF4 "
            f"regardless of GPU. Same reasoning as above: constant across "
            f"baseline/A/B/C here, so it does not confound Sections 4-6, "
            f"but blocks direct comparison to the 0.86 LB number."
        )
        L.append(
            f"- `artifact_mismatch` = **{cav.get('artifact_mismatch')}**: "
            f"this IS the variable Task 4 changes (baseline used "
            f"`{cav.get('old_adapter_dir')}`; A/B/C use "
            f"`{cav.get('b3_adapter_dir')}`). This is the intended, "
            f"isolated comparison (Rule 1)."
        )
    else:
        L.append(
            "- Stage 1 status (STAGE1_STATUS.json) was not found, so the "
            "artifact_mismatch / prompt_harness_mismatch / "
            "quantization_mismatch verdicts could not be loaded. Run "
            "`phase3_adapter_artifact_audit.py` first."
        )
    L.append("")

    L.append("## Resumability")
    L.append("")
    L.append(
        "`phase3_run_golden_validation.py` writes "
        "`golden_validation_predictions.jsonl` incrementally and SKIPS "
        "problem_ids already present in that file on restart. If a "
        "condition's run is interrupted (e.g. Kaggle session timeout), "
        "re-running the SAME command (or this script without "
        "`--rebuild-subset`) resumes from where it left off. For a clean "
        "re-run of a single condition, delete that condition's "
        "`golden_validation_predictions.jsonl` first - do NOT delete or "
        "modify any file under `phase3_config_audit/` or the B3 adapter "
        "directory."
    )
    L.append("")

    L.append("## Condition C (Task 7) escalation subset")
    L.append("")
    c_result = condition_results.get("C")
    if c_result is not None and c_result.get("predictions"):
        c_ids = sorted({r.get("problem_id") for r in c_result["predictions"]})
        L.append(
            f"Condition C ran for {len(c_ids)} problem(s) still "
            f"`finish_reason=length` after condition B (4096 tokens): "
            f"{c_ids}. The exact subset used is recorded in "
            f"`{condition_c_subset_csv}`."
        )
    elif condition_c_subset_csv.exists():
        L.append(
            f"Condition C subset file `{condition_c_subset_csv}` exists "
            f"but condition C predictions were not found - it may not have "
            f"been executed yet (e.g. `--skip-run`)."
        )
    else:
        L.append(
            "Condition C either was not needed (0 subset problems were "
            "still `finish_reason=length` after condition B), was disabled "
            "(`--no-run-condition-c`), or condition B has not been run yet. "
            "See `phase3_token_budget_report.md` Section 4/6 for which case "
            "applies. If condition C ran, its subset is dynamically derived "
            "from condition B's results - re-running condition B with "
            "different results (e.g. after a code change) may change which "
            "problems are escalated to condition C on a subsequent run."
        )
    L.append("")

    L.append("## Tolerances (carried from phase3_analysis/reproducibility_notes.md)")
    L.append("")
    L.append("- +/-1 token in generation length (padding differences) does not affect is_correct.")
    L.append("- +/-0.001 in logprob values (floating point rounding) - N/A here, logprob is OFF.")
    L.append(
        "- GPU kernel non-determinism at bfloat16/4-bit is low under greedy "
        "decoding (do_sample=False) but not strictly zero; if a borderline "
        "case (e.g. a Rule 2/3 'fix') does not reproduce on a re-run, treat "
        "it as inconclusive rather than a hard contradiction, and prefer "
        "the FULL n=20-30 re-run's result."
    )
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _load_existing_subset(subset_csv: Path, baseline_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-hydrate a previously-written subset_cases.csv into the same record
    shape select_subset() produces, by joining back against baseline
    predictions on problem_id (for raw_output / finish_reason / etc. needed
    by diagnose())."""
    baseline_by_id = {r.get("problem_id"): r for r in baseline_records}
    subset: List[Dict[str, Any]] = []
    with subset_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pid = row.get("problem_id")
            base = dict(baseline_by_id.get(pid, {}))
            base["problem_id"] = pid
            base["category"] = row.get("category") or base.get("category", "")
            base["subcategory"] = row.get("subcategory") or base.get("subcategory", "")
            base["gold_answer"] = row.get("gold_answer") or base.get("gold_answer", "")
            base["_selection_bucket"] = row.get("selection_bucket", "")
            subset.append(base)
    return subset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--baseline-predictions", required=True,
        help="Path to the REAL golden_validation_predictions.jsonl from the "
             "original ~20%% (n=20-30) baseline run (OLD adapter, "
             "max_new_tokens=2048). NOT a schema-template file.",
    )
    ap.add_argument(
        "--stage1-status", default=None,
        help="Path to STAGE1_STATUS.json from phase3_adapter_artifact_audit.py "
             "(default: <output-dir>/phase3_config_audit/STAGE1_STATUS.json)",
    )
    ap.add_argument(
        "--adapter", default=DEFAULT_ADAPTER,
        help="B3 adapter directory (Task 4) - used for ALL of conditions A/B/C "
             f"(default: {DEFAULT_ADAPTER}, the default --b3-dest of "
             f"phase3_adapter_artifact_audit.py)",
    )
    ap.add_argument("--runner", default=DEFAULT_RUNNER, help="Path to phase3_run_golden_validation.py")
    ap.add_argument("--problems", default=DEFAULT_PROBLEMS, help="Path to validation problems JSONL")
    ap.add_argument("--category-map", default=DEFAULT_CATEGORY_MAP, help="Path to category_map.csv")
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", ""), help="Base model path (passed through to runner)")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Root output directory (shared with Stage 1)")
    ap.add_argument(
        "--subset-csv", default=None,
        help="Path to subset_cases.csv (default: "
             f"<output-dir>/{SUBSET_SUBDIR_NAME}/subset_cases.csv). If it "
             "already exists, it is REUSED (not regenerated) unless "
             "--rebuild-subset is given.",
    )
    ap.add_argument("--rebuild-subset", action="store_true", help="Regenerate subset_cases.csv even if it already exists")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--python-exe", default=sys.executable, help="Python executable used to invoke --runner")
    ap.add_argument(
        "--skip-run", action="store_true",
        help="Do not invoke the GPU runner for any condition. Still "
             "performs Task 5 (subset build) and writes run_commands.md. "
             "If condition predictions already exist on disk from a prior "
             "real run, they are loaded and the comparison/report are still "
             "produced.",
    )
    ap.add_argument("--run-condition-c", dest="run_condition_c", action="store_true", default=True,
                     help="Run condition C (6144 tokens) for cases still finish_reason=length after condition B (default: on)")
    ap.add_argument("--no-run-condition-c", dest="run_condition_c", action="store_false")
    ap.add_argument("--max-c-cases", type=int, default=2, help="Max cases to escalate to condition C (Task 7)")
    ap.add_argument("--n-bit-manip-noneos", type=int, default=3, help="Task 5 bucket size: bit_manipulation, non-EOS finish_reason")
    ap.add_argument("--n-other-noneos", type=int, default=2, help="Task 5 bucket size: other categories, non-EOS finish_reason")
    ap.add_argument("--n-numeral-correct", type=int, default=2, help="Task 5 bucket size (up to): numeral_conversion, correct")
    ap.add_argument("--n-parse-fail", type=int, default=1, help="Task 5 bucket size: any category, parse_success == False")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage1_status is None:
        args.stage1_status = str(output_dir / STAGE1_STATUS_RELPATH)
    if args.subset_csv is None:
        args.subset_csv = str(output_dir / SUBSET_SUBDIR_NAME / "subset_cases.csv")
    subset_csv = Path(args.subset_csv)

    baseline_path = Path(args.baseline_predictions)
    baseline_records = load_jsonl(baseline_path)
    if not baseline_records:
        print(f"[ERROR] No records loaded from --baseline-predictions {baseline_path}")
        sys.exit(1)
    print(f"[baseline] Loaded {len(baseline_records)} record(s) from {baseline_path}")

    stage1_status = load_stage1_status(Path(args.stage1_status))
    if stage1_status is None:
        print(
            f"[WARNING] Stage1 status not found at {args.stage1_status} - "
            f"artifact_mismatch/prompt_harness_mismatch/quantization_mismatch "
            f"verdicts will be unavailable in the report. Run "
            f"phase3_adapter_artifact_audit.py first if this is unexpected."
        )

    # -----------------------------------------------------------------
    # Task 5: representative subset selection
    # -----------------------------------------------------------------
    if subset_csv.exists() and not args.rebuild_subset:
        print(f"[subset] Reusing existing {subset_csv} (pass --rebuild-subset to regenerate)")
        subset = _load_existing_subset(subset_csv, baseline_records)
        selection_log = [f"Loaded {len(subset)} pre-selected problem(s) from existing {subset_csv}"]
    else:
        subset, selection_log = select_subset(
            baseline_records,
            n_bit_manip_noneos=args.n_bit_manip_noneos,
            n_other_noneos=args.n_other_noneos,
            n_numeral_correct=args.n_numeral_correct,
            n_parse_fail=args.n_parse_fail,
        )
        write_subset_csv(subset, subset_csv)
        print(f"[subset] Wrote {len(subset)} problem(s) to {subset_csv}")

    for line in selection_log:
        print(f"  {line}")

    if not subset:
        print("[ERROR] Subset is empty - cannot proceed.")
        sys.exit(1)

    subset_ids = [s.get("problem_id") for s in subset]

    # -----------------------------------------------------------------
    # Task 6: conditions A (2048) and B (4096), B3 adapter
    # -----------------------------------------------------------------
    condition_results: Dict[str, Dict[str, Any]] = {}
    commands_run: List[Dict[str, Any]] = []
    for letter in ("A", "B"):
        cond_dir = output_dir / CONDITION_DIR_NAMES[letter]
        cmd = build_run_command(
            python_exe=args.python_exe, runner=args.runner, adapter=args.adapter,
            problems=args.problems, category_map=args.category_map,
            subset_csv=subset_csv, max_new_tokens=CONDITION_TOKEN_BUDGETS[letter],
            output_dir=cond_dir, model=args.model or None, seed=args.seed,
        )
        print(f"\n[condition {letter}] {' '.join(cmd)}")
        result = run_condition(cmd, cond_dir, dry_run=args.skip_run)
        condition_results[letter] = result
        commands_run.append({
            "label": f"Condition {letter} ({CONDITION_TOKEN_BUDGETS[letter]} tokens, {len(subset)} problem(s))",
            "cmd": result["cmd"], "executed": result["executed"],
        })
        if result["executed"] and result["returncode"] != 0:
            print(f"[WARNING] condition {letter} runner exited with code {result['returncode']}")
        elif not result["executed"] and not result["predictions"]:
            print(f"[condition {letter}] --skip-run: no prior results found at {result['predictions_path']}")
        elif not result["executed"]:
            print(f"[condition {letter}] --skip-run: loaded {len(result['predictions'])} prior result(s) from {result['predictions_path']}")

    # -----------------------------------------------------------------
    # Task 7: condition C (6144), escalation subset (<= --max-c-cases)
    # -----------------------------------------------------------------
    condition_c_subset_csv = output_dir / SUBSET_SUBDIR_NAME / "subset_cases_condition_c.csv"
    if args.run_condition_c:
        b_preds = condition_results["B"]["predictions"]
        c_candidates = select_condition_c_subset(b_preds, subset_ids, args.max_c_cases)
        if c_candidates:
            for r in c_candidates:
                r = dict(r)
                r["_selection_bucket"] = "condition_c_escalation_still_length_at_4096"
            write_subset_csv(
                [dict(r, _selection_bucket="condition_c_escalation_still_length_at_4096") for r in c_candidates],
                condition_c_subset_csv,
            )
            cond_dir = output_dir / CONDITION_DIR_NAMES["C"]
            cmd = build_run_command(
                python_exe=args.python_exe, runner=args.runner, adapter=args.adapter,
                problems=args.problems, category_map=args.category_map,
                subset_csv=condition_c_subset_csv, max_new_tokens=CONDITION_TOKEN_BUDGETS["C"],
                output_dir=cond_dir, model=args.model or None, seed=args.seed,
            )
            c_ids = [r.get("problem_id") for r in c_candidates]
            print(f"\n[condition C] {len(c_candidates)} case(s) still finish_reason=length at 4096: {c_ids}")
            print(f"[condition C] {' '.join(cmd)}")
            result = run_condition(cmd, cond_dir, dry_run=args.skip_run)
            condition_results["C"] = result
            commands_run.append({
                "label": f"Condition C ({CONDITION_TOKEN_BUDGETS['C']} tokens, {len(c_candidates)} escalated case(s))",
                "cmd": result["cmd"], "executed": result["executed"],
            })
            if result["executed"] and result["returncode"] != 0:
                print(f"[WARNING] condition C runner exited with code {result['returncode']}")
        else:
            if condition_results["B"]["predictions"]:
                print("\n[condition C] Skipped: 0 subset problem(s) still finish_reason=length at 4096 tokens (Task 7 not needed).")
            else:
                print("\n[condition C] Skipped: condition B has no results yet (run without --skip-run, or after condition B completes).")
            commands_run.append({
                "label": "Condition C (6144 tokens)",
                "cmd": "(skipped - see phase3_token_budget_report.md Section 4/6 for reason)",
                "executed": False,
            })
    else:
        commands_run.append({
            "label": "Condition C (6144 tokens)",
            "cmd": "(disabled via --no-run-condition-c)", "executed": False,
        })

    # -----------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------
    old_adapter_label = "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20"
    if stage1_status and stage1_status.get("old_adapter_dir"):
        old_adapter_label = stage1_status["old_adapter_dir"]
    b3_adapter_label = args.adapter

    comparison_rows = build_comparison_rows(subset, baseline_records, condition_results, old_adapter_label, b3_adapter_label)
    comparison_csv_path = output_dir / "phase3_token_budget_comparison.csv"
    write_comparison_csv(comparison_rows, comparison_csv_path)
    print(f"\n[comparison] Wrote {len(comparison_rows)} row(s) to {comparison_csv_path}")

    # -----------------------------------------------------------------
    # Diagnosis (Section 6, 5 rules -> pattern A/B/C/D)
    # -----------------------------------------------------------------
    diagnosis = diagnose(subset, baseline_records, condition_results, stage1_status)
    print(f"\n[diagnosis] overall pattern = {diagnosis['rule5_overall_pattern']}")
    print(f"  {diagnosis['rule5_rationale']}")

    # -----------------------------------------------------------------
    # Reports
    # -----------------------------------------------------------------
    report_path = output_dir / "phase3_token_budget_report.md"
    write_token_budget_report(
        path=report_path, subset=subset, selection_log=selection_log,
        baseline_records=baseline_records, condition_results=condition_results,
        comparison_rows=comparison_rows, diagnosis=diagnosis,
        stage1_status=stage1_status, comparison_csv_path=comparison_csv_path,
        args=args, old_adapter_label=old_adapter_label, b3_adapter_label=b3_adapter_label,
    )
    print(f"[report] Wrote {report_path}")

    run_commands_path = output_dir / "run_commands.md"
    write_run_commands_md(run_commands_path, args, subset_csv, commands_run, stage1_status, output_dir)
    print(f"[run_commands] Wrote {run_commands_path}")

    repro_path = output_dir / "reproducibility_notes.md"
    write_reproducibility_notes_md(repro_path, args, stage1_status, condition_c_subset_csv, condition_results)
    print(f"[reproducibility_notes] Wrote {repro_path}")

    print("\n" + "=" * 70)
    print(f"RESULT: overall pattern = {diagnosis['rule5_overall_pattern']}")
    print(f"See {report_path} for the full 7-section report.")
    print("=" * 70)


if __name__ == "__main__":
    main()
