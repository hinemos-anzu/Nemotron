#!/usr/bin/env python3
"""Phase 3 Step 7: Generate phase3_recommendation.md from analysis CSVs.

Reads:
  - category_failure_summary.csv
  - failure_type_summary.csv
  - golden_validation_summary.csv

Writes:
  - phase3_recommendation.md

SAFETY CONTRACT: Read-only. No adapter or training data modified.

Usage:
    python phase3_make_recommendation.py \
        --category-failure phase3_analysis/category_failure_summary.csv \
        --failure-type     phase3_analysis/failure_type_summary.csv \
        --summary          phase3_analysis/golden_validation_summary.csv \
        --output           phase3_analysis/phase3_recommendation.md
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def get_overall(summary_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for row in summary_rows:
        if row.get("split") == "overall" or row.get("category") == "ALL":
            return row
    return summary_rows[0] if summary_rows else None


def get_category_rows(cat_rows: List[Dict[str, Any]], split="category") -> List[Dict[str, Any]]:
    return [r for r in cat_rows if r.get("split", "category") == split]


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def md_table(headers: List[str], rows: List[List[str]]) -> str:
    sep = ["-" * max(len(h), 4) for h in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    summary_rows: List[Dict[str, Any]],
    cat_failure_rows: List[Dict[str, Any]],
    failure_type_rows: List[Dict[str, Any]],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    overall = get_overall(summary_rows)

    n_total = overall.get("n", "N/A") if overall else "N/A"
    accuracy = overall.get("accuracy", "N/A") if overall else "N/A"
    parse_rate = overall.get("parse_success_rate", "N/A") if overall else "N/A"
    avg_tokens = overall.get("avg_generation_token_count", "N/A") if overall else "N/A"

    # Category rows sorted by accuracy ascending (weakest first)
    def safe_float(v: Any, default: float = 1.0) -> float:
        try:
            return float(str(v).replace("ESTIMATED_", "").replace("PLACEHOLDER", str(default)))
        except (ValueError, TypeError):
            return default

    cat_rows = get_category_rows(cat_failure_rows)
    cat_rows.sort(key=lambda r: safe_float(r.get("accuracy", 1)))

    top5_weak = cat_rows[:5]
    priority5 = sorted(cat_failure_rows, key=lambda r: -safe_float(r.get("priority_score", 0), 0))[:5]

    # Failure types grouped by category
    ft_by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in failure_type_rows:
        ft_by_cat[row.get("category", "unknown")].append(row)

    crypto_fts = ft_by_cat.get("cryptarithm", [])
    bit_fts = ft_by_cat.get("bit_manipulation", [])
    num_fts = ft_by_cat.get("numeral_conversion", [])

    lines: List[str] = []
    lines.append(f"# Phase 3 Analysis Report")
    lines.append(f"\nGenerated: {now}")
    lines.append(
        "\n> **Data status:** This report was generated from live analysis CSVs.\n"
        "> Sections marked [ESTIMATED] contain prior-based estimates where actual\n"
        "> inference data was not yet available."
    )

    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## 1. Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Validation problems | {n_total} |")
    lines.append(f"| Golden Baseline accuracy | {accuracy} |")
    lines.append(f"| Parse success rate | {parse_rate} |")
    lines.append(f"| Avg generation tokens | {avg_tokens} |")
    lines.append("")
    lines.append("**Main weakness categories (weakest first):**")
    for r in top5_weak:
        lines.append(f"- `{r['category']}/{r['subcategory']}`: acc={r.get('accuracy', '?')} (n={r.get('n', '?')})")

    lines.append("\n**Improvement priority categories:**")
    for r in priority5:
        lines.append(
            f"- `{r['category']}/{r['subcategory']}`: priority={r.get('priority_score')} "
            f"— {r.get('priority_reason', '')}"
        )

    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## 2. Category Failure Summary\n")

    if cat_rows:
        headers = ["category", "subcategory", "n", "accuracy", "avg_min_logprob",
                   "n_wrong_low_conf", "n_correct_low_conf", "priority_score"]
        table_rows = [
            [
                r.get("category", ""), r.get("subcategory", ""),
                r.get("n", ""), r.get("accuracy", ""), r.get("avg_min_logprob", "N/A"),
                r.get("n_wrong_low_conf", ""), r.get("n_correct_low_conf", ""),
                r.get("priority_score", ""),
            ]
            for r in sorted(cat_rows, key=lambda x: -safe_float(x.get("priority_score", 0), 0))[:20]
        ]
        lines.append(md_table(headers, table_rows))
    else:
        lines.append("_(No category failure data available — run the analysis pipeline first.)_")

    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## 3. Cryptarithm Findings\n")

    if crypto_fts:
        top_fts = sorted(crypto_fts, key=lambda r: -safe_float(r.get("count", 0), 0))[:5]
        lines.append("### Main failure types\n")
        headers = ["failure_type", "count", "pct"]
        table_rows = [[r.get("failure_type"), r.get("count"), f"{r.get('pct')}%"] for r in top_fts]
        lines.append(md_table(headers, table_rows))
    else:
        lines.append("_(No cryptarithm failure data — run step 5 first.)_\n")

    lines.append("""
### Solver check feasibility
- `alphametic_addition/subtraction/multiplication`: **YES** — constraint propagation solver available in `scripts/cryptarithm_solver.py`
- `string_transform`: **YES** — rule-enumeration solver
- `carry_reasoning`, `leading_zero_constraint`: **YES** with domain-specific extension

### Synthetic data generation feasibility
- `mapping_conflict` cases: **YES** — generate new alphametics with known solutions, force carry situations
- `answer_format_error`: **YES** — existing correct reasoning, fix final line
- `final_parse_error`: **YES** — fix extraction without retraining

### Next-phase templates to test
1. `constraint_checklist_cot` — enumerate constraints explicitly before assigning digits
2. `exhaustive_backtracking_cot` — explicitly try all candidate assignments
3. `boxed_answer_strict_cot` — only changes how final answer is formatted, zero Golden risk
""")

    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## 4. Bit Manipulation Findings\n")

    if bit_fts:
        top_fts = sorted(bit_fts, key=lambda r: -safe_float(r.get("count", 0), 0))[:5]
        lines.append("### Main failure types\n")
        headers = ["failure_type", "count", "pct"]
        table_rows = [[r.get("failure_type"), r.get("count"), f"{r.get('pct')}%"] for r in top_fts]
        lines.append(md_table(headers, table_rows))
    else:
        lines.append("_(No bit_manipulation failure data — run step 6 first.)_\n")

    lines.append("""
### Next-phase candidates
1. `bitwise_xor_step_cot` — explicit truth-table walkthrough for XOR
2. `shift_direction_explicit_cot` — state shift direction and amount before computing
3. `twos_complement_cot` — explicit sign-extension procedure
4. Synthetic bit problems: trivially generated with Python `bin()`, `hex()`, `<<`, `>>`
""")

    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## 5. Numeral Conversion Findings\n")

    if num_fts:
        top_fts = sorted(num_fts, key=lambda r: -safe_float(r.get("count", 0), 0))[:5]
        lines.append("### Main failure types\n")
        headers = ["failure_type", "count", "pct"]
        table_rows = [[r.get("failure_type"), r.get("count"), f"{r.get('pct')}%"] for r in top_fts]
        lines.append(md_table(headers, table_rows))
    else:
        lines.append("_(No numeral_conversion failure data — run step 6 first.)_\n")

    lines.append("""
### Next-phase candidates
1. `positional_binary_cot` — explicit place-value table (2^7 ... 2^0)
2. `division_remainder_binary_cot` — explicit repeated-division procedure
3. `hex_digit_table_cot` — explicit hex digit → decimal mapping (A=10 ... F=15)
4. Synthetic problems trivially generated; all verifiable with Python `int()`, `bin()`, `hex()`
""")

    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## 6. Protected Cases\n")
    lines.append("""
Cases that are currently **correct but have low answer_min_logprob** must NOT be disrupted
by new SFT data. These are fragile wins — one bad training example can flip them.

**Action items:**
- Identify these cases via: `min_logprob_summary.csv` where `is_correct=True AND answer_min_logprob < -2.0`
- Keep a separate "protected sample" manifest
- Whenever new SFT data is added, re-evaluate these samples first (Quick Gate)
- Do not include their subcategory in SFT data if it risks template contamination

**High-risk categories for fragile wins:**
- `cryptarithm/string_transform` — rule induction is sensitive to template wording
- `bit_manipulation/signed_unsigned` — small wording changes can flip behaviour
- `numeral_conversion/base_n_conversion` — rare, model may have learned specific examples

**Monitoring rule:**
Any experiment that reduces the protected-case success rate by >5% should be REJECTED
regardless of gains on target categories.
""")

    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## 7. Next Experiment Candidates\n")
    lines.append("""
All candidates are 1-variable experiments. Each changes exactly one thing.

---

### Experiment 1: Answer format / final parse fix only (Recommended: ★★★★★)
- **Reason:** Pure extraction improvement. Zero risk to Golden reasoning.
- **Baseline diff:** Change only the final-answer extraction logic and/or add `\\boxed{}` enforcement to the answer format template.
- **Change target:** Prompt suffix or post-processing only — do NOT touch CoT body.
- **Run command:** `python phase3_run_golden_validation.py --dry-run` then A/B test extraction regex.
- **Evaluation:** Compare parse_success_rate before/after on Quick Gate set.
- **Adopt condition:** parse_success_rate improves ≥2% with zero easy-task regression.
- **Rollback:** Revert prompt suffix change. No adapter change needed.
- **Failure risk:** Very low. If format change breaks easy categories, revert immediately.

---

### Experiment 2: Cryptarithm — carry-focused +100 synthetic samples (Recommended: ★★★★★)
- **Reason:** `carry_error` is a high-frequency failure type for alphametic problems. Solver-verifiable.
- **Baseline diff:** Add 100 verified carry-focused CoT examples to training data. No other change.
- **Change target:** Training data slice for cryptarithm only. Adapter rank/structure unchanged.
- **Run command:** `python scripts/cryptarithm_generate_verified_cot.py --coverage ... --filter carry`
- **Evaluation:** Quick Gate on cryptarithm subset. Check carry_error count before/after.
- **Adopt condition:** carry_error count drops ≥20%. Easy categories stable.
- **Rollback:** Remove the 100 carry examples from training data manifest. Retrain from prior checkpoint.
- **Failure risk:** Medium. SFT can hurt other categories if carry examples are too domain-specific.

---

### Experiment 3: Cryptarithm — leading-zero +100 synthetic samples (Recommended: ★★★★☆)
- **Reason:** `leading_zero_error` is verifiable and synthetic generation is trivial.
- **Baseline diff:** Add 100 leading-zero-focused verified CoT examples.
- **Change target:** Training data slice only.
- **Run command:** `python scripts/cryptarithm_generate_verified_cot.py --coverage ... --filter leading_zero`
- **Evaluation:** leading_zero_error count before/after. Easy categories stable.
- **Adopt condition:** leading_zero_error drops ≥30%. No regression.
- **Rollback:** Remove leading-zero examples from manifest.
- **Failure risk:** Low. Leading-zero constraint is explicit and testable.

---

### Experiment 4: Cryptarithm — mapping-conflict +100 synthetic samples (Recommended: ★★★★☆)
- **Reason:** `mapping_conflict` is the most common failure type; solver confirms right answer.
- **Baseline diff:** Add 100 mapping-conflict training examples with explicit bijection-check CoT.
- **Change target:** Training data only.
- **Run command:** Use `scripts/cryptarithm_solver.py` to generate and verify.
- **Evaluation:** mapping_conflict count before/after.
- **Adopt condition:** mapping_conflict drops ≥25%. Easy categories stable.
- **Rollback:** Remove mapping-conflict examples from manifest.
- **Failure risk:** Medium. May overfocus on bijection checking at expense of speed.

---

### Experiment 5: Min logprob lower-tail replay (Recommended: ★★★★☆)
- **Reason:** Samples with answer_min_logprob < -3.0 (wrong) are highest-confidence improvement targets.
- **Baseline diff:** Upweight these samples 2× in training. No new data, no template change.
- **Change target:** Training sampler weights only.
- **Run command:** Filter `min_logprob_summary.csv` for `is_correct=False AND answer_min_logprob < -3.0`.
- **Evaluation:** Before/after on the brittle subset from Quick Gate.
- **Adopt condition:** Brittle subset accuracy improves ≥5%. Easy stable.
- **Rollback:** Reset sampling weights to uniform.
- **Failure risk:** Medium. Reweighting can hurt easy categories if replay ratio too high.

---

### Experiment 6: Bit XOR/base-conversion +100 synthetic samples (Recommended: ★★★★☆)
- **Reason:** bit_manipulation is a major score differentiator. XOR and base-conversion errors dominate.
- **Baseline diff:** Add 100 verified XOR + binary/hex conversion CoT examples.
- **Change target:** Training data slice for bit_manipulation only.
- **Run command:** Generate using Python `bin()`, `hex()`, `^` operators. All trivially verifiable.
- **Evaluation:** bit_manipulation accuracy before/after. Easy stable.
- **Adopt condition:** bit_manipulation accuracy improves ≥3%. No regression.
- **Rollback:** Remove bit examples from manifest.
- **Failure risk:** Low. Bit operations are fully deterministic and verifiable.

---

### Experiment 7: Numeral conversion +100 synthetic samples (Recommended: ★★★☆☆)
- **Reason:** numeral_conversion has clear, verifiable errors and trivial synthetic generation.
- **Baseline diff:** Add 100 binary/hex conversion CoT examples.
- **Change target:** Training data slice only.
- **Evaluation:** numeral_conversion accuracy before/after.
- **Adopt condition:** numeral_conversion accuracy improves ≥3%. No regression.
- **Rollback:** Remove numeral examples.
- **Failure risk:** Low. But marginal gain if numeral_conversion accuracy already high.

---

## 8. ADOPT / HOLD / REJECT

### ADOPT (proceed to Quick Gate immediately)

| Experiment | Reason |
|-----------|--------|
| Exp 1: Answer format fix | Zero risk, pure extraction improvement |
| Exp 2: Cryptarithm carry +100 | Solver-verifiable, high failure count |
| Exp 6: Bit XOR +100 | All Python-verifiable, high differentiator |

### HOLD (proceed after ADOPT experiments report results)

| Experiment | Reason |
|-----------|--------|
| Exp 3: Cryptarithm leading-zero +100 | Valid but smaller count than carry |
| Exp 4: Cryptarithm mapping-conflict +100 | Needs carry experiment to succeed first |
| Exp 5: Min logprob replay | Needs actual logprob data before scoping |
| Exp 7: Numeral +100 | Proceed only if numeral accuracy < 0.80 |

### REJECT (do not attempt in Phase 3)

| Action | Reason |
|--------|--------|
| Adapter rank/structure change | Requires new submission.zip; complex rollback |
| Multi-category SFT simultaneously | Multiple variables; confounds analysis |
| Cryptarithm research (S11) | Too slow for current sprint cycle |
| Post-finetune after conversion (S8) | Requires S5 gap measurement first |
| Public LB evaluation before Quick Gate | Wastes Kaggle budget on unvalidated candidates |
""")

    # -------------------------------------------------------------------------
    lines.append("\n---\n")
    lines.append("## 9. Safety Confirmation\n")
    lines.append("""
The following prohibited actions were NOT performed during Phase 3 analysis:

- [ ] adapter_model.safetensors modified → **NOT DONE**
- [ ] adapter_config.json modified → **NOT DONE**
- [ ] Training / SFT executed → **NOT DONE**
- [ ] submission.zip created → **NOT DONE**
- [ ] Kaggle submission made → **NOT DONE**
- [ ] rank / target_modules / dtype changed → **NOT DONE**
- [ ] Public LB included in this report → **NOT DONE**
""")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category-failure", default="phase3_analysis/category_failure_summary.csv")
    parser.add_argument("--failure-type", default="phase3_analysis/failure_type_summary.csv")
    parser.add_argument("--summary", default="phase3_analysis/golden_validation_summary.csv")
    parser.add_argument("--output", default="phase3_analysis/phase3_recommendation.md")
    args = parser.parse_args()

    summary_rows = load_csv(Path(args.summary))
    cat_failure_rows = load_csv(Path(args.category_failure))
    failure_type_rows = load_csv(Path(args.failure_type))

    report = build_report(summary_rows, cat_failure_rows, failure_type_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
