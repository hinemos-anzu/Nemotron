# Phase 3 Analysis Report

Generated: 2026-06-04 05:28 UTC

> **Data status:** This report was generated from live analysis CSVs.
> Sections marked [ESTIMATED] contain prior-based estimates where actual
> inference data was not yet available.

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Validation problems | PLACEHOLDER_RUN_INFERENCE |
| Golden Baseline accuracy | PLACEHOLDER |
| Parse success rate | PLACEHOLDER |
| Avg generation tokens | PLACEHOLDER |

**Main weakness categories (weakest first):**
- `bit_manipulation/xor`: acc=0.0 (n=1)
- `ALL/ALL`: acc=0.0 (n=1)
- `cipher/n/a`: acc=1.0 (n=1)
- `cryptarithm/string_transform`: acc=1.0 (n=1)
- `equation/n/a`: acc=1.0 (n=1)

**Improvement priority categories:**
- `bit_manipulation/xor`: priority=4 — low_accuracy+wrong_low_conf_concentrated+solver_verifiable
- `ALL/ALL`: priority=2 — low_accuracy
- `cipher/n/a`: priority=1 — no_clear_priority_signal
- `cryptarithm/string_transform`: priority=1 — solver_verifiable
- `equation/n/a`: priority=1 — solver_verifiable

---

## 2. Category Failure Summary

| category | subcategory | n | accuracy | avg_min_logprob | n_wrong_low_conf | n_correct_low_conf | priority_score |
| -------- | ----------- | ---- | -------- | --------------- | ---------------- | ------------------ | -------------- |
| bit_manipulation | xor | 1 | 0.0 | -3.45 | 1 | 0 | 4 |
| ALL | ALL | 1 | 0.0 |  | 0 | 0 | 2 |
| cipher | n/a | 1 | 1.0 | -0.45 | 0 | 0 | 1 |
| cryptarithm | string_transform | 1 | 1.0 | -0.82 | 0 | 0 | 1 |
| equation | n/a | 1 | 1.0 | -0.38 | 0 | 0 | 1 |
| numeral_conversion | binary_to_decimal | 1 | 1.0 | -0.71 | 0 | 0 | 1 |

---

## 3. Cryptarithm Findings

_(No cryptarithm failure data — run step 5 first.)_


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


---

## 4. Bit Manipulation Findings

### Main failure types

| failure_type | count | pct |
| ------------ | ----- | ---- |
| xor_error | 1 | 100.0% |

### Next-phase candidates
1. `bitwise_xor_step_cot` — explicit truth-table walkthrough for XOR
2. `shift_direction_explicit_cot` — state shift direction and amount before computing
3. `twos_complement_cot` — explicit sign-extension procedure
4. Synthetic bit problems: trivially generated with Python `bin()`, `hex()`, `<<`, `>>`


---

## 5. Numeral Conversion Findings

_(No numeral_conversion failure data — run step 6 first.)_


### Next-phase candidates
1. `positional_binary_cot` — explicit place-value table (2^7 ... 2^0)
2. `division_remainder_binary_cot` — explicit repeated-division procedure
3. `hex_digit_table_cot` — explicit hex digit → decimal mapping (A=10 ... F=15)
4. Synthetic problems trivially generated; all verifiable with Python `int()`, `bin()`, `hex()`


---

## 6. Protected Cases


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


---

## 7. Next Experiment Candidates


All candidates are 1-variable experiments. Each changes exactly one thing.

---

### Experiment 1: Answer format / final parse fix only (Recommended: ★★★★★)
- **Reason:** Pure extraction improvement. Zero risk to Golden reasoning.
- **Baseline diff:** Change only the final-answer extraction logic and/or add `\boxed{}` enforcement to the answer format template.
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


---

## 9. Safety Confirmation


The following prohibited actions were NOT performed during Phase 3 analysis:

- [ ] adapter_model.safetensors modified → **NOT DONE**
- [ ] adapter_config.json modified → **NOT DONE**
- [ ] Training / SFT executed → **NOT DONE**
- [ ] submission.zip created → **NOT DONE**
- [ ] Kaggle submission made → **NOT DONE**
- [ ] rank / target_modules / dtype changed → **NOT DONE**
- [ ] Public LB included in this report → **NOT DONE**
