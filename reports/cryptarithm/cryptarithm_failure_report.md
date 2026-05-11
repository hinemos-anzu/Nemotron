# Cryptarithm Solver Failure Report

## 1. Problem count by category

| Category | Total | Solver correct | Coverage |
|---|---|---|---|
| cryptarithm_deduce | 10 | 9 | 90.0% |
| cryptarithm_guess  | 10 | 10 | 100.0% |
| **Total** | **20** | **19** | **95.0%** |

## 2. Rule coverage breakdown

| Rule | Problems matched | Correct | Accuracy |
|---|---|---|---|
| forward_concat | 5 | 5 | 100% |
| reverse_both | 3 | 3 | 100% |
| reverse_concat | 2 | 2 | 100% |
| reverse_left | 2 | 2 | 100% |
| reverse_right | 2 | 2 | 100% |
| interleave_lr | 2 | 2 | 100% |
| interleave_rl | 2 | 2 | 100% |
| unknown_operator_fallback | 2 | 1 | 50% |

## 3. Parse failures

Total parse failures: 0


## 4. Answer mismatch examples (solver parsed but predicted wrong)

_No answer mismatches (all parsed problems solved correctly or fell through to unknown)._

## 5. Unresolved rules (not matching any example set)

- **unknown_operator_fallback**: 2 problems, 1 correct, 1 unresolved

## 6. Next templates to add

The following rules returned no matches and likely need new templates: `unknown_operator_fallback`.

Suggested additions based on failure patterns:

- **digit_sum_concat**: Combine digit sums of left and right
- **mod_concat**: Apply modulo then concatenate
- **length_conditioned**: Rule changes based on string length of operands
- **operator_symbol_map**: Map operator to a named operation (e.g. ★→XOR)
- **numeric_arithmetic_fallback**: Evaluate as standard arithmetic before string rules

## 7. Verified CoT count

Verified CoT records generated: **19**
(saved to `reports/cryptarithm/cryptarithm_generated_cot_sample.jsonl`)

## 8. Recommendation: proceed to training?

**YES** — Solver coverage is 95.0% (≥70%). Sufficient verified CoT data to proceed with SFT data preparation. Recommend: run `train_sft.py` on `cryptarithm_generated_cot_sample.jsonl` after Planner review.