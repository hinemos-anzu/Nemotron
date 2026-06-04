# Phase 3 Analysis — NVIDIA Nemotron Model Reasoning Challenge

## Purpose

This directory contains the Phase 3 diagnostic analysis for the Golden Baseline
(Public LB 0.86, ~100th rank). The analysis identifies which problem categories
the model fails on, characterises failure types at the token/reasoning level, and
produces prioritised next-experiment candidates.

**No adapter weights, training, or submission artefacts are modified here.**

---

## Data status

| File | Status | Notes |
|------|--------|-------|
| `category_map.csv` | SCHEMA TEMPLATE | Populate by running `phase3_build_category_map.py` on Kaggle |
| `validation_set_labeled.csv` | SCHEMA TEMPLATE | Same script |
| `golden_validation_predictions.jsonl` | SCHEMA TEMPLATE | Populate by running `phase3_run_golden_validation.py` |
| `golden_validation_summary.csv` | SCHEMA TEMPLATE | Same script |
| `min_logprob_summary.csv` | SCHEMA TEMPLATE | Populate by running `phase3_extract_logprob.py` |
| `category_failure_summary.csv` | SCHEMA TEMPLATE | `phase3_analyze_category_failures.py` |
| `failure_type_summary.csv` | SCHEMA TEMPLATE | `phase3_classify_cryptarithm_failures.py` + `phase3_classify_bit_numeral_failures.py` |
| `failure_cases_cryptarithm.csv` | SCHEMA TEMPLATE | `phase3_classify_cryptarithm_failures.py` |
| `failure_cases_bit_manipulation.csv` | SCHEMA TEMPLATE | `phase3_classify_bit_numeral_failures.py` |
| `failure_cases_numeral_conversion.csv` | SCHEMA TEMPLATE | `phase3_classify_bit_numeral_failures.py` |
| `phase3_recommendation.md` | COMPLETE — structure ready | Refresh numeric sections after real data |
| `run_commands.md` | COMPLETE | Update timestamps and paths after actual run |
| `reproducibility_notes.md` | COMPLETE | Review before each Kaggle run |

---

## Classification rules

### Category classification (keyword/rule-based)

Priority order: if multiple rules match, the first matching category wins.

#### 1. cryptarithm
**Primary trigger:** problem `category` field is `cryptarithm_deduce` or `cryptarithm_guess`.
Fallback keyword matches (question text, lowercased):
- `"?"` between two alphanumeric tokens with `->` or `=` in examples
- presence of `examples` list with `input/output` pairs containing only alphabetic tokens

**Subcategory rules (applied in order):**
| subcategory | rule |
|------------|------|
| `alphametic_addition` | operator `+` or `plus` in question/examples |
| `alphametic_subtraction` | operator `-` or `minus` |
| `alphametic_multiplication` | operator `*` or `×` |
| `digit_assignment` | "assign", "digit" in question |
| `carry_reasoning` | "carry" in question or long multi-digit example |
| `leading_zero_constraint` | "leading zero", or answer starts with letter mapped to 0 |
| `string_transform` | no arithmetic operator; pure string transformation |
| `unknown` | none of the above |

#### 2. bit_manipulation
**Triggers:**
- category field contains `bit`
- keywords: `XOR`, `AND`, `OR`, `NOT`, `shift`, `<<`, `>>`, `0b`, `0x`, `binary`, `bitwise`
- examples contain strings of `0` and `1`

**Subcategory rules:**
| subcategory | keywords |
|------------|---------|
| `xor` | XOR, ⊕ |
| `and` | AND, & |
| `or` | OR, \| |
| `shift_left` | <<, shl, shift left |
| `shift_right` | >>, shr, shift right |
| `mask` | mask, bitmask |
| `signed_unsigned` | signed, unsigned, two's complement |
| `binary_arithmetic` | binary addition/subtraction without named bit-op |
| `unknown` | fallback |

#### 3. numeral_conversion
**Triggers:**
- category field contains `numeral` or `conversion`
- keywords: `binary`, `decimal`, `hexadecimal`, `base`, `octal`, `roman`, `convert`
- presence of `0b`, `0x`, `0o` prefix in question

**Subcategory rules:**
| subcategory | keywords |
|------------|---------|
| `binary_to_decimal` | "binary to decimal", "0b" → numeric |
| `decimal_to_binary` | "decimal to binary", numeric → "0b" |
| `hex_to_decimal` | "hex to decimal", "0x" → numeric |
| `decimal_to_hex` | "decimal to hex", numeric → "0x" |
| `base_n_conversion` | "base N", "base-N" where N not 2 or 16 |
| `roman_numeral` | roman, numeral |
| `unknown` | fallback |

#### 4. cipher
**Triggers:** keywords: `cipher`, `encode`, `decode`, `encrypt`, `decrypt`, `caesar`, `shift`, `rotate`, `ROT`

#### 5. equation
**Triggers:** keywords: `solve`, `find x`, `=`, `equation`, `algebra`

#### 6. unit_conversion
**Triggers:** keywords: `km`, `miles`, `kg`, `pounds`, `celsius`, `fahrenheit`, `liters`, `gallons`, `meters`, `feet`

#### 7. rule_induction
**Triggers:** problem contains examples but doesn't match cryptarithm; keyword: `pattern`, `sequence`, `rule`

#### 8. arithmetic
**Triggers:** pure numeric expression without conversion or equation structure

#### 9. logic
**Triggers:** keywords: `true`, `false`, `and`, `or`, `not`, `if then`, `implies`, `logic`

#### 10. other
**Default:** none of the above. If any two categories match simultaneously, set `manual_review_required=True`.

### Confidence scoring
- `high` (≥0.9): category field explicit match
- `medium` (0.7–0.9): single strong keyword match
- `low` (<0.7): multiple weak matches, or fallback

---

## Script execution order

```
# Step 1: build category map from raw validation data
python phase3_build_category_map.py \
  --input /kaggle/input/problems.jsonl \
  --output phase3_analysis/category_map.csv

# Step 2: run inference (read-only — does NOT modify adapter)
python phase3_run_golden_validation.py \
  --adapter /kaggle/input/adapter \
  --model /kaggle/input/model \
  --category-map phase3_analysis/category_map.csv \
  --output-dir phase3_analysis/

# Step 3: extract logprob (requires --return_logprobs in step 2)
python phase3_extract_logprob.py \
  --predictions phase3_analysis/golden_validation_predictions.jsonl \
  --output phase3_analysis/min_logprob_summary.csv

# Step 4: aggregate category failures
python phase3_analyze_category_failures.py \
  --predictions phase3_analysis/golden_validation_predictions.jsonl \
  --logprob phase3_analysis/min_logprob_summary.csv \
  --output phase3_analysis/category_failure_summary.csv

# Step 5: classify cryptarithm failures
python phase3_classify_cryptarithm_failures.py \
  --predictions phase3_analysis/golden_validation_predictions.jsonl \
  --logprob phase3_analysis/min_logprob_summary.csv \
  --output phase3_analysis/failure_cases_cryptarithm.csv \
  --failure-type-summary phase3_analysis/failure_type_summary.csv

# Step 6: classify bit/numeral failures
python phase3_classify_bit_numeral_failures.py \
  --predictions phase3_analysis/golden_validation_predictions.jsonl \
  --logprob phase3_analysis/min_logprob_summary.csv \
  --output-bit phase3_analysis/failure_cases_bit_manipulation.csv \
  --output-numeral phase3_analysis/failure_cases_numeral_conversion.csv

# Step 7: generate recommendation
python phase3_make_recommendation.py \
  --category-failure phase3_analysis/category_failure_summary.csv \
  --failure-type phase3_analysis/failure_type_summary.csv \
  --summary phase3_analysis/golden_validation_summary.csv \
  --output phase3_analysis/phase3_recommendation.md
```

---

## Prohibited actions in this phase

- Do not modify `adapter_model.safetensors`
- Do not modify `adapter_config.json`
- Do not run training or SFT
- Do not create `submission.zip`
- Do not submit to Kaggle
- Do not change rank, target_modules, dtype, or LoRA structure
