# Phase 3 Run Commands

## Execution environment
- Platform: Kaggle GPU (T4 or P100)
- Python: 3.10+
- CUDA: available
- Internet: OFF (competition requirement)

## Script versions
- `phase3_build_category_map.py` — v1.0 (committed in this branch)
- `phase3_run_golden_validation.py` — v1.0 (READ-ONLY; adapter not modified)
- `phase3_extract_logprob.py` — v1.0
- `phase3_analyze_category_failures.py` — v1.0
- `phase3_classify_cryptarithm_failures.py` — v1.0
- `phase3_classify_bit_numeral_failures.py` — v1.0
- `phase3_make_recommendation.py` — v1.0

## Git state at time of execution
```
git rev-parse HEAD: [TO BE FILLED AFTER ACTUAL RUN]
git branch: claude/upbeat-galileo-gqBKp
```

---

## Step 1: Build category map

```bash
# Input: competition validation problems
python phase3_build_category_map.py \
  --input /kaggle/input/nvidia-nemotron-model-reasoning-challenge/problems.jsonl \
          /kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv \
  --output phase3_analysis/category_map.csv \
  --labeled-output phase3_analysis/validation_set_labeled.csv
```

**Input paths:**
- Primary: `/kaggle/input/nvidia-nemotron-model-reasoning-challenge/problems.jsonl`
- Fallback: `/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv`

**Expected output:**
- `phase3_analysis/category_map.csv` — N rows where N = total validation problems
- `phase3_analysis/validation_set_labeled.csv` — same N rows with question/answer

---

## Step 2: Run Golden validation inference

```bash
# IMPORTANT: --adapter points to EXISTING adapter — read-only
# Do NOT pass --output to any adapter path
python phase3_run_golden_validation.py \
  --adapter /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20 \
  --model   /kaggle/input/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default \
  --problems /kaggle/input/nvidia-nemotron-model-reasoning-challenge/problems.jsonl \
  --category-map phase3_analysis/category_map.csv \
  --output-dir phase3_analysis/ \
  --seed 42

# First do a dry run to verify config
python phase3_run_golden_validation.py \
  --adapter /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20 \
  --model   /kaggle/input/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default \
  --problems /kaggle/input/nvidia-nemotron-model-reasoning-challenge/problems.jsonl \
  --category-map phase3_analysis/category_map.csv \
  --output-dir phase3_analysis/ \
  --seed 42 \
  --dry-run
```

**Adapter path (read-only):**
- `/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20`

**Model path:**
- `/kaggle/input/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`

**Validation problems path:**
- `/kaggle/input/nvidia-nemotron-model-reasoning-challenge/problems.jsonl`

**Generation config (matching Golden Baseline):**
```json
{
  "max_new_tokens": 2048,
  "temperature": 0.0,
  "do_sample": false,
  "repetition_penalty": 1.0
}
```

**Seed:** 42

**Expected output:**
- `phase3_analysis/golden_validation_predictions.jsonl`
- `phase3_analysis/golden_validation_summary.csv`

**Estimated runtime:** 45–90 minutes on T4 GPU

---

## Step 3: Extract logprob

```bash
# Mode A: If predictions.jsonl has inline token_logprobs field (populated in step 2 with --save-logprobs)
python phase3_extract_logprob.py \
  --predictions phase3_analysis/golden_validation_predictions.jsonl \
  --output phase3_analysis/min_logprob_summary.csv \
  --mode inline

# Mode B: Re-run inference with logprob scoring (adds ~2x time)
python phase3_extract_logprob.py \
  --predictions phase3_analysis/golden_validation_predictions.jsonl \
  --adapter /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20 \
  --model   /kaggle/input/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default \
  --output phase3_analysis/min_logprob_summary.csv \
  --mode rerun \
  --seed 42
```

**Preferred mode:** inline (faster; requires step 2 to save logprobs)
**Fallback mode:** rerun (always works; takes 2x longer)

---

## Step 4: Aggregate category failures

```bash
python phase3_analyze_category_failures.py \
  --predictions phase3_analysis/golden_validation_predictions.jsonl \
  --logprob phase3_analysis/min_logprob_summary.csv \
  --output phase3_analysis/category_failure_summary.csv
```

---

## Step 5: Classify cryptarithm failures

```bash
# Standard: wrong predictions only
python phase3_classify_cryptarithm_failures.py \
  --predictions phase3_analysis/golden_validation_predictions.jsonl \
  --logprob     phase3_analysis/min_logprob_summary.csv \
  --output      phase3_analysis/failure_cases_cryptarithm.csv \
  --failure-type-summary phase3_analysis/failure_type_summary.csv

# Include fragile correct predictions (low logprob)
python phase3_classify_cryptarithm_failures.py \
  --predictions phase3_analysis/golden_validation_predictions.jsonl \
  --logprob     phase3_analysis/min_logprob_summary.csv \
  --output      phase3_analysis/failure_cases_cryptarithm.csv \
  --failure-type-summary phase3_analysis/failure_type_summary.csv \
  --include-correct-low-logprob
```

---

## Step 6: Classify bit/numeral failures

```bash
python phase3_classify_bit_numeral_failures.py \
  --predictions phase3_analysis/golden_validation_predictions.jsonl \
  --logprob     phase3_analysis/min_logprob_summary.csv \
  --output-bit  phase3_analysis/failure_cases_bit_manipulation.csv \
  --output-numeral phase3_analysis/failure_cases_numeral_conversion.csv \
  --failure-type-summary phase3_analysis/failure_type_summary.csv
```

---

## Step 7: Generate recommendation

```bash
python phase3_make_recommendation.py \
  --category-failure phase3_analysis/category_failure_summary.csv \
  --failure-type     phase3_analysis/failure_type_summary.csv \
  --summary          phase3_analysis/golden_validation_summary.csv \
  --output           phase3_analysis/phase3_recommendation.md
```

---

## Generated artefacts (after full run)

| File | Size (estimate) | Notes |
|------|-----------------|-------|
| `phase3_analysis/category_map.csv` | ~500KB | N rows |
| `phase3_analysis/validation_set_labeled.csv` | ~800KB | N rows with text |
| `phase3_analysis/golden_validation_predictions.jsonl` | ~200MB | Full raw outputs |
| `phase3_analysis/golden_validation_summary.csv` | ~5KB | Category stats |
| `phase3_analysis/min_logprob_summary.csv` | ~50KB | Per-sample logprobs |
| `phase3_analysis/category_failure_summary.csv` | ~10KB | Priority table |
| `phase3_analysis/failure_type_summary.csv` | ~5KB | Failure type counts |
| `phase3_analysis/failure_cases_cryptarithm.csv` | ~2MB | Detailed cases |
| `phase3_analysis/failure_cases_bit_manipulation.csv` | ~1MB | Detailed cases |
| `phase3_analysis/failure_cases_numeral_conversion.csv` | ~500KB | Detailed cases |
| `phase3_analysis/phase3_recommendation.md` | ~30KB | Final report |

---

## Execution timestamps

| Step | Start | End | Status |
|------|-------|-----|--------|
| Step 1: category map | [NOT YET RUN] | - | PENDING |
| Step 2: validation inference | [NOT YET RUN] | - | PENDING |
| Step 3: logprob extraction | [NOT YET RUN] | - | PENDING |
| Step 4: category aggregation | [NOT YET RUN] | - | PENDING |
| Step 5: cryptarithm classification | [NOT YET RUN] | - | PENDING |
| Step 6: bit/numeral classification | [NOT YET RUN] | - | PENDING |
| Step 7: recommendation generation | [NOT YET RUN] | - | PENDING |

---

## Environment variables to set on Kaggle

```bash
export ADAPTER_PATH="/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20"
export MODEL_PATH="/kaggle/input/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
```

---

## Prohibited actions checklist

Before running each step, verify:
- [ ] Adapter path is an INPUT path (read-only)
- [ ] No `save_file()` call targets adapter directory
- [ ] No training loop is invoked
- [ ] No `submission.zip` is created
- [ ] Kaggle submit button is NOT pressed
