# Phase 3 Reproducibility Notes

## Conditions required for deterministic reproduction

### Hard requirements
1. **Same adapter checkpoint** — adapter_model.safetensors must be byte-for-byte identical.
   SHA-256 of the adapter must be logged before and after each run.
2. **Same base model** — model path and checkpoint version must be pinned.
3. **Same seed** — `--seed 42` must be passed consistently. All random seeds in
   `torch.manual_seed`, `transformers.set_seed`, and `numpy.random.seed` must match.
4. **temperature=0.0 / greedy decoding** — do_sample=False. Any sampling introduces
   non-determinism.
5. **Same problem ordering** — problems processed in the same order as the JSONL.
   Do NOT shuffle before inference.
6. **Same generation config** — max_new_tokens, repetition_penalty, stop tokens
   must match Golden Baseline exactly.

### Sources of non-determinism (even with seed=42)
| Source | Risk level | Mitigation |
|--------|-----------|------------|
| GPU kernel non-determinism at bfloat16 | Low (greedy decoding reduces this) | Use `torch.backends.cudnn.deterministic=True` |
| Multi-GPU tensor parallel | Medium | Single GPU only for validation |
| Flash Attention vs vanilla attention | High | Pin attention implementation |
| CUDA version differences across Kaggle sessions | Low | Pin CUDA version in Dockerfile |
| Tokenizer version | Low | Pin transformers version |

### Tolerable non-determinism
- ±1 token in generation length (from padding differences)
- ±0.001 in logprob values (from floating point rounding)
- Does NOT affect is_correct if both runs produce the same answer string

---

## Logprob availability

### HuggingFace transformers (offline Kaggle environment)
- **Available**: Yes, via `output_scores=True` + `return_dict_in_generate=True`
- **Method**: `out.scores` gives per-step logit distributions
- **Cost**: +15-20% memory; +5-10% runtime
- **Limitation**: scores are per-step logits over full vocab; convert to logprob via `log_softmax`

### vLLM (if available)
- **Available**: Conditionally (vLLM may not be pre-installed on Kaggle)
- **Method**: `SamplingParams(logprobs=1)` returns top-k logprobs
- **Advantage**: 3-5x faster than transformers; native logprob API
- **Limitation**: requires vllm package; may conflict with PEFT adapter loading

### Recommended approach for Kaggle
1. Use transformers path (always available)
2. Set `output_scores=True` in step 2
3. Store token logprobs inline in predictions.jsonl
4. Use `--mode inline` in step 3

---

## vLLM vs transformers differences

| Aspect | transformers | vLLM |
|--------|-------------|------|
| Speed | Baseline | 3–5x faster |
| PEFT adapter support | Native (PeftModel) | Requires LoRA merge or custom loading |
| Logprob API | output_scores=True | SamplingParams(logprobs=N) |
| Memory | Higher (separate model+adapter) | Lower (merged) |
| Numerical equivalence | Reference | ~Equal (±0.001 logprob) |

**For Phase 3 analysis: use transformers to ensure exact reproduction of Golden Baseline.**

---

## Kaggle Internet-OFF compatibility

| Step | Internet required? | Offline alternative |
|------|--------------------|---------------------|
| Step 1: category map | No | Runs offline (no network calls) |
| Step 2: inference | No | Model/adapter loaded from /kaggle/input |
| Step 3: logprob | No | Runs offline |
| Step 4-7: analysis | No | Pure CSV/JSONL processing |
| Python packages | Must be pre-installed | Install from offline wheel cache |

**Required packages (pre-install as Kaggle dataset):**
- torch >= 2.0
- transformers >= 4.40
- peft >= 0.10
- safetensors >= 0.4
- accelerate >= 0.28

---

## Failure scenarios and fallback procedures

### Scenario A: problems.jsonl not found
```bash
# Fallback: check alternative competition dataset names
ls /kaggle/input/
# Common alternatives:
#   /kaggle/input/nvidia-nemotron/
#   /kaggle/input/nemotron-reasoning-challenge/
# Update --problems path accordingly
```

### Scenario B: Adapter path not found
```bash
# The adapter path may change between Kaggle dataset versions
ls /kaggle/input/models/huikang/nemotron-adapter/transformers/default/
# Use the highest numbered version directory
```

### Scenario C: OOM during inference
```bash
# Use smaller batch or reduce max_new_tokens for analysis only
python phase3_run_golden_validation.py \
  --adapter ... --model ... \
  --seed 42 \
  --max-new-tokens 1024  # reduce from 2048 for OOM recovery
# Note: shorter outputs may increase parse failure rate
```

### Scenario D: Logprob data missing (inline mode fails)
```bash
# Fall back to rerun mode
python phase3_extract_logprob.py \
  --predictions phase3_analysis/golden_validation_predictions.jsonl \
  --adapter ... --model ... \
  --output phase3_analysis/min_logprob_summary.csv \
  --mode rerun --seed 42
```

### Scenario E: category_map.csv is empty
```bash
# Check if problems.jsonl has expected fields
python -c "
import json
with open('/kaggle/input/.../problems.jsonl') as f:
    rec = json.loads(f.readline())
    print(rec.keys())
"
# If category field is named differently, update
# CRYPTARITHM_CATEGORIES in phase3_build_category_map.py
```

### Scenario F: All accuracy values are 0 (model outputs garbage)
- Most likely cause: adapter key mismatch (model.model vs model.backbone)
- The Golden Baseline notebook handles this with `trained_adapter_key_rename()`
- Verify that the working adapter directory has backbone keys, not model.model keys

---

## Adapters not to touch

The following adapter files must NOT be written to, overwritten, or deleted:
- `/kaggle/input/models/huikang/nemotron-adapter/.../adapter_model.safetensors`
- `/kaggle/input/models/huikang/nemotron-adapter/.../adapter_config.json`

The working adapter (post-SVD surgery) at `/kaggle/working/adapter/` is the
submission artefact and also must not be modified during analysis.

---

## Version history

| Version | Date | Change |
|---------|------|--------|
| v1.0 | 2026-06-04 | Initial Phase 3 framework created |
