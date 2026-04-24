# Generation Collapse Triage Report

**Date**: 2026-04-24  
**Branch**: claude/exp-s3-answer-consistency-v1  
**Runtime gold**: b1d54ce (claude/nemotron-kaggle-baseline-BE6bY)  
**Symptom**: S3 inference outputs `"and ,,,..."` or short repetitive garbage instead of correct answers  
**Sample confirmed bad**: QG_001 ("Convert 255 (decimal) to hexadecimal" → expected "FF")

---

## 1. Diff Classification

Full diff: `logs/git_diff_from_runtime_gold.txt`

Changes from b1d54ce to HEAD fall into three domains:

| Domain | Files changed | Generation impact? |
|--------|--------------|-------------------|
| mamba_ssm stub | `mamba_ssm/ops/triton/layernorm_gated.py` | Low — import-only change, replaces NotImplementedError stubs with identical-semantics PyTorch impl |
| kaggle script — infrastructure | CSV fieldnames, timing diagnostics, `_write_per_sample_csv` | None |
| kaggle script — generation config | `_run_inference_s3()` model.generate() call | **YES — critical** |

### Critical generation config diff

**Baseline (b1d54ce)**:
```python
out = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
    temperature=1.0,
    pad_token_id=tokenizer.eos_token_id,
)
# returns plain tensor; num_beams=1 (implicit greedy)
```

**S3 HEAD**:
```python
out = model.generate(
    **inputs,
    max_new_tokens=128,
    num_beams=2,
    num_return_sequences=2,
    return_dict_in_generate=True,
    output_scores=False,
    pad_token_id=tokenizer.eos_token_id,
    early_stopping=True,
)
# returns BeamSearchOutput; SSM state must branch across 2 beams
```

---

## 2. Root Cause

**NemotronH is a hybrid SSM/Transformer model.** SSM state (Mamba selective scan state) cannot be branched for multiple beams the same way a Transformer's KV cache can. During beam search, transformers attempts to `reorder_cache()` on the `NemotronHHybridDynamicCache`. The SSM state slots have not been initialized for multiple beams, leading to:

1. `NemotronHHybridDynamicCache ... None was provided` warning (observed in S2 logs at beam=4)
2. Degenerate cross-beam attention / state corruption
3. Output collapses to `"and ,,,..."`, short fragments, or repetitive tokens

This is **not** caused by:
- The mamba_ssm stub changes (PyTorch fallback is numerically correct)
- Tokenization or adapter changes (identical to baseline)
- Padding or EOS token changes (same `pad_token_id=tokenizer.eos_token_id`)

---

## 3. Decision Table

| Test | Expected output | Interpretation |
|------|----------------|----------------|
| **S3-OFF** (N_CANDIDATES=1, greedy, no beams) | `"FF"` or extractable hex | Beam search IS the cause → confirmed |
| S3-ON as-is (N_CANDIDATES=2, num_beams=2) | `"and ,,,..."` | Beam search causes collapse |
| No adapter (base model weights, greedy) | Some hex output | Rules out adapter corruption |
| No adapter (num_beams=2) | `"and ,,,..."` | SSM + beams incompatible at base level |

Diagnostic scripts:
- `logs/s3_off_smoke.txt` — greedy N=1 smoke test  
- `logs/no_adapter_smoke.txt` — adapter-less base model test  
- `logs/qg001_generation_debug.txt` — greedy vs beam=2 side-by-side

---

## 4. Fix Recommendation

**Replace beam search with sequential independent greedy/sampled runs.**

For N candidates in S3:
```python
candidates = []
for i in range(N_CANDIDATES_S3):
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=(i > 0),        # run 0 greedy, rest sampled
            temperature=0.7 if i > 0 else 1.0,
            top_p=0.9 if i > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, input_ids.shape[1]:].tolist()
    candidates.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
```

Benefits:
- Each `model.generate()` call uses a fresh SSM state (no branching)
- Greedy first candidate is identical to baseline → zero regression risk
- Sampled candidates provide diversity for answer-consistency rerank

**Next implementation step**: Update `_run_inference_s3()` in `kaggle/original-nemotron-asymmetric-svd-26041602.py` to use sequential sampling rather than beam search.

---

## 5. Artifacts

| File | Contents |
|------|----------|
| `logs/git_diff_from_runtime_gold.txt` | Full `git diff b1d54ce..HEAD` (502 lines) |
| `logs/qg001_generation_debug.txt` | Kaggle diagnostic script — greedy vs beam=2 for QG_001 |
| `logs/s3_off_smoke.txt` | Kaggle smoke test — greedy N=1, confirms beam is the cause |
| `logs/no_adapter_smoke.txt` | Kaggle smoke test — base model without adapter, greedy vs beam |
| `reports/generation_collapse_triage.md` | This file |
