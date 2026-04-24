# Beam Search Ban Policy

**Status**: ENFORCED  
**Branch**: claude/fix-mamba-stub-scynC  
**Effective from**: b1d54ce + this patch  
**Enforcement location**: `kaggle/original-nemotron-asymmetric-svd-26041602.py` → `_run_inference()`

---

## Policy

**`num_beams` MUST equal 1 at all times on this branch.**

Any attempt to set `num_beams > 1` will raise `RuntimeError` at inference time (hard fail). This is not a warning; it stops execution.

```python
_GENERATION_NUM_BEAMS = 1  # FROZEN — do not change
```

---

## Rationale

NemotronH (nvidia/nemotron-3-nano-30b-a3b-bf16) is a **hybrid SSM/Transformer model**. The SSM layers (Mamba selective scan) maintain a recurrent state that is fundamentally different from a Transformer's KV cache:

| Property | Transformer KV cache | SSM recurrent state |
|----------|---------------------|---------------------|
| Beam branching | Supported (`reorder_cache`) | **Not supported** |
| State shape | `[batch, heads, seq, dim]` | `[batch, d_state, d_inner]` |
| Independent per sequence? | Yes | Yes (but must be initialized per beam) |

During beam search with `num_beams=2`, `transformers` calls `reorder_cache()` on the `NemotronHHybridDynamicCache`. The SSM state slots are **not initialized for multiple beams**, causing:

1. Warning: `NemotronHHybridDynamicCache ... None was provided`
2. Cross-beam SSM state corruption
3. **Generation collapse**: output degenerates to `"and ,,,..."`, short repetitive tokens, or single tokens repeated

### Evidence

| Experiment | Config | Output |
|------------|--------|--------|
| b1d54ce baseline | `num_beams=1` (implicit greedy) | Correct answers, 0.85 accuracy |
| S2 (beam=4) | `num_beams=4` | 105-108 sec/sample, `"and ,,,..."` collapse |
| S3 (beam=2) | `num_beams=2` | `"and ,,,..."` collapse on QG_001 and others |

Full analysis: `reports/generation_collapse_triage.md`  
Diagnostic scripts: `logs/s3_off_smoke.txt`, `logs/no_adapter_smoke.txt`

---

## What is Allowed

| Technique | Allowed? | Notes |
|-----------|---------|-------|
| Greedy (`num_beams=1, do_sample=False`) | **YES** | Baseline config |
| Temperature sampling (`num_beams=1, do_sample=True`) | YES | Sequential runs only |
| Top-p sampling (`num_beams=1, top_p=0.9`) | YES | Sequential runs only |
| Beam search (`num_beams > 1`) | **NO** | Banned — SSM incompatible |
| `num_return_sequences > 1` in one call | NO | Requires beam search |
| Multiple sequential `model.generate()` calls | YES | Each call gets fresh SSM state |

---

## How to Generate Multiple Candidates (S3 / future)

Use **sequential independent calls**, not beam search:

```python
candidates = []
for i in range(N_CANDIDATES):
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=1,                      # MUST be 1
            do_sample=(i > 0),                # greedy first, sampled rest
            temperature=0.7 if i > 0 else 1.0,
            top_p=0.9 if i > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, input_ids.shape[1]:].tolist()
    candidates.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
```

Each call creates a fresh `NemotronHHybridDynamicCache`, avoiding SSM state corruption.

---

## Guard Implementation

```python
# kaggle/original-nemotron-asymmetric-svd-26041602.py

_GENERATION_NUM_BEAMS = 1  # FROZEN — do not change

def _run_inference(model, tokenizer, problem: str) -> str:
    if _GENERATION_NUM_BEAMS != 1:
        raise RuntimeError(
            f"[baseline] HARD FAIL: _GENERATION_NUM_BEAMS={_GENERATION_NUM_BEAMS} — "
            "beam search is banned on NemotronH (SSM state cannot branch). "
            "See reports/beam_search_ban_policy.md"
        )
    ...
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=_GENERATION_NUM_BEAMS,  # 1
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
```

Startup log (once per run):
```
[baseline][gen_policy] num_beams=1 do_sample=False max_new_tokens=128 temperature=1.0 — beam search BANNED
```
