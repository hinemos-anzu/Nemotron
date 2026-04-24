# Runtime Gold Refreeze

**Date**: 2026-04-24  
**Branch**: claude/fix-mamba-stub-scynC  
**Base commit**: b1d54ce (claude/nemotron-kaggle-baseline-BE6bY)  
**Purpose**: Restore safe starting point (baseline 0.85) with beam-search prevention

---

## What Changed from b1d54ce

This branch adds **three minimal safety patches** to the runtime gold. No experiment logic (S2/S3/S4) is included.

### 1. mamba_ssm stub — PyTorch implementation

**File**: `mamba_ssm/ops/triton/layernorm_gated.py`  
(and bundle mirror: `artifacts/kaggle/Nemotron_kaggle_bundle/mamba_ssm/ops/triton/layernorm_gated.py`)

**Before** (b1d54ce — 9 lines, NotImplementedError):
```python
def rms_norm_fn(*args, **kwargs):
    raise NotImplementedError("rms_norm_fn: stub only — should not be called")

def layer_norm_fn(*args, **kwargs):
    raise NotImplementedError("layer_norm_fn: stub only — should not be called")
```

**After** (this branch — full PyTorch impl, 140 lines):
```python
def rmsnorm_fn(x, weight, bias=None, residual=None, ..., prenorm=False, ...):
    # numerically correct RMSNorm via torch.rsqrt(mean(x²))
    ...
rms_norm_fn = rmsnorm_fn  # both spellings available

def layer_norm_fn(x, weight, bias=None, residual=None, ..., prenorm=False, ...):
    # via F.layer_norm
    ...
# RMSNorm and LayerNorm nn.Module classes also provided
```

**Why**: The b1d54ce stubs raised `NotImplementedError` if ever called. With `is_mamba_2_ssm_available = lambda: False`, these were not triggered during baseline execution. However, the Kaggle environment may call them if the patch ever fails to apply. Defense-in-depth: the stub now works correctly even if invoked.

### 2. Generation policy constant

**File**: `kaggle/original-nemotron-asymmetric-svd-26041602.py`

Added after `EXPECTED_ARTIFACTS`:
```python
# ─── generation policy ────────────────────────────────────────────────────────
# NemotronH is a hybrid SSM/Transformer model. SSM state cannot be branched
# across multiple beams, causing generation collapse ("and ,,,..." garbage).
# num_beams MUST remain 1. See reports/beam_search_ban_policy.md for analysis.
_GENERATION_NUM_BEAMS = 1  # FROZEN — do not change
```

### 3. Hard fail guard + explicit num_beams + startup log

**File**: `kaggle/original-nemotron-asymmetric-svd-26041602.py`

Modified `_run_inference()`:
- Guard: raises `RuntimeError` if `_GENERATION_NUM_BEAMS != 1`
- Explicit: `num_beams=_GENERATION_NUM_BEAMS` in `model.generate()`
- Startup: `[baseline][gen_policy] num_beams=1 ...` printed once per run

---

## What Was NOT Changed

- No S2 changes (logprob selection, `_run_inference_minlogprob`, per-sample CSV S2 fields)
- No S3 changes (answer-consistency rerank, `_run_inference_s3`, N_CANDIDATES_S3)
- No S4 changes
- No eval loop changes
- No answer key solver changes
- No model loading changes (ADAPTER_PATH, BASE_MODEL_ID unchanged)
- No data file changes

---

## Bundle

**File**: `artifacts/kaggle/Nemotron_kaggle_bundle.zip`  
**Size**: 47,475 bytes  
**Entries**: 18  

Includes:
- Updated `kaggle/original-nemotron-asymmetric-svd-26041602.py` (with guards)
- Updated `mamba_ssm/ops/triton/layernorm_gated.py` (PyTorch impl)
- All other files unchanged from b1d54ce

Verification log: `logs/generation_config_guard.txt`

---

## Baseline 0.85 Confirmation

**Status**: PENDING_KAGGLE_EXECUTION

Expected Kaggle log on successful run:
```
[baseline] REPO_ROOT inserted into sys.path: /path/to/repo
[baseline] Patched is_mamba_2_ssm_available() -> False
[baseline] Local stub mamba_ssm package detected: ...
[baseline][gen_policy] num_beams=1 do_sample=False max_new_tokens=128 temperature=1.0 — beam search BANNED
...
[baseline] Evaluation complete
```

Expected accuracy: ≥ 0.85 (matching b1d54ce confirmed result)

---

## Next Steps (after 0.85 confirmation)

1. Tag this commit as the new frozen baseline
2. Re-implement multi-candidate S3 using **sequential independent `model.generate()` calls** (not beam search)
3. See `reports/beam_search_ban_policy.md` for the correct implementation pattern
