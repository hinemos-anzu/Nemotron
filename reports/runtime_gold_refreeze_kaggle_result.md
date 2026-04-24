# Runtime Gold Refreeze — Kaggle Result

**Status**: PENDING_KAGGLE_EXECUTION  
**Branch**: claude/fix-mamba-stub-scynC  
**Commit**: 3b17c3d  
**Base**: b1d54ce (confirmed baseline 0.85)  
**Date**: 2026-04-24

---

## 1. Branch / Commit

| Item | Value |
|------|-------|
| branch | claude/fix-mamba-stub-scynC |
| commit | 3b17c3d |
| base | b1d54ce |
| parent (base) description | feat(kaggle-runtime): stabilize Kaggle baseline frame with mamba_ssm stub and hard-fail |

## 2. Bundle

| Item | Value |
|------|-------|
| path | artifacts/kaggle/Nemotron_kaggle_bundle.zip |
| size | 47,475 bytes |
| entries | 18 |
| validation | ALL CHECKS PASS (13/13) |

### Validation Detail

| Check | Result |
|-------|--------|
| `NotImplementedError` absent from stub | OK |
| `rmsnorm_fn` defined in stub | OK |
| `rms_norm_fn = rmsnorm_fn` alias | OK |
| `RMSNorm` / `LayerNorm` classes | OK |
| `_GENERATION_NUM_BEAMS = 1` const (L98) | OK |
| `num_beams=_GENERATION_NUM_BEAMS` explicit (L588) | OK |
| hard fail guard `!= 1` → RuntimeError (L569) | OK |
| gen_policy startup log (L744) | OK |
| `beam_search_ban_policy` reference | OK |
| `num_beams=2` absent | OK |
| `is_mamba_2_ssm_available` patch | OK |
| `sys.path.insert(0, REPO_ROOT)` | OK |

## 3. Kaggle Execution

| Item | Value |
|------|-------|
| status | PENDING_KAGGLE_EXECUTION |
| execution date | _______________ |
| eval set | _______________ |
| LB score | _______________ |
| baseline expected | 0.85 |
| submission.zip generated | _______________ |
| runtime duration | _______________ |
| runtime errors | _______________ |

## 4. Guard Verification (Kaggle log — fill after run)

| Check | Expected | Actual |
|-------|---------|--------|
| `[baseline][gen_policy] num_beams=1` in startup log | YES | ___ |
| `num_beams=2` absent from all logs | YES | ___ |
| `HARD FAIL: _GENERATION_NUM_BEAMS` absent (guard not triggered) | YES | ___ |
| `NotImplementedError` absent | YES | ___ |
| `NemotronHHybridDynamicCache ... None was provided` absent | YES | ___ |
| `run_complete.flag` written | YES | ___ |

## 5. Decision

| Item | Value |
|------|-------|
| reproduced baseline 0.85 | UNKNOWN (pending) |
| decision | **HOLD** |
| reason | Kaggle execution not yet performed. All local checks pass. Awaiting LB score confirmation. |

## 6. Files Created / Updated

| File | Status |
|------|--------|
| `reports/runtime_gold_refreeze_kaggle_result.md` | Created (this file) |
| `logs/runtime_gold_refreeze_kaggle_log.txt` | Created |
| `reports/frozen_baseline_decision.md` | Created |

## 7. Next Action

**Current decision: HOLD**

To move to `ADOPT_AS_NEW_FROZEN_BASELINE`, the following evidence is required:

- [ ] Kaggle execution completes without crash
- [ ] `submission.zip` generated
- [ ] LB score ≥ 0.85
- [ ] `[baseline][gen_policy] num_beams=1` confirmed in Kaggle log
- [ ] `num_beams=2` absent from Kaggle log
- [ ] `NotImplementedError` absent from Kaggle log
- [ ] No silent fallback detected

**On ADOPT**: Create git tag `baseline-refreeze-v1` at commit `3b17c3d`, update `reports/frozen_baseline_decision.md` with final decision.

**On REJECT**: Identify whether failure is caused by (a) `mamba_ssm` stub PyTorch impl difference, (b) explicit `num_beams=1` param side-effect, or (c) unrelated runtime environment issue. Revert to `b1d54ce` and re-apply changes incrementally.
