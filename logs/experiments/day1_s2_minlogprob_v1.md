# S2: Min Logprob Answer Selection v1 — Experiment Log

**Date:** 2026-04-21  
**Strategy ID:** S2  
**Branch:** `claude/exp-s2-minlogprob-v1`  
**Base branch:** `claude/nemotron-kaggle-baseline-BE6bY`  
**Base commit:** `b1d54ce`  
**Single main variable:** min logprob answer selection (beam search, N=4)

---

## Hypothesis

Generating multiple candidate answers (N=4 beams) and selecting the one with the
highest beam score (= mean token log-prob = min negative logprob) should surface
higher-quality answers than greedy decoding, particularly for:
- numeric/format-sensitive categories (numeral, unit_conversion)
- categories where the model hedges between equivalent forms

## What Changed

| Component | Baseline | S2 |
|---|---|---|
| Decoding | greedy (`num_beams=1, do_sample=False`) | beam search (`num_beams=4, num_return_sequences=4`) |
| Selection | first (only) output | highest `sequences_scores` beam |
| Columns added | — | `s2_n_candidates`, `s2_selected_idx`, `s2_selected_mean_logprob`, `s2_selection_method` |
| Stage log events | — | `s2_start`, `s2_end` |

## What Did NOT Change

- Runtime bootstrap (REPO_ROOT sys.path, is_mamba_2_ssm_available patch)
- local mamba_ssm stub
- model / adapter / tokenizer
- prompt template
- answer extraction (`_extract_answer`)
- answer matching (`_answers_match`)
- hard-fail policy
- ANSWER_KEY_ONLY: still disabled
- debug_harness.py
- evaluation set membership

## Execution Plan

1. quick_gate_v1 (75 samples) → gate check
2. diagnostic_v1 (150 samples) → only if quick passes
3. promotion_v1 (400 samples) → only if diagnostic passes

## Success Criteria

Technical:
- `model_load_end = mode=INFERENCE` in stage_log
- `run_finish = SUCCESS`
- `error_report.md = NO_ERROR`
- `s2_start` and `s2_end` events in stage_log
- logprob values observed in CSV

Evaluation:
- quick_gate: no regression vs baseline; ideally improvement
- diagnostic: no category collapse
- promotion: improvement or same vs baseline

## Fail Criteria

- runtime gold broken
- ANSWER_KEY_ONLY restored
- quick_gate incomplete
- no logprob values observed
- multiple variables changed
- OOM during beam search (→ reduce N_CANDIDATES or report)

---

## Results

*(to be filled after Kaggle execution)*

### Quick Gate v1 (75 samples)

- Status: PENDING
- model_load_end: —
- run_finish: —
- error_report: —
- s2_start/s2_end observed: —
- correctness rate: —
- format_failure rate: —
- extraction_failure rate: —
- selected_non_beam0 rate: —

### Diagnostic v1 (150 samples)

- Status: PENDING

### Promotion v1 (400 samples)

- Status: PENDING

---

## Logprob-Selection Observations

*(to be filled after Kaggle execution)*

- n_candidates used: 4
- mean selected_idx: —
- selected_non_beam0 fraction: —
- mean selected_mean_logprob: —
- range of selected_mean_logprob: —

---

## Adoption Recommendation

*(to be filled after results)*
