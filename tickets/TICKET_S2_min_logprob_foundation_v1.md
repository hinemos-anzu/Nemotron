# TICKET_S2_min_logprob_foundation_v1

## Ticket type
Implementation design ticket

## Strategy ID
S2

## Strategy name
Min-logprob analysis foundation

## Objective
Implement the first reusable min-logprob analysis pipeline so the project can identify brittle tokens and brittle local trace segments, not just weak samples in the aggregate.
This ticket covers logging, extraction, aggregation, and reporting only.
It does not include any model-weight change or replay/reweighting yet.

## Single main variable
Min-logprob analysis foundation version v1.

## In scope
- add or wire logging needed to capture token-level confidence for evaluated traces
- extract per-sample minimum logprob, responsible token, and local context window
- map results to category and failure-mode labels from S1 manifests
- produce ranked brittle sample outputs and token/context exemplars
- generate one complete baseline experiment log using the shared template

## Out of scope
- replay / reweighting
- prompt/CoT redesign
- conversion logic changes
- training-data composition changes
- Kaggle run for strategy comparison

## Target layer
- inference/evaluation logging
- token-level trace post-processing
- metric aggregation
- reports and ranked brittle manifests

## Allowed changes
- logging hooks used only for evaluation/analysis
- evaluation scripts and parsers
- reporting notebooks/scripts
- `reports/eval/*`
- `logs/experiments/*`
- `data/eval/*` only for adding analysis-derived annotations or companion files

## Forbidden changes
- model weights or adapters
- training sampler / replay ratios
- prompt templates / CoT templates
- adapter conversion flow
- offline asymmetric SVD surgery
- key rename / merge logic
- expert unfuse logic
- gate_x_to_in_proj_merge logic
- submission.zip generation path

## Protected assets touched
No

## Required inputs
- `tickets/TICKET_S1_A1_eval_foundation_v1.md`
- frozen Quick Gate / Diagnostic / Promotion assets from S1
- `docs/research/spec_research_result_v1.md`
- `docs/specs/design_spec_from_research_v1.md`
- baseline evaluation outputs
- access to token/logprob outputs from the current evaluation path

## Required outputs
1. `reports/eval/min_logprob_spec_v1.md`
2. `reports/eval/min_logprob_baseline_v1.md`
3. `data/eval/brittle_samples_minlogprob_v1.csv`
4. `data/eval/token_fragility_examples_v1.jsonl`
5. `logs/experiments/dayX_s2_min_logprob_baseline.md`

## Deliverable details
### Required fields per analyzed sample
- sample_id
- split_name
- category
- correctness
- min_logprob_value
- min_logprob_token
- local_context_left
- local_context_right
- failure_mode_primary
- failure_mode_secondary if available
- pre_or_post_conversion_label when available

### Required ranked outputs
- lowest-min-logprob samples overall
- lowest-min-logprob samples by category
- lowest-min-logprob samples among correct predictions
- lowest-min-logprob samples among incorrect predictions
- token/context exemplars grouped by dominant failure mode

### Required report sections
- logging path used
- overhead note
- category-wise brittle concentration
- top recurring fragile fragments
- candidate consumer map for S4 / S5 / S6 / S7

## Expected effect
- convert vague regressions into concrete token-level and local-context coding targets
- identify the brittle subset for replay/reweighting
- identify tokenization-sensitive phrases for deterministic CoT work
- identify conversion-sensitive brittle cases for misalignment analysis

## Acceptable collateral damage
- small evaluation-only runtime overhead is acceptable
- no change to training or submission behavior is acceptable

## Fast-fail criteria
Fail this ticket if any of the following occur:
- only averages or histograms are produced, with no sample/token/context exemplars
- outputs cannot be mapped back to category or sample_id
- logging overhead is too high for repeated use in the evaluation loop
- the final report does not state how S4, S5, S6, and S7 will consume the outputs

## Evaluation path
- run baseline evaluation path with min-logprob instrumentation
- generate brittle rankings on at least Quick Gate and Diagnostic sets
- verify that repeated analysis of the same run is stable
- Planner reviews whether outputs are implementation-useful, not merely interesting

## Planner acceptance criteria
Planner accepts only if:
- token/context exemplars exist
- brittle subset ranking exists
- category-wise brittle concentration is visible
- one explicit handoff is written for S4, one for S5, one for S6, and one for S7

## Consumer strategy for outputs
- S4 consumes `brittle_samples_minlogprob_v1.csv`
- S5 consumes conversion-sensitive brittle rows and pre/post labels
- S6 consumes recurring fragile phrases and local context exemplars
- S7 consumes bit-category brittle rows and bit-specific fragile fragments

## Owner recommendation
Primary owner: Generator
Approver: Planner
Kaggle execution role: not required
