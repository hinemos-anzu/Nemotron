# TICKET_S1_5_baseline_measured_reference_v1

## Ticket type
Implementation design ticket

## Strategy ID
S1.5

## Strategy name
Baseline measured reference promotion on frozen evaluation assets

## Objective
Promote S1 evaluation foundation v1 from a design-fixed reference into a measured baseline reference by running the current baseline model on the frozen Quick Gate / Diagnostic / Promotion assets exactly once under a controlled evaluation-only scope.
This ticket exists to close the gap between:
- "evaluation assets are defined"
and
- "baseline behavior is measured and frozen on those assets"

## Single main variable
Evaluation run execution only on frozen evaluation foundation v1.

## In scope
- run the current baseline evaluation path on frozen Quick Gate v1
- run the current baseline evaluation path on frozen Diagnostic v1
- run the current baseline evaluation path on frozen Promotion v1
- record category-wise pass/fail/error behavior
- update baseline reference report from design-fixed to measured reference
- produce one complete experiment log for the measured baseline promotion step
- verify whether Quick Gate is practically usable as an early filter with real outputs

## Out of scope
- model-weight changes
- prompt / CoT changes
- training-data changes
- replay / reweighting
- conversion logic changes
- submission asset redesign
- Kaggle strategy comparison

## Target layer
- evaluation execution only
- baseline measurement reporting
- experiment log completion

## Allowed changes
- evaluation execution wrappers or runner scripts if needed only to run the frozen sets
- reporting files under `reports/eval/*`
- experiment logs under `logs/experiments/*`
- measurement-only companion files under `data/eval/*`

## Forbidden changes
- frozen membership of quick_gate_v1 / diagnostic_v1 / promotion_v1
- category_manifest_v1 row semantics except for adding measured columns in companion files
- model weights or adapters
- training sampler
- prompt / CoT templates
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
- frozen `data/eval/quick_gate_v1.*`
- frozen `data/eval/diagnostic_v1.*`
- frozen `data/eval/promotion_v1.*`
- frozen `data/eval/category_manifest_v1.csv`
- `reports/eval/baseline_reference_v1.md`
- `docs/specs/experiment_log_template.md`
- current baseline model / adapter evaluation path

## Required outputs
1. `reports/eval/baseline_reference_v1_measured.md`
2. `data/eval/baseline_measured_results_quick_gate_v1.csv`
3. `data/eval/baseline_measured_results_diagnostic_v1.csv`
4. `data/eval/baseline_measured_results_promotion_v1.csv`
5. `data/eval/baseline_measured_category_summary_v1.csv`
6. `logs/experiments/dayX_s1_5_baseline_measured_reference.md`

## Deliverable details
### Per-sample measured result files must include
- sample_id
- split_name
- category
- baseline_prediction_status
- baseline_correctness if answer truth is available in the evaluation path
- format_failure_flag
- extraction_failure_flag
- runtime_status
- notes field for abnormal cases

### Category summary file must include
- split_name
- category
- sample_count
- measured_pass_count
- measured_fail_count
- measured_error_count
- format_failure_count
- extraction_failure_count
- pass_rate_or_accuracy field as appropriate to available labels

### Measured baseline reference report must include
- what baseline path was executed
- what was frozen and not changed
- per-split summary
- category-wise measured table
- Quick Gate practical filter assessment
- known ambiguities or blocked measurements

## Expected effect
- convert S1 from design-only evaluation foundation into measured baseline reference
- provide the first real category × split baseline table
- enable later S2 / S3 / S4 / S5 comparisons against measured reality rather than design assumptions

## Acceptable collateral damage
- evaluation runtime cost only
- no changes to protected submission assets

## Fast-fail criteria
Fail this ticket if any of the following occur:
- frozen evaluation-set membership is changed
- the baseline path used for measurement is not documented clearly
- measured outputs are missing for any of the three frozen splits without explanation
- category-wise measured summary is not produced
- the report cannot tell whether Quick Gate is usable as an early practical filter

## Evaluation path
- confirm frozen S1 assets are unchanged
- run baseline once on Quick Gate v1
- run baseline once on Diagnostic v1
- run baseline once on Promotion v1
- produce per-sample and per-category measured outputs
- update the measured reference report
- Planner reviews whether S1 can now be treated as complete

## Planner acceptance criteria
Planner accepts only if:
- measured outputs exist for all three frozen splits
- category-wise measured table exists
- Quick Gate usability is assessed with real outputs
- no protected asset or model-change scope creep occurred
- experiment log is complete and references the exact baseline path used

## Consumer strategy for outputs
- S2 consumes measured baseline outputs as the analysis baseline
- S3 consumes measured easy-category failure and formatting behavior
- S5 consumes measured baseline split/category tables as the control reference
- Planner uses this ticket to declare S1 complete or still incomplete

## Owner recommendation
Primary owner: Generator
Approver: Planner
Kaggle execution role: not required unless Planner requests environment-only confirmation
