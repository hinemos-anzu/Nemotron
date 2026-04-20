# TICKET_S1_6A_kaggle_runtime_execution_enablement_for_baseline_path

## Ticket type
Implementation design ticket

## Strategy ID
S1.6A

## Strategy name
Kaggle runtime execution enablement for baseline path

## Objective
Enable the current baseline path to run in the Kaggle source-of-truth environment on the frozen evaluation assets, so the blocked Kaggle-suffixed measured outputs can be replaced with real measured outputs.
This ticket is environment-enablement only.
It does not authorize strategy changes, model changes, prompt changes, or submission-asset redesign.

## Single main variable
Execution environment enablement for the existing baseline path in the Kaggle source-of-truth runtime.

## In scope
- establish a runnable Kaggle source-of-truth execution path for `kaggle/original-nemotron-asymmetric-svd-26041602.py`
- resolve environment-only blockers needed to execute that unchanged baseline path
- run the unchanged baseline path once on frozen Quick Gate v1
- run the unchanged baseline path once on frozen Diagnostic v1
- run the unchanged baseline path once on frozen Promotion v1
- overwrite blocked Kaggle-suffixed measured result files with real measured outputs
- update the Kaggle measured reference report and experiment log from BLOCKED to measured or partially measured status

## Out of scope
- model-weight changes
- adapter changes
- prompt / CoT changes
- training-data changes
- replay / reweighting
- conversion logic redesign
- SVD redesign
- key mapping redesign
- merge logic redesign
- strategy comparison against alternatives

## Target layer
- Kaggle runtime environment only
- baseline execution wrappers only as needed to run the unchanged baseline path
- measurement reporting for frozen evaluation assets

## Allowed changes
- Kaggle notebook/job configuration
- environment bootstrap code strictly needed to run the existing baseline path
- compatibility wrappers needed to satisfy Python/runtime differences without changing strategy behavior
- measurement scripts and reporting files under `reports/eval/*`, `logs/experiments/*`, and `data/eval/*`

## Forbidden changes
- frozen membership of `quick_gate_v1`, `diagnostic_v1`, `promotion_v1`
- semantics of `category_manifest_v1.csv`
- baseline model weights or adapters
- baseline reasoning logic or strategy
- training procedure
- submission asset flow redesign
- bundling environment enablement with strategy modifications
- changing the baseline path to a different script

## Protected assets touched
No strategy-level protected asset changes allowed.
Environment-only wrappers are allowed if they preserve baseline behavior and are explicitly documented.

## Required inputs
- `tickets/TICKET_S1_5_baseline_measured_reference_v1.md`
- blocked evidence from S1.6 / Kaggle-suffixed blocked outputs
- frozen `data/eval/quick_gate_v1.*`
- frozen `data/eval/diagnostic_v1.*`
- frozen `data/eval/promotion_v1.*`
- frozen `data/eval/category_manifest_v1.csv`
- current baseline path: `kaggle/original-nemotron-asymmetric-svd-26041602.py`
- current Kaggle-suffixed measured files and logs

## Required outputs
1. `reports/eval/baseline_reference_v1_measured_kaggle.md` (updated from BLOCKED to measured/partially measured)
2. `data/eval/baseline_measured_results_quick_gate_v1_kaggle.csv`
3. `data/eval/baseline_measured_results_diagnostic_v1_kaggle.csv`
4. `data/eval/baseline_measured_results_promotion_v1_kaggle.csv`
5. `data/eval/baseline_measured_category_summary_v1_kaggle.csv`
6. `logs/experiments/dayX_s1_6A_kaggle_runtime_enablement.md`
7. `reports/eval/kaggle_runtime_enablement_provenance_v1.md`

## Deliverable details
### Provenance report must include
- exact Kaggle runtime used
- notebook/job identifier or equivalent execution reference
- Python version
- key dependency versions
- filesystem assumptions satisfied
- exact environment blockers from S1.6 and whether each was resolved
- exact compatibility layer applied, if any
- explicit statement that the baseline path itself was not swapped out

### Required measured-result fields
- sample_id
- split_name
- category
- baseline_prediction_status
- baseline_correctness if available
- format_failure_flag
- extraction_failure_flag
- runtime_status
- notes

### Required category summary fields
- split_name
- category
- sample_count
- measured_pass_count
- measured_fail_count
- measured_error_count
- format_failure_count
- extraction_failure_count
- pass_rate_or_accuracy

## Expected effect
- convert Kaggle BLOCKED status into real measured baseline outputs on frozen assets
- establish a valid Kaggle source-of-truth measurement baseline for later strategy promotion
- separate environment failure from model or strategy failure conclusively

## Acceptable collateral damage
- environment-only code or notebook setup overhead
- no changes to baseline strategy behavior
- no changes to protected submission assets

## Fast-fail criteria
Fail this ticket if any of the following occur:
- frozen evaluation memberships are changed
- the baseline path is replaced by a different strategy script
- environment fixes silently alter baseline reasoning behavior
- measured outputs remain blocked with no precise blocker update
- Kaggle provenance is not recorded
- the resulting outputs cannot be compared back to the blocked S1.6 files

## Evaluation path
- confirm frozen S1 assets are unchanged
- confirm the same baseline path is targeted
- reproduce and document current blockers
- resolve environment-only blockers in Kaggle source-of-truth runtime
- execute unchanged baseline path once on each frozen split
- write per-sample Kaggle measured outputs
- write category summary
- update measured Kaggle report and log
- Planner reviews whether S1.5/S1.6 can now be treated as complete on Kaggle source-of-truth

## Planner acceptance criteria
Planner accepts only if:
- Kaggle source-of-truth execution reference is documented
- real measured outputs exist for all three frozen splits, or any remaining gap is precisely classified
- baseline path was not swapped
- no strategy-level scope creep occurred
- category-wise measured summary exists
- Quick Gate practical filter can now be assessed with real Kaggle outputs, or any remaining limitation is explicitly bounded

## Consumer strategy for outputs
- S1.5 is considered complete only after these measured Kaggle outputs exist or are conclusively bounded
- S2 consumes the measured Kaggle baseline as the external control reference when available
- S5 consumes the measured Kaggle baseline for later pre/post-conversion external comparison
- Kaggle execution role uses this as the definitive environment-enablement boundary ticket

## Owner recommendation
Primary owner: Kaggle execution role or Generator with Kaggle authority
Approver: Planner
Generator may assist with environment bootstrap and provenance, but must not change strategy scope
