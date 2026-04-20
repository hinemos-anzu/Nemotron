# TICKET_S1_6_kaggle_source_of_truth_measured_reference_v1

## Ticket type
Implementation design ticket

## Strategy ID
S1.6

## Strategy name
Kaggle source-of-truth baseline measured reference on frozen evaluation assets

## Objective
Complete the unfinished goal of S1.5 by running the current baseline path in the Kaggle source-of-truth environment on the frozen S1 evaluation assets and producing real measured outputs instead of blocked-only outputs.

## Single main variable
Execution environment only: local blocked path -> Kaggle source-of-truth path.

## In scope
- execute the current baseline path in Kaggle source-of-truth environment
- reuse the same measured CSV schema from S1.5
- run on frozen Quick Gate / Diagnostic / Promotion assets
- produce real pass/fail/error measured outputs
- finalize Quick Gate practical filter assessment with real outputs
- update measured reference report to source-of-truth status

## Out of scope
- model-weight changes
- prompt / CoT changes
- training-data changes
- conversion fixes
- replay / reweighting
- evaluation-set membership changes

## Target layer
- Kaggle execution only
- measured baseline reporting

## Allowed changes
- Kaggle execution wrapper adjustments needed only to run the frozen assets in the Kaggle environment
- report files under `reports/eval/*`
- experiment logs under `logs/experiments/*`
- measured companion files under `data/eval/*`

## Forbidden changes
- frozen Quick Gate / Diagnostic / Promotion membership
- category manifest semantics
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
- S1 frozen evaluation assets
- S1.5 measured CSV schema and wrapper logic
- current Kaggle-runnable baseline path
- current baseline model / adapter assets available in Kaggle

## Required outputs
1. `reports/eval/baseline_reference_v1_measured_kaggle.md`
2. `data/eval/baseline_measured_results_quick_gate_v1_kaggle.csv`
3. `data/eval/baseline_measured_results_diagnostic_v1_kaggle.csv`
4. `data/eval/baseline_measured_results_promotion_v1_kaggle.csv`
5. `data/eval/baseline_measured_category_summary_v1_kaggle.csv`
6. `logs/experiments/dayX_s1_6_kaggle_measured_reference.md`

## Expected effect
- turn S1 from frozen-design foundation into a real measured baseline foundation
- provide source-of-truth category × split baseline outputs
- unblock S2 and S5 from using real measured control outputs

## Fast-fail criteria
Fail this ticket if:
- Kaggle run does not actually use the frozen S1 assets
- measured outputs are not produced for all three splits without explanation
- Quick Gate practical filter assessment is still not possible after Kaggle execution
- scope creep changes model or submission assets

## Planner acceptance criteria
Planner accepts only if:
- all three frozen splits have Kaggle source-of-truth measured outputs
- category-wise measured summary exists
- Quick Gate practical usefulness is assessed with real outputs
- S1 can be declared complete after this ticket

## Consumer strategy for outputs
- S2 consumes Kaggle-measured baseline outputs as the analysis control
- S5 consumes Kaggle-measured baseline outputs as the pre/post comparison control
- Planner uses this ticket to close S1 completely

## Owner recommendation
Primary owner: Kaggle execution role with Generator support
Approver: Planner
