# TICKET_S5_misalignment_harness_v1

## Ticket type
Implementation design ticket

## Strategy ID
S5

## Strategy name
Training-serving misalignment quantification and repair harness

## Objective
Build the first isolated comparison harness for measuring training-serving misalignment.
This ticket is only for measurement and isolation, not for fixing multiple conversion variables at once.
The immediate goal is to compare pre-conversion and post-conversion behavior on the same sample IDs and produce a residual-gap map.

## Single main variable
Misalignment comparison harness version v1.

## In scope
- define exact pre-conversion vs post-conversion comparison protocol
- run the comparison on the same frozen sample IDs
- generate category-wise, brittle-sample-wise, and format-failure deltas
- produce a variable-isolation plan for later repair experiments
- write one complete baseline experiment log using the shared template

## Out of scope
- actual conversion repair patches beyond what is strictly required to establish a valid comparison harness
- replay / reweighting
- prompt / CoT redesign
- multi-variable conversion experiments
- Kaggle run for strategy comparison unless Planner explicitly requests an environment-only check

## Target layer
- pre/post-conversion evaluation harness
- comparison scripts
- conversion-side reporting only

## Allowed changes
- evaluation harness scripts
- reporting scripts
- comparison manifests
- `reports/eval/*`
- `logs/experiments/*`
- non-invasive wrappers needed to run pre/post comparison reproducibly

## Forbidden changes
- training-data composition
- prompt templates / CoT templates
- replay ratios
- more than one conversion variable in the same follow-up fix experiment
- submission.zip generation path as part of this harness ticket
- bundling SVD precision changes, rank changes, merge-order changes, key-map changes, or lm_head changes together in one measurement step

## Protected assets touched
Potentially yes, but only for measurement and manifesting, not for bundled redesign.

## Required inputs
- `tickets/TICKET_S1_A1_eval_foundation_v1.md`
- frozen Quick Gate / Diagnostic / Promotion assets from S1
- `tickets/TICKET_S2_min_logprob_foundation_v1.md` outputs if available
- `docs/research/spec_research_result_v1.md`
- `docs/specs/design_spec_from_research_v1.md`
- current baseline adapter assets before and after conversion
- current protected submission asset manifest

## Required outputs
1. `reports/eval/misalignment_protocol_v1.md`
2. `reports/eval/pre_post_conversion_delta_v1.md`
3. `data/eval/pre_post_conversion_delta_v1.csv`
4. `data/eval/conversion_sensitive_brittle_cases_v1.csv`
5. `logs/experiments/dayX_s5_misalignment_baseline.md`
6. `reports/eval/conversion_followup_variables_v1.md`

## Deliverable details
### Required comparison axes
- total accuracy delta
- category-wise delta
- brittle-sample delta
- format failure delta
- min-logprob delta when available

### Required row fields for the CSV
- sample_id
- split_name
- category
- baseline_pre_conversion_result
- baseline_post_conversion_result
- delta_label
- format_failure_pre
- format_failure_post
- min_logprob_pre if available
- min_logprob_post if available
- brittle_subset_flag

### Required follow-up variable list
The report must explicitly separate future one-variable repair candidates such as:
- SVD precision
- rank allocation policy
- merge order
- key mapping
- lm_head handling
- architecture-specific surgery switch

## Expected effect
- establish whether a real pre/post-conversion gap exists
- locate which categories and brittle cases are most conversion-sensitive
- tell Planner whether S5 repair should continue and whether S8 post-finetune is even justified later

## Acceptable collateral damage
- minor extra measurement runtime is acceptable
- no degradation of the protected submission asset flow is acceptable

## Fast-fail criteria
Fail this ticket if any of the following occur:
- pre/post results are not measured on the same sample IDs
- category-wise comparison is missing
- protected asset reproducibility is not recorded
- the output does not isolate future one-variable repair candidates
- the harness itself silently changes conversion behavior without declaring it

## Evaluation path
- run pre-conversion evaluation on frozen sets
- run post-conversion evaluation on the exact same frozen sets
- produce comparison tables and brittle-case subset reports
- Planner decides whether a real misalignment gap exists and which single variable should be tested first in the next S5 repair ticket

## Planner acceptance criteria
Planner accepts only if:
- a reproducible pre/post comparison protocol exists
- deltas are visible by category and brittle subset
- conversion-sensitive cases are explicitly listed
- a follow-up one-variable repair queue is written
- no accidental multi-variable change has been introduced

## Consumer strategy for outputs
- next S5 repair ticket consumes the isolated follow-up variable queue
- S8 consumes the residual-gap decision only if the gap remains after later isolated repair work
- S2 outputs may be joined to this ticket later for min-logprob delta analysis, but that is not required to complete this ticket

## Owner recommendation
Primary owner: Generator
Approver: Planner
Kaggle execution role: optional only for environment or submission-side confirmation requested by Planner
