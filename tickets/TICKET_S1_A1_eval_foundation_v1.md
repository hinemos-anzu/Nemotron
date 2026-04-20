# TICKET_S1_A1_eval_foundation_v1

## Ticket type
Implementation design ticket

## Strategy ID
S1

## Strategy name
Held-out validation / evaluation foundation

## Objective
Build the first stable evaluation OS for the project so weak ideas can be rejected quickly and promoted ideas can be justified consistently.
This ticket covers only the creation and freezing of the first evaluation assets and baseline reference outputs.

## Single main variable
Evaluation foundation version v1.

## In scope
- Define and create Quick Gate set v1
- Define and create Diagnostic set v1
- Define and create Promotion set v1
- Define category manifest v1
- Generate baseline reference report on the frozen sets
- Produce one completed experiment log using the shared template for the baseline

## Out of scope
- model-weight changes
- prompt/trace redesign
- conversion logic changes
- reweighting
- Kaggle run for strategy comparison

## Target layer
- evaluation datasets
- dataset manifests
- evaluation scripts/config wrappers
- baseline reference reporting

## Allowed changes
- `data/eval/*`
- `reports/eval/*`
- `reports/day1/*`
- `logs/experiments/*`
- evaluation-only scripts or config wrappers under repo evaluation paths

## Forbidden changes
- training data contents
- adapter conversion flow
- offline asymmetric SVD surgery
- key rename / merge logic
- expert unfuse logic
- gate_x_to_in_proj_merge logic
- submission.zip generation path
- model weights or adapters

## Protected assets touched
No

## Required inputs
- `docs/research/spec_research_result_v1.md`
- `docs/specs/design_spec_from_research_v1.md`
- `docs/specs/a1_evaluation_set_design.md`
- `docs/specs/category_evaluation_criteria.md`
- `docs/specs/experiment_log_template.md`
- current baseline evaluation artifacts if already available
- known brittle / historical-failure samples if available

## Required outputs
1. `data/eval/quick_gate_v1.*`
2. `data/eval/diagnostic_v1.*`
3. `data/eval/promotion_v1.*`
4. `data/eval/category_manifest_v1.csv`
5. `reports/eval/baseline_reference_v1.md`
6. `logs/experiments/day1_s1_baseline_eval.md`

## Deliverable details
### Quick Gate v1
- target size: 50 to 100 samples
- must include easy categories, one representative symbolic slice, one representative bit slice, and known brittle cases
- purpose: reject obvious regressions fast

### Diagnostic v1
- target size: 100 to 200 samples
- must include brittle cases, low-confidence suspects if known, and conversion-sensitive suspects if known
- purpose: explain why a run improved or regressed

### Promotion v1
- target size: 300 to 500 samples
- stable category distribution
- purpose: decide whether Kaggle budget should be spent

### Category manifest v1
Each row must contain at least:
- sample_id
- split_name
- category
- difficulty_bucket
- failure_mode_primary
- failure_mode_secondary
- conversion_sensitive_flag
- inclusion_reason

## Expected effect
- faster early rejection of weak strategies
- reproducible baseline comparisons
- shared category language for S2, S3, S4, S5, S6, and S7

## Acceptable collateral damage
- none on model or submission assets
- minor iteration on sample composition is acceptable before freeze

## Fast-fail criteria
Fail this ticket if any of the following occur:
- baseline rerun on the same frozen sets gives inconsistent judgment without explanation
- category manifest is too sparse to detect easy or bit regressions
- Quick Gate cannot be run quickly enough to serve as an early filter
- baseline report does not include category-wise results

## Evaluation path
- create candidate sets
- run baseline on all three sets
- verify repeatability on at least one rerun or deterministic replay
- freeze v1 only after Planner sign-off

## Planner acceptance criteria
Planner accepts only if:
- all three sets exist and are versioned
- category manifest exists and is usable
- baseline reference report exists
- experiment log exists and is complete
- weak ideas can be rejected on Quick Gate without touching Kaggle

## Consumer strategy for outputs
- S2 consumes brittle samples, manifests, and evaluation structure
- S3 consumes easy-category grouping and output-failure labels
- S5 consumes conversion-sensitive subset definitions
- Kaggle execution role consumes promotion rules indirectly through baseline reference

## Owner recommendation
Primary owner: Generator
Approver: Planner
Kaggle execution role: not required unless Planner requests environment-only confirmation
