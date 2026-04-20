# Design Specification from Research v1.0

## 1. Objective
This document converts the consolidated research result into a coding-design specification that can be shared across Planner, Generator, and Kaggle execution roles.
Its purpose is to move the project from research into implementation planning without gray areas.

## 2. Scope
This specification covers mainline sprint work only.
Included:
- S1 evaluation foundation
- S2 min-logprob analysis
- S3 easy-task miss recovery
- S4 low-min-logprob reweighting
- S5 training-serving misalignment quantification and repair
- S6 deterministic CoT redesign
- S7 bit manipulation specialized strengthening
- S8 post-finetune after conversion (conditional)

Excluded from mainline sprint:
- S9 token-direct generation pipeline
- S10 synthetic problem expansion as a mainline success criterion
- S11 cryptarithm breakthrough research

## 3. Protected baseline assets
The following are protected and cannot be changed casually or in bundles:
- adapter conversion flow
- offline asymmetric SVD surgery path
- key rename / merge handling
- expert unfuse logic
- gate_x_to_in_proj_merge behavior
- submission.zip generation path
- validated submission file structure

Any experiment touching protected assets must:
- version the artifact bundle
- keep a before/after manifest
- keep reproducible sample IDs for evaluation

## 4. Team responsibilities
### Planner
- owns priority and gate thresholds
- owns promotion decisions
- freezes experiment scope
- rejects non-implementation-ready tickets

### Generator
- owns coding-design drafts and implementation prep
- owns evaluation artifacts and logs
- owns sample manifests and category labeling updates

### Kaggle execution role
- runs only promoted candidates or explicit environment checks
- returns Kaggle evidence into the shared experiment log
- does not invent new strategy scope

## 5. Mandatory engineering rules
- one experiment = one main variable
- every experiment must declare allowed changes and forbidden changes
- every experiment must run through Quick Gate first
- every experiment must produce a category-wise before/after table
- every experiment must state fast-fail criteria before coding starts
- every experiment must state what next strategy will consume from its outputs

## 6. Evaluation OS contract
### Quick Gate
Purpose: reject obvious regressions quickly.
Required outputs:
- total delta vs baseline
- easy delta vs baseline
- format failure delta
- main target-category delta
Decision: PASS_TO_DIAGNOSTIC or FAIL_QUICK_GATE.

### Diagnostic
Purpose: explain the observed delta.
Required outputs:
- category table
- dominant failure mode changes
- brittle-case changes
- pre/post-conversion deltas if applicable
Decision: PASS_TO_PROMOTION / FAIL_DIAGNOSTIC / HOLD_INCONCLUSIVE.

### Promotion
Purpose: decide whether Kaggle budget should be spent.
Required outputs:
- promotion-set delta vs baseline
- category-risk review
- Planner decision note
Decision: PROMOTE_TO_KAGGLE / HOLD_INCONCLUSIVE / REJECT.

## 7. Strategy implementation sequence
### Phase A: evaluation and visibility
#### 7.1 S1 evaluation foundation
Deliverables:
- quick_gate_v1
- diagnostic_v1
- promotion_v1
- category manifest
- baseline reference report
Done when:
- repeated baseline judgments are consistent
- weak ideas can be rejected early

#### 7.2 S2 min-logprob analysis
Deliverables:
- brittle sample ranking
- token/context exemplars
- failure-mode mapping
- conversion-sensitive brittle subset
Done when:
- token-level weak points are actionable
- S4/S5/S6/S7 can consume outputs directly

### Phase B: high-leverage score protection and recovery
#### 7.3 S3 easy-task miss recovery
Deliverables:
- easy output contract
- easy failure catalog
- easy before/after report
Done when:
- easy-task regression risk is near zero
- format/extraction failures drop or stay stable

#### 7.4 S5 training-serving misalignment repair
Deliverables:
- pre/post-conversion comparison table
- conversion-variable ablation report
- residual-gap decision
Done when:
- conversion losses are either reduced or clearly bounded

### Phase C: targeted model improvement
#### 7.5 S4 low-min-logprob reweighting
Deliverables:
- replay/reweight manifest
- brittle-subset improvement report
- overfit-risk note
Done when:
- brittle subset improves without unacceptable collateral damage

#### 7.6 S6 deterministic CoT redesign
Deliverables:
- category-specific stable phrasing list
- forbidden unstable phrasing list
- target-category template diff
Done when:
- greedy stability improves on target categories
- brittle segments or formatting regressions do not worsen

#### 7.7 S7 bit manipulation specialized strengthening
Deliverables:
- bit notation contract
- bit failure taxonomy
- bit rare-pattern manifest
- bit before/after report
Done when:
- bit category shows meaningful repeatable gain

### Phase D: conditional escalation
#### 7.8 S8 post-finetune after conversion
Precondition:
- S5 has proven a residual post-conversion gap remains.
Deliverables:
- residual-gap recovery report
- converted-adapter comparison
Done when:
- residual post-conversion losses shrink on Promotion evaluation

## 8. Ticket template required for all coding-design work
Every ticket must include:
- Strategy ID
- Objective
- Target layer
- Allowed changes
- Forbidden changes
- Protected assets touched: yes/no
- Required inputs
- Required outputs
- Expected effect
- Acceptable collateral damage
- Fast-fail criteria
- Evaluation path
- Consumer strategy for the outputs

If any field is missing, Planner must reject the ticket as not implementation-ready.

## 9. Day-by-day handoff recommendation
### Day 1
- finalize S1 assets
- baseline reference run
- category manifest freeze

### Day 2
- implement S2 logging and token/context extraction
- draft S5 comparison harness in parallel

### Day 3
- run S3 easy-task protection
- run first S5 isolated conversion ablation

### Day 4
- run S4 brittle-subset reweighting

### Day 5
- choose between S6 and S7 as the main next-step focus based on S2+S3+S4+S5 evidence

## 10. Promotion rule for Kaggle execution
A candidate may go to Kaggle only if:
- Quick Gate passed
- Diagnostic passed or Planner explicitly overrides with written reason
- Promotion-set risk is acceptable
- protected submission assets remain reproducible
- the experiment log is complete
