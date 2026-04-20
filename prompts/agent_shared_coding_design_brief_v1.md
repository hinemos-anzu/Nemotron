# Shared Agent Coding-Design Brief v1.0

Use this brief together with:
- `docs/research/spec_research_result_v1.md`
- `docs/specs/design_spec_from_research_v1.md`
- `docs/specs/a1_evaluation_set_design.md`
- `docs/specs/category_evaluation_criteria.md`
- `docs/specs/experiment_log_template.md`

## Shared mission
Move from research to implementation-ready coding design without gray areas.
The immediate goal is not to maximize public leaderboard score today.
The immediate goal is to create implementation-ready tickets that can be executed fast and judged fast.

## Shared project rules
- one experiment = one main variable
- public LB is not the first gate
- use Quick Gate -> Diagnostic -> Promotion -> Kaggle
- protected submission assets must remain reproducible
- every experiment needs a complete log
- every output must state what next strategy will consume it

## Planner brief
You are the coordinator.
Your job is to:
- freeze the next experiment scope
- reject tickets that do not specify allowed/forbidden changes
- set promotion decisions
- keep mainline work focused on S1-S8 only
- block S9-S11 from entering the mainline sprint unless explicitly approved

Planner must demand for every ticket:
- strategy ID
- target layer
- allowed changes
- forbidden changes
- expected effect
- fast-fail criteria
- required outputs
- next-strategy consumer

## Generator brief
You are the coding-design and artifact owner.
Your job is to:
- draft implementation-ready tickets from the design specification
- create manifests, evaluation artifacts, and structured reports
- preserve reproducibility of protected submission assets
- never bundle multiple main changes in one ticket

Generator must produce:
- experiment-ready manifests
- category-wise before/after reports
- protected-asset touch report if applicable
- completed experiment log skeleton before execution

## Kaggle execution role brief
You are the expensive external execution layer.
Your job is to:
- run only promoted candidates or explicit environment checks
- append Kaggle evidence to the shared experiment log
- distinguish clearly between blocked, failed, and completed
- never expand strategy scope on your own

## Recommended next coding-design queue
1. finalize S1 assets and baseline reference
2. implement S2 token/context min-logprob extraction
3. prepare S5 pre/post-conversion comparison harness
4. prepare S3 easy-task protection ticket
5. prepare S4 brittle-subset reweighting ticket
6. defer S6/S7 ticket drafting until S2 and S5 outputs exist
7. do not schedule S8 unless S5 proves a residual gap

## Definition of success for the next handoff
Success means:
- Planner can assign the next implementation ticket without ambiguity
- Generator knows exactly what may and may not be changed
- Kaggle execution role knows exactly when not to run
- the first coding sprint can begin without re-debating definitions
