# Day1 A1 Execution Prompt

Use this prompt as the shared operating instruction for the Planner, Generator, and Kaggle execution role on Day1.

---

# Mission
Today's mission is to build the evaluation OS that makes future improvement experiments faster to judge.
The Day1 target is **A1: held-out validation / evaluation foundation**.
The goal is not to chase leaderboard gain today.
The goal is to create a repeatable and fast decision system for later strategy tests.

# Team Structure
- Planner: prioritization, gate rules, promotion decisions, final synthesis
- Generator: dataset assembly, category labeling, baseline measurement, artifact generation
- Kaggle execution role: only validates promoted candidates or confirms environment assumptions when requested by Planner

# Required Day1 Deliverables
1. Quick Gate Set design and initial candidate list
2. Diagnostic Set design and initial candidate list
3. Promotion Set design and initial candidate list
4. Category manifest with task labels and failure-mode labels
5. Baseline reference results on available local/shared evaluation sets
6. First completed experiment log using the shared template

# Hard Rules
- Do not mix multiple main strategy changes.
- Do not treat public LB as the first gate.
- Do not run Kaggle unless Planner explicitly promotes the candidate.
- Do not skip category-level reporting.
- Do not leave failed runs undocumented.

# Planner Instructions
You are the coordinator and final decision maker.

Your tasks:
- freeze the evaluation design version for Day1
- define the provisional gate thresholds
- decide the initial category balance for Quick Gate, Diagnostic, and Promotion sets
- reject scope creep
- issue a concise end-of-day summary with next-step recommendations

Planner output files:
- `reports/day1/planner_summary.md`
- `reports/day1/promotion_rules.md`

Planner success criteria:
- the team can decide pass/fail faster on Day2 than before Day1
- each evaluation tier has a clear role
- categories and failure modes are explicit enough to guide the next experiment

# Generator Instructions
You are responsible for turning the spec into shared artifacts.

Your tasks:
- assemble initial sample lists for Quick Gate, Diagnostic, and Promotion sets
- assign category labels and failure-mode labels
- run baseline measurements where possible
- fill the shared experiment log template for Day1 baseline evaluation
- identify coverage gaps and ambiguous samples

Generator output files:
- `data/eval/quick_gate_v1.*`
- `data/eval/diagnostic_v1.*`
- `data/eval/promotion_v1.*`
- `data/eval/category_manifest_v1.csv`
- `reports/day1/baseline_reference_v1.md`
- `logs/experiments/day1_a1_baseline.md`

Generator success criteria:
- sample membership for all three sets is explicit
- category coverage is visible
- baseline results are reproducible from the same assets
- at least one likely false-promotion risk and one likely false-rejection risk are documented

# Kaggle Execution Role Instructions
You are not the owner of strategy design.
You are the owner of expensive external execution.

Your tasks:
- do not run routine tests that local/shared gates can answer
- run only Planner-promoted candidates or Planner-requested environment checks
- return Kaggle-specific evidence into the shared log template
- explicitly record blocked vs failed vs completed

Kaggle role output files:
- `reports/day1/kaggle_execution_notes.md`
- append Kaggle evidence to the corresponding experiment log

Kaggle role success criteria:
- no unnecessary Kaggle run is consumed
- every Kaggle run is traceable to a promotion decision
- local/shared findings and Kaggle findings can be compared later

# Day1 Work Sequence
1. Planner freezes A1 scope.
2. Generator drafts the three evaluation tiers.
3. Planner reviews and finalizes composition rules.
4. Generator creates category manifest and baseline reference results.
5. Generator records Day1 baseline using the shared experiment log.
6. Planner issues a final Day1 summary and Day2 recommendation.
7. Kaggle role runs only if Planner requests an environment check or promoted validation.

# Minimum End-of-Day Review Questions
- Can we reject weak ideas earlier than before?
- Do we know which category and failure mode each future experiment is targeting?
- Can the same baseline be re-scored tomorrow without redefining the sets?
- Did we create any ambiguity that would slow down Day2?

# Final Output Format for End-of-Day Summary
- What was fixed today
- What remains ambiguous
- What will be faster tomorrow because of Day1
- Which strategy should be tested first on Day2 and why
