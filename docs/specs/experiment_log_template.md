# Experiment Log Template v1.0

## Purpose
This template standardizes how Planner, Generator, and the Kaggle execution role share experiment evidence.
Every experiment must be logged, including failed experiments.

## File Naming Convention
Recommended path:
- `logs/experiments/YYYYMMDD_<experiment_id>.md`

Recommended experiment id format:
- `A1-QG-001`
- `A2-LOGPROB-002`
- `A5-MISALIGN-003`

---

# Experiment Log

## 1. Header
- Experiment ID:
- Date:
- Owner Role: Planner / Generator / Kaggle Runner
- Baseline Reference:
- Target Strategy ID:
- Main Variable Changed:
- Variables Explicitly Not Changed:

## 2. Objective
- What is the single question this experiment is trying to answer?
- What is the fastest acceptable decision after this run?

## 3. Hypothesis
- Expected gain:
- Expected risk:
- Why this is being tried now:

## 4. Inputs
- Code / branch / commit:
- Adapter or model asset:
- Evaluation set version:
- Conversion pipeline version:
- Prompt / template version if applicable:

## 5. Procedure
- Steps run:
- Commands or notebooks used:
- Data changes applied:
- Conversion applied or not applied:

## 6. Quick Gate Result
- Status: PASS_TO_PROMOTION / FAIL_QUICK_GATE / HOLD_INCONCLUSIVE
- Total accuracy vs baseline:
- Easy-task delta:
- Format failure delta:
- Main category delta:
- Short decision note:

## 7. Diagnostic Result
- Status: PASS / FAIL / HOLD
- Category table:

| Category | Baseline | Candidate | Delta | Failure Mode Change | Note |
|---|---:|---:|---:|---|---|
| Easy |  |  |  |  |  |
| Equation |  |  |  |  |  |
| Bit |  |  |  |  |  |
| Formatting |  |  |  |  |  |
| Conversion-sensitive |  |  |  |  |  |

- Dominant new failure modes:
- Improved brittle cases:
- Newly broken cases:

## 8. Promotion Result
- Status: PROMOTE_TO_KAGGLE / FAIL_DIAGNOSTIC / HOLD_INCONCLUSIVE
- Promotion-set total accuracy vs baseline:
- Reason for promotion or rejection:

## 9. Kaggle Result
- Run status: NOT_RUN / RUN_COMPLETE / BLOCKED_ENVIRONMENT
- Submission asset path:
- Kaggle notebook / run link:
- LB result if available:
- Notes on mismatch between local and Kaggle evidence:

## 10. Time Accounting
- Start time:
- Quick Gate end time:
- Diagnostic end time:
- Promotion end time:
- Kaggle end time if run:
- Total time to first decision:
- Total time to final decision:
- Time saved vs previous comparable experiment:

## 11. Reusability Review
- What worked and should be reused?
- What failed but taught us something reusable?
- What should be prohibited in the next cycle?
- Which next strategy benefits most from this result?

## 12. Final Decision
- Adopt / Reject / Hold:
- Reason in one sentence:
- Recommended next action:

## 13. Attachments
- Linked reports:
- Metrics files:
- Plots or tables:
- Kaggle artifacts:

---

## Minimal Completion Rule
An experiment log is incomplete unless it contains:
- the single main variable changed
- Quick Gate result
- at least one category-level diagnostic judgment
- a final decision
- one reusable lesson

## Planner Review Checklist
Planner should reject the log if:
- multiple main variables were changed without declaration
- the evaluation set version is missing
- there is no before/after comparison
- the failure mode is not described
- the next action is not specified
