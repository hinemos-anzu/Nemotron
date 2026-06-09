# B3_GOLDEN_EXACT_REGEN Diagnostics Template

Fill this file during the later execution phase before any Kaggle submission. Do not fill it with guessed values.

## 1. Run identity

- Experiment: `B3_GOLDEN_EXACT_REGEN`
- Date/time UTC:
- Operator / role:
- Git commit:
- Execution environment:
- Notebook or command path:
- Base model path:
- Adapter source path:
- Output directory:

## 2. Golden fixed-condition confirmation

| Fixed condition | Expected | Observed | PASS/HOLD/FAIL | Evidence path or note |
|---|---|---|---|---|
| Training data unchanged | Golden Baseline |  |  |  |
| Adapter rank map unchanged | Golden Baseline |  |  |  |
| `target_modules` unchanged | Golden Baseline |  |  |  |
| Conversion script unchanged | Golden Baseline |  |  |  |
| Gate/X split | `16/16` |  |  |  |
| Generation config unchanged | Golden Baseline |  |  |  |
| Root `submission.zip` packaging unchanged | Golden Baseline |  |  |  |

## 3. Adapter structural diagnostics

| Diagnostic | Expected | Observed | PASS/HOLD/FAIL | Evidence path or command output |
|---|---|---:|---|---|
| Gate/X split | `16/16` |  |  |  |
| Tensor count | Golden-compatible |  |  |  |
| `max_rank_seen` | Golden-compatible |  |  |  |
| Rank violations | `0` / none |  |  |  |
| NaN/Inf | `0` / none |  |  |  |
| Adapter key consistency | matches Golden modules |  |  |  |
| Adapter module consistency | matches Golden `target_modules` |  |  |  |

## 4. Root archive diagnostics

Complete this section only in an approved artifact-generation phase.

| Diagnostic | Expected | Observed | PASS/HOLD/FAIL | Evidence |
|---|---|---|---|---|
| Root artifact name | `submission.zip` |  |  |  |
| Root archive layout | Golden packaging layout |  |  |  |
| No nested accidental archive | none |  |  |  |
| Required adapter files present | Golden payload |  |  |  |
| Unexpected large/debug files | none |  |  |  |
| Archive SHA256 | recorded |  |  |  |

## 5. Local validation / smoke diagnostics

- Command or notebook used:
- Output files:
- Summary metric:
- Structural diagnostic log path:
- Any mismatch versus Golden:

## 6. Submission decision gate

- [ ] All Golden fixed conditions are PASS.
- [ ] Adapter structural diagnostics are PASS.
- [ ] Root archive diagnostics are PASS, if artifact generation was approved.
- [ ] No prohibited action was performed.
- [ ] Candidate is promoted only as Golden exact control, not as an improvement candidate.

## 7. Public LB result

Fill only after an approved Kaggle submission in a later phase.

- Kaggle submission date/time UTC:
- Submission artifact SHA256:
- Public LB:
- Decision: ADOPT_CONTROL / REJECT_CONTROL / HOLD
- Interpretation if not `0.86` equivalent:
