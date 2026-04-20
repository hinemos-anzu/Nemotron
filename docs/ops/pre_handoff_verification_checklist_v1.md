# Pre-Handoff Verification Checklist v1

## Purpose
This checklist is mandatory before Planner issues any execution instruction to Generator or the Kaggle execution role.
Its purpose is to prevent invalid repository, branch, or file references from being communicated as facts.

## Mandatory checks
Before sending an execution instruction, Planner must verify all of the following:

### A. Repository
- Repository name is correct.
- Repository is accessible.

### B. Working branch
- Working branch name is explicitly decided.
- Working branch existence has been verified.
- If the branch does not exist, Planner must create it before issuing the instruction.
- Planner must never state that a branch exists unless it has been verified.

### C. Entry path
- Entry script path exists, or
- the instruction explicitly says the file is to be created in this task.

### D. Frozen assets
- Required frozen asset paths are listed.
- Their existence has been checked when they are supposed to already exist.

### E. Ticket references
- Referenced tickets or specs actually exist in the repository, or
- the instruction explicitly marks them as missing and blocked.

## Required block in every execution prompt
Every execution prompt must include:
- Repository
- Base branch
- Working branch
- Entry script path
- Required frozen assets
- Whether the branch and entry path were verified or are expected to be created

## Failure rule
If any of the above are unknown, Planner must not present them as facts.
Unknown items must be marked explicitly as:
- VERIFIED
- TO_BE_CREATED
- BLOCKED_MISSING

## Process failure definition
The following count as Planner-side process failures:
- naming a non-verified branch as if it exists
- naming a non-verified file path as if it exists
- omitting repository or branch information from an execution handoff

## Immediate usage rule
This checklist takes effect immediately for all future Planner -> Generator and Planner -> Kaggle execution handoffs.
