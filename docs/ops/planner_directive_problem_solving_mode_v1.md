# Planner Directive: Problem-Solving Mode v1

## Purpose
This directive changes the operating mode for Generator work.
From this point forward, Generator must not stop at checklist completion or blocked-state reporting.
Generator must identify the root cause of blockers, propose the smallest valid corrective action, and either execute that corrective action if it is in scope or escalate with a concrete next ticket if it is out of scope.

## Why this directive exists
Recent work showed a repeated pattern:
- required artifacts were created
- reports were well structured
- blockers were reported honestly
- but the work stopped before converting the blocker into the next solvable step

This directive exists to stop wasted iteration cycles.

## New operating rule
For every ticket, Generator must produce not only:
- what happened
- what passed
- what blocked

but also:
- what the blocker actually means at the system level
- whether the blocker is caused by environment, missing files, wrong scope, wrong execution path, or missing permissions
- the minimum next action needed to remove the blocker
- whether that next action can be executed immediately inside current scope
- if not, the exact follow-up ticket or request required

## Required blocker diagnosis taxonomy
Every blocker must be labeled as exactly one primary type:
- ENVIRONMENT_MISSING
- REPO_STATE_MISMATCH
- REQUIRED_INPUT_MISSING
- WRONG_EXECUTION_PATH
- PERMISSION_OR_ACCESS_BLOCK
- OUT_OF_SCOPE_FOR_CURRENT_TICKET
- UNKNOWN_ROOT_CAUSE

And every blocker must include:
- observed symptom
- likely root cause
- evidence
- confidence level
- smallest unblocking action

## Required response behavior
### If blocker is in-scope and low-risk
Generator must attempt the fix in the same ticket instead of stopping.

Examples:
- wrong local path assumptions in an evaluation wrapper
- missing repo file path lookup that can be corrected without changing strategy scope
- reporting script unable to find frozen assets because of path mismatch

### If blocker is out-of-scope but clearly diagnosable
Generator must propose a concrete next ticket and pre-fill:
- objective
- single main variable
- in scope
- out of scope
- required outputs
- fast-fail criteria

### If blocker is caused by environment that cannot be solved locally
Generator must produce a source-of-truth execution plan using the same schemas and artifacts, not just say BLOCKED.

## Mandatory end-of-report section
Every Generator report must now end with:

## Root cause and unblocking action
- Blocker type:
- Root cause:
- Evidence:
- Can I fix this now within scope? yes/no
- If yes: what I changed
- If no: exact next ticket needed
- What has been preserved for the next ticket so no work is lost

## Planner override rule
Planner may reject any report as incomplete if:
- BLOCKED is reported without root-cause classification
- there is no smallest unblocking action
- the report forces Planner to infer the next ticket manually

## Goal of this directive
Reduce wasted iterations and move from passive reporting to active problem resolution while preserving one-main-variable discipline.
