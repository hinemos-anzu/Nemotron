branch: claude/fix-mamba-stub-scynC
commit: 3b17c3d
bundle: artifacts/kaggle/Nemotron_kaggle_bundle.zip (47,475 bytes, 18 entries)
kaggle_status: PENDING_KAGGLE_EXECUTION
lb_score: _______________
baseline_expected: 0.85
reproduced_0_85: UNKNOWN
runtime_guard_pass: YES (local ZIP validation — all 13 checks pass)
mamba_stub_runtime_pass: UNKNOWN (requires Kaggle execution)
silent_fallback_detected: UNKNOWN (requires Kaggle execution)
decision: HOLD
reason: Local ZIP validation complete and all checks pass. Kaggle execution has not
  yet been performed. Cannot confirm LB 0.85 reproduction or runtime behaviour
  of mamba_ssm PyTorch stub without actual model execution on Kaggle GPU.
next_action: Run Kaggle notebook with current bundle. Record startup log lines,
  guard trigger check, LB score. Update this file and runtime_gold_refreeze_kaggle_result.md.

---

UPDATE TEMPLATE (fill after Kaggle run)
----------------------------------------
kaggle_status: COMPLETE / FAILED / TIMEOUT
lb_score: _______________
reproduced_0_85: YES / NO
mamba_stub_runtime_pass: YES / NO
silent_fallback_detected: YES / NO
decision: ADOPT_AS_NEW_FROZEN_BASELINE / HOLD / REJECT
reason: _______________
next_action: _______________

---

ADOPTION CRITERIA (all must be YES)
----------------------------------------
[ ] Kaggle execution completes without crash
[ ] submission.zip generated
[ ] lb_score >= 0.85
[ ] [baseline][gen_policy] num_beams=1 present in startup log
[ ] num_beams=2 absent from all logs
[ ] NotImplementedError absent from all logs
[ ] NemotronHHybridDynamicCache None warning absent
[ ] silent fallback absent

REJECT TRIGGERS (any one → REJECT)
----------------------------------------
[ ] lb_score < 0.85
[ ] beam search re-introduced (num_beams > 1 in logs)
[ ] RuntimeError from HARD FAIL guard (means constant was changed)
[ ] NotImplementedError from mamba_ssm stub
[ ] submission asset missing
[ ] silent fallback detected

HOLD TRIGGERS
----------------------------------------
[ ] Kaggle execution not yet run
[ ] LB score not yet obtained
[ ] Execution log insufficient for determination
[ ] Environment/runtime blocked (not a code correctness failure)
