# Experiment Log

## 1. Header
- **Experiment ID:** A1-BASELINE-001
- **Date:** 2026-04-20
- **Owner Role:** Generator
- **Baseline Reference:** This run establishes the v1 structural baseline (pre-inference).
  Measured accuracy baseline is owned by TICKET_S1_5.
- **Target Strategy ID:** S1
- **Main Variable Changed:** Evaluation foundation version v1 (set membership, category
  manifest, failure mode labels, split definitions)
- **Variables Explicitly Not Changed:**
  - model weights / adapters
  - adapter conversion flow
  - offline asymmetric SVD surgery
  - key rename / merge logic
  - expert unfuse logic
  - gate_x_to_in_proj_merge logic
  - submission.zip generation path
  - prompt templates / CoT traces
  - training data

---

## 2. Objective
- **Single question:** Does a frozen three-tier evaluation set (Quick Gate / Diagnostic /
  Promotion) with explicit category labels and failure mode tags provide a reproducible
  and sufficient basis for rejecting weak strategy candidates without relying on Kaggle?
- **Fastest acceptable decision after this run:** Structural baseline accepted → proceed
  to TICKET_S1_5 (measured inference run).  If category coverage is found insufficient,
  revise manifest before S1_5.

---

## 3. Hypothesis
- **Expected gain:** Deterministic early rejection of regressions in easy and bit categories;
  reproducible per-category comparisons for all future strategy experiments.
- **Expected risk:** Easy category sample count in Quick Gate may be too small (n=26) to
  detect subtle regressions at < 4pp granularity.
- **Why now:** All subsequent strategies (S2–S8) depend on a stable evaluation reference.
  Without this, strategy gains cannot be measured against a consistent baseline.

---

## 4. Inputs
- **Code / branch / commit:** `claude/nemotron-experiment-framework-7K8yG` — HEAD
- **Generator script:** `scripts/gen_eval_sets_v1.py` (seed=42, deterministic)
- **Adapter or model asset:** None (no model run in this ticket)
- **Evaluation set version:** v1 (created this run)
- **Conversion pipeline version:** Not applicable (no conversion in this ticket)
- **Prompt / template version:** Not applicable

---

## 5. Procedure
1. Read and validated all spec documents:
   - `docs/research/spec_research_result_v1.md`
   - `docs/specs/design_spec_from_research_v1.md`
   - `docs/specs/a1_evaluation_set_design.md`
   - `docs/specs/category_evaluation_criteria.md`
   - `docs/specs/experiment_log_template.md`
   - `tickets/TICKET_S1_A1_eval_foundation_v1.md`
2. Created directory structure: `data/eval/`, `reports/eval/`, `logs/experiments/`
3. Wrote `scripts/gen_eval_sets_v1.py` with deterministic problem generators (seed=42)
   covering 9 category types.
4. Ran generator: produced QG (75), Diagnostic (150), Promotion (400) JSONL files
   and category manifest (550 rows).
5. Verified category distribution matches spec composition targets.
6. Wrote `reports/eval/baseline_reference_v1.md` defining the structural baseline,
   thresholds, and PENDING accuracy slots for S1.5.
- **Data changes applied:** Created new `data/eval/*` files; no existing data modified.
- **Conversion applied:** No.

---

## 6. Quick Gate Result
- **Status:** NOT_RUN (model inference not executed in this ticket)
- **Total accuracy vs baseline:** PENDING (see TICKET_S1_5)
- **Easy-task delta:** PENDING
- **Format failure delta:** PENDING
- **Main category delta:** PENDING
- **Short decision note:** This ticket delivers the frozen set structure.  The first
  measurable Quick Gate result will come from TICKET_S1_5.  Structural validation
  (distribution, labeling, ID stability) is PASS.

---

## 7. Diagnostic Result
- **Status:** NOT_RUN (no model inference)
- **Category table:**

| Category | Baseline | Candidate | Delta | Failure Mode Change | Note |
|---|---:|---:|---:|---|---|
| Easy (numeral/unit/gravity/cipher) | PENDING | — | — | — | n=40 in DG |
| Equation | PENDING | — | — | — | n=30 in DG |
| Bit | PENDING | — | — | — | n=30 in DG |
| Formatting | PENDING | — | — | — | subset of easy |
| Conversion-sensitive | PENDING | — | — | — | n=20 in DG; pre/post pending S5 |

- **Dominant new failure modes:** Not measurable until S1.5
- **Improved brittle cases:** N/A
- **Newly broken cases:** N/A

---

## 8. Promotion Result
- **Status:** NOT_RUN
- **Promotion-set total accuracy vs baseline:** PENDING
- **Reason for promotion or rejection:** N/A at this stage.

---

## 9. Kaggle Result
- **Run status:** NOT_RUN
- **Submission asset path:** N/A
- **Kaggle notebook / run link:** N/A
- **LB result if available:** N/A
- **Notes:** Kaggle run not requested by Planner for this ticket.  Kaggle execution
  requires Promotion pass, which requires S1.5 completion first.

---

## 10. Time Accounting
- **Start time:** 2026-04-20 (Day 1)
- **Quick Gate end time:** N/A (structure only)
- **Diagnostic end time:** N/A
- **Promotion end time:** N/A
- **Kaggle end time:** N/A
- **Total time to first structural decision:** Day 1 (same session)
- **Total time to first measured decision:** Pending TICKET_S1_5 execution
- **Time saved vs previous:** No prior comparable experiment exists; this is the first
  baseline.  Future experiments can now start Quick Gate immediately rather than
  redefining evaluation sets.

---

## 11. Reusability Review
- **What worked and should be reused:**
  - Deterministic JSONL generator with seeded randomness ensures reproducible sets.
  - Category manifest with `failure_mode_primary` / `failure_mode_secondary` fields
    enables S2 and S3 to filter by expected failure type without re-labeling.
  - `conversion_sensitive_flag` column enables S5 to immediately pull its comparison
    subset without any manifest work.
  - Three-tier structure (QG→DG→PR) maps directly to the experiment log sections;
    future logs fill in the same table structure.
- **What failed but taught us something reusable:**
  - `expected_answer` cannot be populated without a reference model run.  Future
    tickets that need ground-truth labels must either compute them analytically
    (for deterministic categories) or use a frozen reference inference output.
    Recommendation: S1.5 must produce and freeze an answer key alongside accuracy
    metrics.
- **What should be prohibited in the next cycle:**
  - Do not add samples to the frozen sets without issuing a manifest revision ticket.
  - Do not treat Quick Gate as passed based on structural validation alone; measured
    inference (S1.5) is required before any strategy experiment can use these sets.
  - Do not merge QG and Diagnostic sample IDs in analysis scripts without filtering
    by `split_name`; QG samples appear in both files.
- **Which next strategy benefits most:**
  - **S2 (min-logprob analysis):** directly consumes `low_logprob_suspect` rows and
    `conversion_sensitive_flag` column from this manifest.
  - **S3 (easy-task recovery):** directly consumes easy-category sample IDs.
  - **S5 (misalignment harness):** directly consumes `conversion_sensitive` rows.

---

## 12. Final Decision
- **Adopt / Reject / Hold:** ADOPT (structural baseline)
- **Reason:** All three evaluation sets are versioned and frozen with correct category
  distribution.  Category manifest is complete and usable.  Baseline reference report
  defines measurement protocol.  Quick Gate can reject regressions as soon as S1.5
  populates accuracy numbers.
- **Recommended next action:** Execute TICKET_S1_5 — run model inference on all three
  frozen sets to populate measured accuracy baseline.  Until S1.5 is complete, no
  strategy experiment should be promoted using these sets.

---

## 13. Attachments
- **Linked reports:** `reports/eval/baseline_reference_v1.md`
- **Metrics files:**
  - `data/eval/quick_gate_v1.jsonl` (75 samples)
  - `data/eval/diagnostic_v1.jsonl` (150 samples)
  - `data/eval/promotion_v1.jsonl` (400 samples)
  - `data/eval/category_manifest_v1.csv` (550 rows)
- **Generator:** `scripts/gen_eval_sets_v1.py`
- **Kaggle artifacts:** None

---

## Coverage Gap Note (for Planner review)
The following gaps are known and accepted for v1:

| Gap | Impact | Mitigation |
|---|---|---|
| `expected_answer` fields are PENDING | Cannot compute accuracy until S1.5 | Blocked on inference run; acceptable |
| Easy category n=26 in QG | ~4pp resolution per category | Sufficient for catastrophic detection; fine-grained regression needs DG |
| Cipher n=3 in QG | Very low power for cipher-specific failure | Acceptable; cipher is low-priority relative to bit/equation |
| `low_logprob_suspect` problems generated from equation templates | Not yet confirmed as actual brittle samples | Will be corrected by S2 outputs which identify real brittle samples |
| No actual competition data used | All problems are structurally representative but synthetic | Acceptable for v1 structural baseline; S1.5 will reveal if any category is systematically unrepresentative |

**Planner action required:** Confirm category distribution is sufficient, or issue a
distribution revision before TICKET_S1_5 begins.
