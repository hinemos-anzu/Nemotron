# Specification Research Result v1.0

## 1. Purpose
This document consolidates the deep-dive research findings for the improvement strategies used in the NVIDIA Nemotron Model Reasoning Challenge project.
It is intended to be the final research baseline used to write implementation specifications and assign coding-design work to the Planner, Generator, and Kaggle execution role.

## 2. Competition and public-evidence baseline
Public competition materials describe the challenge as improving reasoning techniques using NVIDIA Nemotron open models on a novel benchmark. Public competition and ecosystem materials also indicate that participants are expected to explore prompting, data filtering, synthetic data generation, reinforcement learning, and lightweight fine-tuning. The public competition page identifies the challenge as a Featured Prediction Competition. Public model and ecosystem materials position Nemotron 3 Nano as an open baseline model family. Public mirror datasets and public winning-submission materials show that the problem space includes at least bit_manipulation, gravity, unit_conversion, cipher, numeral, and equation-like categories, and that public winning-submission analysis explicitly tracks training logprob, minimum logprob, token-level traces, and category-level synthetic/corpus/training/metrics views.

## 3. Project operating assumptions
- Team structure: Planner / Generator / Kaggle execution role.
- Primary goal: improve accuracy while reducing time-to-decision for each strategy experiment.
- Evaluation order: Quick Gate -> Diagnostic -> Promotion -> Kaggle.
- One experiment changes one main variable only.
- Submission asset flow is protected.
- Current protected baseline submission flow includes adapter conversion, offline asymmetric SVD surgery, key rename handling, expert unfuse, gate_x_to_in_proj_merge, and submission.zip generation.
- Current validated submission asset bundle consists of README.md, checkpoint_complete, adapter_model.safetensors, and adapter_config.json.

## 4. Priority conclusion
The strongest practical mainline remains:
1. held-out validation / evaluation foundation
2. min-logprob analysis foundation
3. easy-task miss recovery
4. low-min-logprob reweighting
5. training-serving misalignment quantification and repair
6. deterministic CoT redesign
7. bit manipulation specialized strengthening
8. post-finetune after conversion

Heavy research tracks should remain separate from mainline sprint execution:
9. token-direct generation pipeline
10. synthetic problem expansion
11. cryptarithm breakthrough research

## 5. Strategy deep-dive findings

### S1. Held-out validation / evaluation foundation
**Definition**
Create three stable evaluation tiers: Quick Gate, Diagnostic, and Promotion.

**Why it matters**
Without this layer, later strategy decisions become noisy and slow. Public winning-submission artifacts indicate that effective work separates synthetic, corpus, training, and metrics views rather than relying on one leaderboard number.

**Target layer**
Evaluation datasets, manifests, evaluation runners, metric aggregation, reference baseline reports.

**Expected effect**
- Faster rejection of weak ideas.
- Lower dependence on public leaderboard.
- More reproducible experiment decisions.

**Difficulty**
Medium.

**Implementation notes**
- Freeze sample IDs for at least one sprint.
- Tag each sample with category, difficulty, and failure mode.
- Maintain a conversion-sensitive subset.
- Store baseline reference results per evaluation-set version.

**Fast-fail condition**
- Same baseline produces conflicting judgments without explanation.
- Category coverage is too weak to detect regressions.
- Quick Gate still cannot reject obvious failures early.

**Output contract**
- quick_gate_v1
- diagnostic_v1
- promotion_v1
- category manifest
- baseline reference report

### S2. Min-logprob analysis foundation
**Definition**
Track the minimum-confidence token or local segment in each reasoning trace and connect it to category, failure mode, and pre/post-conversion state.

**Why it matters**
Public winning-submission artifacts explicitly track per-problem training logprob, minimum logprob, and token-level trace changes. This supports using min logprob as a real control signal rather than a decorative metric.

**Target layer**
Inference logging, trace parser, token-level metric extraction, dashboard/report generation.

**Expected effect**
- Identify brittle tokens and trace positions.
- Turn vague regressions into precise coding targets.
- Feed S4, S5, S6, and S7 directly.

**Difficulty**
Medium.

**Implementation notes**
- Required fields: sample_id, category, run_id/epoch, min_logprob, token, local context, correctness, format/extraction status, pre/post-conversion label.
- Sample-only summaries are insufficient.
- Must save token-level exemplars.
- Logging overhead must remain acceptable for repeated use.

**Fast-fail condition**
- Only averages or histograms are produced.
- Token/context exemplars are missing.
- Results cannot be mapped back to category and failure mode.
- Outputs are not reusable by S4/S5/S6/S7.

**Output contract**
- ranked brittle sample list
- token-fragility list
- conversion-sensitive brittle subset
- failure-mode taxonomy updates

### S3. Easy-task miss recovery
**Definition**
Eliminate avoidable losses in easy categories such as numeral, unit_conversion, gravity, and cipher by fixing format drift, answer extraction failures, repeated collapse, and unstable final wording.

**Why it matters**
This is score protection rather than a moonshot. Public category mirrors confirm that these are real standalone categories worth guarding explicitly.

**Target layer**
Easy-category trace templates, output formatting, extraction rules, easy-category training-data slice.

**Expected effect**
- Immediate score recovery.
- Better reproducibility.
- Fewer false negatives from avoidable delivery mistakes.

**Difficulty**
Low to medium.

**Implementation notes**
- Define accepted-output contract per easy subcategory.
- Treat formatting as part of correctness.
- Keep regression tolerance near zero.
- Do not mix with bit or conversion work in the same experiment.

**Fast-fail condition**
- Easy categories do not improve or become unstable.
- Format failures increase.
- Gains come only from hacks that hurt other categories.

**Output contract**
- easy failure pattern catalog
- output contract updates
- extraction/formatting guardrail updates

### S4. Low-min-logprob reweighting
**Definition**
Replay or upweight persistently low-min-logprob samples during training.

**Why it matters**
This is one of the most practical follow-ups once S2 exists. It converts brittle-subset analysis into a direct training action.

**Target layer**
Training sampler, replay lists, weighting config.

**Expected effect**
- Local robustness gains.
- Efficient finishing improvements.
- A concrete test of whether brittle subset repair is enough.

**Difficulty**
Low to medium.

**Implementation notes**
- Reweight only a clearly defined subset from S2.
- Prevent holdout leakage.
- Track whether gains generalize.
- Cap replay ratio to limit overfit risk.

**Fast-fail condition**
- Target subset does not improve.
- Easy categories regress.
- Gains remain isolated to the diagnostic subset only.

**Output contract**
- replay/reweight manifest
- target-subset before/after comparison
- replay ratio guidance

### S5. Training-serving misalignment quantification and repair
**Definition**
Measure and reduce the performance gap between the training-side adapter and the post-conversion submission-side adapter.

**Why it matters**
This is the most efficient recovery path when learned ability is lost during conversion, compression, surgery, or submission packaging.

**Target layer**
Adapter conversion scripts, surgery scripts, SVD path, merge logic, key mapping, pre/post-conversion evaluation.

**Expected effect**
- Recover already learned performance lost at submission time.
- Shrink pre/post-conversion gap.
- Stabilize easy and symbolic categories when conversion damage is causal.

**Difficulty**
High.

**Implementation notes**
- Compare pre/post-conversion on the exact same sample IDs.
- Track total accuracy, category accuracy, brittle-sample accuracy, min-logprob deltas, and format failures.
- Change one conversion variable at a time: SVD precision, rank allocation, merge order, lm_head handling, key mapping, or architecture-specific surgery.
- Preserve submission asset reproducibility at every step.

**Fast-fail condition**
- Pre/post-conversion gap is not measured directly.
- Conversion repair is mixed with data or prompt changes.
- Submission asset reproducibility breaks.

**Output contract**
- pre/post-conversion delta table
- brittle-sample pre/post report
- residual-gap decision for S8

### S6. Deterministic CoT redesign
**Definition**
Redesign traces so that under temperature 0 / greedy decoding the desired token path is as unique and stable as possible.

**Why it matters**
The improvement analysis identifies this as a mainline strategy: simplify reasoning into stable operation sequences, make expressions tokenization-aware, and explicitly cover rare operations.

**Target layer**
Trace generation templates, category-specific supervised traces, prompt/training text construction.

**Expected effect**
- Less drift under greedy decoding.
- Fewer tokenization-induced failures.
- Better target-category stability without relying on RL first.

**Difficulty**
Medium to high.

**Implementation notes**
- One step = one semantic action.
- Maintain forbidden unstable phrasings.
- Use S2 token-fragility evidence to drive edits.
- Redesign by category, not globally.
- Do not combine with reweighting or conversion changes in the same experiment.

**Fast-fail condition**
- Traces become longer but not more stable.
- Target-category brittle segments do not improve.
- Formatting or extraction gets worse.

**Output contract**
- stable phrasing list
- forbidden phrasing list
- category-specific template diffs

### S7. Bit manipulation specialized strengthening
**Definition**
Treat bit manipulation as a dedicated category strategy with its own failure taxonomy, notation contract, and trace design.

**Why it matters**
Public mirrors show bit_manipulation as a large, standalone category. The improvement analysis identifies it as one of the largest remaining upside areas.

**Target layer**
Bit-category CoT templates, bit-category training slice, bit error taxonomy, bit brittle manifest.

**Expected effect**
- Meaningful bit-category gains.
- Fewer rule-detection and positional failures.
- Better stability for fragile symbol/spacing outputs where relevant.

**Difficulty**
High.

**Implementation notes**
- Fix comparison order.
- Minimize symbol inventory.
- Define rule-candidate enumeration order.
- Separate bit failures from generic symbolic failures.
- Start from S2 token-fragility and S1 brittle cases.

**Fast-fail condition**
- Bit gain is tiny or inconsistent.
- Easy/formatting categories regress due to contamination.
- Improvement appears only on a tiny cherry-picked subset.

**Output contract**
- bit notation contract
- bit failure taxonomy
- rare-pattern manifest

### S8. Post-finetune after conversion
**Definition**
Fine-tune on the converted adapter form to recover residual performance lost after conversion/compression.

**Why it matters**
Useful only after S5 proves a residual post-conversion gap remains after conversion bugs and pipeline issues are addressed.

**Target layer**
Converted-adapter training path, residual-loss subset selection.

**Expected effect**
- Recover remaining post-conversion losses.

**Difficulty**
High.

**Implementation notes**
- Start only after S5 isolates a stable residual gap.
- Train on the post-conversion failure subset rather than the full world.
- Do not use S8 to hide unknown conversion bugs.

**Fast-fail condition**
- S5 is incomplete.
- Promotion-set gains do not hold.
- Submission asset reproducibility degrades.

**Output contract**
- residual-gap recovery report
- post-conversion fine-tune artifact comparison

### S9. Token-direct generation pipeline
**Definition**
Construct reasoning traces as token sequences directly rather than text-first traces.

**Why it matters**
Potentially reduces tokenization traps at the source, but infrastructure cost is high.

**Target layer**
Tokenizer-aware data builder, token-level trace serialization, token-level debug tools.

**Expected effect**
- Better control of brittle token sequences.

**Difficulty**
Very high.

**Implementation notes**
- Pilot only.
- Validate tokenizer round-trip behavior.
- Keep a text rendering path for debugging.

**Fast-fail condition**
- Debugging slows down more than expected benefit.
- Token-direct control does not reduce target failure modes.

### S10. Synthetic problem expansion
**Definition**
Generate new problems to expand training or validation coverage for rare patterns.

**Why it matters**
Potentially useful as a diagnostic amplifier and rare-pattern supplement, but high quality control is mandatory.

**Target layer**
Problem generators, provenance tracking, synthetic validation subsets.

**Expected effect**
- Better rare-pattern coverage.
- More scalable stress testing.

**Difficulty**
High.

**Implementation notes**
- Every synthetic sample must keep provenance.
- Never mix unlabeled synthetic items into core Promotion evaluation.
- Use synthetic first for diagnosis, not proof of success.

**Fast-fail condition**
- Synthetic samples are noisy or inconsistent.
- Synthetic gains do not transfer to real held-out samples.

### S11. Cryptarithm breakthrough research
**Definition**
A research-track effort for cryptarithm-like bottlenecks requiring new reasoning structure rather than cleanup.

**Why it matters**
High upside, but too slow and uncertain for mainline sprint execution.

**Target layer**
Separate research prototypes and dedicated datasets.

**Expected effect**
- Long-term ceiling lift.

**Difficulty**
Very high.

**Implementation notes**
- Keep separate from the mainline sprint.
- Require explicit Planner approval before Kaggle budget use.

**Fast-fail condition**
- Cannot be judged quickly.
- Consumes budget better spent on higher-probability strategies.

## 6. Final operating recommendation
Mainline coding-design work should proceed in this order:
1. S1 evaluation foundation finalization
2. S2 min-logprob logging and brittle-sample extraction
3. S3 easy-task miss recovery and S5 misalignment auditing in parallel
4. S4 reweighting on S2 brittle subset
5. S6 deterministic CoT redesign by category
6. S7 bit manipulation specialization
7. S8 only if residual post-conversion gap remains

S9-S11 remain outside the mainline sprint until Tier A and Tier B are stable.

## 7. Definition of implementation-ready work
A coding-design ticket is implementation-ready only if it specifies:
- strategy ID
- target layer
- allowed changes
- forbidden changes
- expected effect
- acceptable collateral damage
- fast-fail criteria
- required outputs
- next-strategy feedback deliverable
