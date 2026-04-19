# Detailed Strategy Implementation Spec v1.0

## Purpose
This document defines the improvement strategies at an implementation-ready level.
Its job is to remove gray areas before coding starts.
For each strategy, this spec fixes:
- what the strategy means
- what code or data layer it is allowed to change
- what it must not change
- what signal is expected if it works
- what signal means it failed
- what coding mistakes are especially dangerous

This spec is derived from the project improvement analysis. The main thrust is to raise performance by combining deterministic CoT design, min-logprob-driven analysis and reweighting, training-serving misalignment fixes, bit manipulation strengthening, and robust held-out validation. fileciteturn3file4
The improvement analysis also states that the most practical combination is min logprob analysis + low-logprob reweighting + easy-task loss recovery + misalignment fixing. fileciteturn3file1turn3file3

---

# Global Coding Contract

## Shared assumptions
- Baseline submission asset flow must remain reproducible.
- Kaggle is the source of truth for final external validation.
- Submission.zip structure and adapter conversion path are protected assets. Current preserved assets include adapter conversion, offline asymmetric SVD surgery, key rename, expert unfuse, gate/x merge, and submission zip generation. fileciteturn3file5
- One experiment changes one main variable only.
- Every strategy must be evaluated through Quick Gate -> Diagnostic -> Promotion.

## Forbidden ambiguity
The following are not allowed in implementation tickets:
- vague verbs such as "improve", "stabilize", or "optimize" without an explicit code or data target
- touching both training data and conversion logic in the same experiment unless Planner explicitly marks one as frozen control and one as the single main change
- strategy definitions that do not specify target categories, expected failure modes, and rollback condition

## Required experiment fields for all strategies
Every coding task must define:
1. target layer
2. exact files or modules allowed to change
3. exact files or modules forbidden to change
4. intended category gain
5. acceptable collateral damage
6. rollback trigger

---

# Strategy Priority

## Tier A: must build first or use first
1. S1 held-out validation / evaluation foundation
2. S2 min-logprob analysis foundation
3. S3 easy-task miss recovery
4. S4 low-min-logprob reweighting
5. S5 training-serving misalignment quantification and repair

## Tier B: strong mainline model-improvement strategies
6. S6 deterministic CoT redesign
7. S7 bit manipulation specialized strengthening
8. S8 post-finetune after conversion

## Tier C: heavy research or infrastructure bets
9. S9 token-direct generation pipeline
10. S10 synthetic problem expansion
11. S11 cryptarithm breakthrough research

---

# S1. Held-Out Validation / Evaluation Foundation

## Definition
Create three stable evaluation tiers that can judge strategies quickly and consistently:
- Quick Gate
- Diagnostic
- Promotion

## Coding base information
This strategy is based on the project recommendation that held-out validation is the first requirement because otherwise all later strategy judgments become unreliable. fileciteturn3file1turn3file2

## Target layer
- evaluation data
- evaluation runners
- metric aggregation
- category manifest

## Allowed changes
- evaluation dataset files
- manifest CSV or JSONL
- scoring scripts
- report generators

## Forbidden changes
- model weights
- training data
- adapter conversion logic
- inference prompt templates

## Expected effect
- faster rejection of weak strategies
- lower dependence on public LB
- reproducible before/after decisions

## Difficulty
1. S1 held-out validation / 評価基盤整備（推奨度：★★★★★）
- 理由: 何が効いたかを正しく判定できないと、以後の改善が全部ぶれます。最初に入れる価値が最も高いです。

## Coding notes
- Keep the same sample IDs fixed across at least one full sprint.
- Every sample must carry category, difficulty, and failure-mode labels.
- Keep a separate manifest for conversion-sensitive samples.
- Store baseline reference results together with set version.

## Fast failure criteria
Fail this strategy if:
- two different runs on the same baseline produce conflicting judgments without explanation
- category coverage is too sparse to detect regressions
- the Quick Gate still takes too long to reject obvious failures

## Feedback to next strategy
S1 must produce:
- the brittle sample list used by S2 and S5
- category priors used by S3 and S7
- promotion rules used by Kaggle execution

---

# S2. Min-Logprob Analysis Foundation

## Definition
Measure where the model becomes locally brittle by tracking the minimum logprob token or segment in each reasoning trace.
The purpose is not aggregate loss monitoring but locating exactly which token or local segment collapses. fileciteturn3file1turn3file3turn3file4

## Target layer
- inference logging
- trace parser
- token-level metric aggregation
- dashboard/report scripts

## Allowed changes
- logging hooks
- metric extraction scripts
- post-processing dashboards

## Forbidden changes
- model weights in the same experiment
- training dataset composition in the same experiment
- prompt templates in the same experiment

## Expected effect
- identify brittle tokens and brittle trace positions
- turn vague failure clusters into concrete coding targets
- enable efficient data reweighting and CoT editing

## Difficulty
2. S2 min logprob 分析基盤の整備（推奨度：★★★★★）
- 理由: 投稿者の勝ち筋の核です。平均lossでは拾えない「壊れる1トークン」を見つけられます。

## Coding notes
- Record both absolute minimum logprob and its local context window.
- Aggregate by sample, category, failure mode, and pre/post-conversion state.
- Do not stop at summary histograms; store token/segment exemplars.
- Keep sample IDs stable so S4 can reweight exactly the same brittle subset.

## Fast failure criteria
Fail this strategy if:
- it only produces averages and no actionable low-confidence exemplars
- it cannot map low-confidence segments back to category/failure mode
- it creates too much logging overhead for regular use

## Feedback to next strategy
S2 must output:
- ranked brittle sample list for S4
- tokenization-sensitive expression list for S6 and S7
- conversion-sensitive low-confidence subset for S5

---

# S3. Easy-Task Miss Recovery

## Definition
Remove avoidable losses on easy categories such as numeral, unit conversion, gravity-style templates, and cipher-style templates by fixing formatting drift, extraction failure, repeated collapse, and unstable answer wording. fileciteturn3file1

## Target layer
- prompt / trace templates for easy categories
- answer formatting / extraction rules
- training data slice for easy categories

## Allowed changes
- easy-category templates
- answer-extraction normalizers
- easy-category data balance

## Forbidden changes
- bit manipulation logic
- adapter conversion logic
- unrelated hard-category CoT templates

## Expected effect
- immediate score-protection gains
- improved reproducibility
- fewer false negatives from avoidable delivery errors

## Difficulty
3. S3 easy task の取りこぼしゼロ化（推奨度：★★★★★）
- 理由: 工数が比較的小さい割に、安定した改善が見込めます。終盤ほど重要です。

## Coding notes
- Define an explicit accepted-output contract per easy subcategory.
- Treat format compliance as part of correctness, not a separate nice-to-have.
- Keep regression tolerance near zero.
- Isolate easy-template changes from harder logic changes.

## Fast failure criteria
Fail this strategy if:
- easy accuracy does not improve or remains unstable
- format failures increase on any easy subcategory
- gains come only from extraction hacks that hurt other categories

## Feedback to next strategy
S3 must produce:
- exact avoided-failure patterns for template cleanup
- updated format failure taxonomy
- protected-output conventions reused by S6

---

# S4. Low-Min-Logprob Reweighting

## Definition
Reweight or replay persistently low-min-logprob samples during training so the model spends more learning budget on the brittle subset. This is described as one of the most practical next-step improvements after min-logprob analysis. fileciteturn3file1turn3file3

## Target layer
- training data sampler
- curriculum / weighting config
- sample replay lists

## Allowed changes
- sample weights
- replay ratios
- category-specific data emphasis

## Forbidden changes
- base architecture
- conversion logic
- prompt template redesign in the same experiment

## Expected effect
- localized robustness gains
- improved low-confidence subset performance
- efficient finishing gains without full redesign

## Difficulty
4. S4 low minlogprob サンプルの再重点学習（推奨度：★★★★☆）
- 理由: 既存基盤を活かしやすく、仕上げ施策として非常に優秀です。

## Coding notes
- Reweight only a clearly defined subset from S2 outputs.
- Keep holdout leakage prevention strict.
- Track whether gains remain localized or generalize.
- Cap replay ratio to avoid catastrophic overfit.

## Fast failure criteria
Fail this strategy if:
- held-out target subset does not improve
- easy categories regress noticeably
- training appears to memorize diagnostic samples without broader benefit

## Feedback to next strategy
S4 must produce:
- which failure modes respond well to reweighting
- which brittle cases require template redesign instead
- replay ratio guidance for future runs

---

# S5. Training-Serving Misalignment Quantification and Repair

## Definition
Measure and reduce the performance gap between the training-side adapter behavior and the submission-side converted adapter behavior. The improvement analysis highlights adapter conversion, SVD loss, post-finetune, and validation strengthening as the core of this layer. fileciteturn3file4

## Target layer
- adapter conversion
- surgery scripts
- SVD compression path
- key mapping / merge logic
- pre/post-conversion evaluation

## Allowed changes
- conversion scripts
- merge order
- rank policy
- validation around conversion outputs

## Forbidden changes
- training dataset in the same experiment
- prompt templates in the same experiment
- category weighting in the same experiment

## Expected effect
- recover performance already learned but lost at submission time
- shrink pre/post-conversion gap
- stabilize easy and symbolic categories if conversion damage is the root cause

## Difficulty
5. S5 training-serving misalignment の定量化と修正（推奨度：★★★★★）
- 理由: 学習時にできていることを提出時に失っているなら、最も効率の良い回収ポイントです。

## Coding notes
- Always evaluate both pre-conversion and post-conversion with the same sample IDs.
- Log category-wise and brittle-sample-wise deltas.
- Treat conversion code as protected infrastructure; patch carefully and version every change.
- Do not mix conversion repair with data or prompt changes.
- Preserve required submission assets and paths. Current preserved baseline asset expectations are documented in Day1 evidence. fileciteturn3file5

## Fast failure criteria
Fail this strategy if:
- pre/post-conversion gap is not measured directly
- a conversion patch changes multiple transformation steps at once
- submission asset reproducibility breaks

## Feedback to next strategy
S5 must output:
- the residual gap after repair
- whether residual loss is conversion-driven or content-driven
- whether S8 post-finetune is warranted

---

# S6. Deterministic CoT Redesign

## Definition
Redesign traces so that under temperature 0 the desired token path is as unique and as stable as possible. The improvement analysis states this is the core idea: simplify reasoning into stable operation sequences, make tokenization-aware expressions, and cover rare operations explicitly. fileciteturn3file4

## Target layer
- trace generation templates
- category-specific reasoning exemplars
- supervised training data construction

## Allowed changes
- CoT wording
- step decomposition style
- ordering of operations
- rare-operation coverage patterns

## Forbidden changes
- conversion logic in the same experiment
- dataset reweighting in the same experiment
- bit-specific logic unless the experiment is explicitly S7

## Expected effect
- lower variance under greedy decoding
- fewer tokenization-induced drifts
- better target-category performance without relying on RL

## Difficulty
6. S6 決定論的CoTの再設計（推奨度：★★★★★）
- 理由: この投稿そのものの中心施策です。難しいですが、競争力の源泉です。

## Coding notes
- Each step should do one semantic action only.
- Prefer expressions that tokenize cleanly and consistently.
- For every template, define the forbidden alternative phrasings.
- Encode rare operations explicitly; do not assume the model will improvise them reliably.
- Design per category, not one universal trace style.

## Fast failure criteria
Fail this strategy if:
- the new templates increase verbosity without reducing drift
- low-confidence segments do not improve on target categories
- answer extraction or formatting gets worse

## Feedback to next strategy
S6 must produce:
- category-specific stable phrasing list
- forbidden unstable phrasing list
- tokenization-sensitive fragments reused by S7 and S9

---

# S7. Bit Manipulation Specialized Strengthening

## Definition
Create a category-specific redesign for bit manipulation tasks, since this is identified as a major score differentiator and one of the largest remaining upside areas. fileciteturn3file3turn3file4

## Target layer
- bit-category CoT templates
- bit-category training slice
- bit-specific error taxonomy

## Allowed changes
- rule-discovery procedure
- comparison order
- symbol conventions
- rare bit-pattern training coverage

## Forbidden changes
- easy-category templates in the same experiment
- conversion logic in the same experiment
- generic all-category CoT changes in the same experiment

## Expected effect
- measurable gains on bit manipulation
- reduced rule-detection failures
- better stability on whitespace / y-like fragile outputs if those are present in task formatting

## Difficulty
7. S7 bit manipulation 専用CoT強化（推奨度：★★★★★）
- 理由: 差が最もつく領域です。難しいですが、上位を狙うなら避けにくいです。

## Coding notes
- Fix a deterministic comparison order and never vary it.
- Use a minimal symbol inventory.
- Explicitly define how to describe rule candidates and how to eliminate them.
- Keep a dedicated bit failure manifest separate from general symbolic failures.
- Start from S2 token-fragility evidence and S1 brittle cases.

## Fast failure criteria
Fail this strategy if:
- bit gains are tiny and inconsistent
- improvements depend on a tiny hand-picked subset only
- easy or formatting categories regress because of shared template contamination

## Feedback to next strategy
S7 must produce:
- bit-specific stable notation contract
- high-value rare pattern list
- evidence on whether bit should remain mainline priority

---

# S8. Post-Finetune After Conversion

## Definition
Fine-tune again on the converted adapter form to compensate residual performance lost after compression or conversion. The improvement analysis notes this as a strong but heavier follow-up once misalignment is measured. fileciteturn3file3turn3file4

## Target layer
- converted adapter training pipeline
- residual-loss subset selection

## Allowed changes
- post-conversion fine-tune config
- residual-focused training subset

## Forbidden changes
- changing the conversion pipeline in the same experiment
- category-wide prompt redesign in the same experiment

## Expected effect
- recover residual conversion loss not fixed by surgery alone

## Difficulty
8. S8 変換後adapterへの post-finetune（推奨度：★★★★☆）
- 理由: misalignment が大きいならかなり有効です。

## Coding notes
- Only start after S5 shows a residual, stable post-conversion gap.
- Do not use S8 to hide unknown conversion bugs.
- Train against the post-conversion target failure set, not the whole world.

## Fast failure criteria
Fail this strategy if:
- S5 has not cleanly isolated a residual conversion gap
- post-finetune gains disappear on Promotion set
- converted artifact reproducibility degrades

---

# S9. Token-Direct Generation Pipeline

## Definition
Move from text-first trace construction to token-direct construction so tokenization-induced mismatch is reduced at the source. The improvement analysis treats this as promising but infrastructure-heavy. fileciteturn3file4

## Target layer
- data generation pipeline
- tokenizer-aware trace construction
- dataset serialization format

## Allowed changes
- token-level dataset builder
- token-level validation utilities

## Forbidden changes
- do not combine with broad CoT redesign in the first experiment
- do not combine with synthetic generation in the first experiment

## Expected effect
- reduced tokenization traps
- stronger control over brittle symbol sequences

## Difficulty
9. S9 token 直接生成ベースへの移行（推奨度：★★★★☆）
- 理由: 将来性は高いですが、短期の勝負では重いです。

## Coding notes
- Create a small pilot first.
- Validate exact tokenizer round-trip behavior.
- Keep an equivalent text rendering for debugging.

## Fast failure criteria
Fail this strategy if:
- debugging becomes slower than the expected benefit
- token-level control does not reduce the intended failure class

---

# S10. Synthetic Problem Expansion

## Definition
Use generated problems to expand training or validation coverage for rare patterns, with careful quality control. The improvement analysis treats this as potentially useful but risky if generator quality is weak. fileciteturn3file2turn3file3

## Target layer
- data generation scripts
- validation pipeline for generated items

## Allowed changes
- synthetic task generators
- synthetic validation subsets

## Forbidden changes
- mixing synthetic and real held-out without labeling
- using synthetic-only gains as promotion evidence

## Expected effect
- better rare-pattern coverage
- more scalable stress testing

## Difficulty
10. S10 synthetic problem 生成による拡張学習（推奨度：★★★☆☆）
- 理由: 可能性はありますが、まずは既存データの活かし方を詰めた方が効率的です。

## Coding notes
- Every synthetic sample must record generator provenance.
- Never contaminate core Promotion evaluation with unlabeled synthetic items.
- Use synthetic first as a diagnostic amplifier, not as proof of success.

## Fast failure criteria
Fail this strategy if:
- generated samples are noisy or inconsistent
- synthetic gains do not transfer to real held-out

---

# S11. Cryptarithm Breakthrough Research

## Definition
Research a new path for cryptarithm-like bottlenecks that likely require new reasoning structure rather than cleanup. The improvement analysis marks this as high-upside but research-heavy. fileciteturn3file2turn3file3

## Target layer
- research prototypes
- dedicated category datasets
- specialized trace or solver ideas

## Allowed changes
- isolated prototype branches only

## Forbidden changes
- do not consume core production sprint budget until Tier A and Tier B are stable

## Expected effect
- long-term ceiling lift

## Difficulty
11. S11 cryptarithm の本格突破（推奨度：★★★☆☆）
- 理由: 当たれば大きいですが、短期改善策としては重すぎます。

## Coding notes
- This is a research track, not a Day2/Day3 fast-cycle track.
- Keep separate from the mainline repo path if needed.

## Fast failure criteria
Fail this strategy for mainline use if:
- it cannot be judged quickly
- it steals budget from higher-probability strategies

---

# Cross-Strategy Dependency Rules

## Allowed early-cycle sequence
S1 -> S2 -> S3/S5 -> S4 -> S6 -> S7 -> S8

## Default next-step logic
- If judgment is unreliable: go back to S1.
- If weak spots are unclear: do S2.
- If easy losses exist: do S3.
- If low-confidence subset is concentrated: do S4.
- If training and submission mismatch exists: do S5.
- If model behavior is still too free-form: do S6.
- If bit remains top bottleneck: do S7.
- If residual post-conversion gap remains after S5: do S8.

---

# Required Deliverable for Every Strategy Ticket
Each implementation ticket must contain this exact block:
- Strategy ID
- Definition
- Target layer
- Allowed changes
- Forbidden changes
- Expected effect
- Difficulty
- Coding notes
- Fast failure criteria
- Required outputs
- Feedback expected for next strategy

If any item is missing, the ticket is not implementation-ready.
