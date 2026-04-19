# Category Evaluation Criteria Spec v1.0

## Purpose
This document defines what must be checked for each category during Quick Gate, Diagnostic review, and Promotion evaluation.
The goal is not only to measure accuracy, but also to identify the shortest path to the next better experiment.

## Common Metrics for All Categories
Every category review must report:
- sample count
- accuracy
- delta vs baseline
- format failure count
- answer extraction failure count
- notable failure modes
- average evaluation time if category-specific latency matters

## Category Definitions and Review Focus

### 1. Easy / Deterministic Tasks
Examples: numeral, unit conversion, direct arithmetic, direct mapping, gravity-like template tasks, basic cipher templates.

**Primary concern**
- These should behave like score-protection categories.

**What to check**
- wrong final answer
- avoidable formatting error
- unnecessary reasoning drift
- unstable output on tasks that should be deterministic

**Decision bias**
- Regression tolerance should be near zero.
- Any experiment that improves hard categories but damages this group significantly should usually be rejected.

### 2. Equation / Symbolic Pattern Tasks
Examples: equation solving, symbolic manipulation, algebraic operator identification.

**Primary concern**
- operator confusion
- sign mistakes
- brittle symbolic formatting
- degradation under conversion or compression

**What to check**
- correct symbolic interpretation
- stable final answer extraction
- whether errors cluster around specific operator types
- whether deterministic CoT changes reduce symbolic drift

### 3. Bit Manipulation Tasks
Examples: bit comparison, transformation rule detection, binary/string bit pattern tasks.

**Primary concern**
- this is a likely score-differentiating category, but it is also expensive to improve

**What to check**
- rule-detection failure
- positional mismatch
- tokenization-sensitive output breakdown
- false confidence on wrong transformation

**Decision bias**
- meaningful isolated gains are valuable, but only if they do not trigger broad regressions elsewhere.

### 4. Formatting-Sensitive Tasks
Examples: tasks where the model mostly knows the answer but misses the accepted output style.

**Primary concern**
- answer extraction failure
- boxed/final-line/template mismatch
- extra text that corrupts accepted output

**What to check**
- exact format compliance
- extraction robustness
- whether a change improves answer quality but worsens deliverability

### 5. Conversion / Alignment-Sensitive Tasks
Examples: samples that are correct before conversion but degrade after adapter conversion, surgery, SVD, or serving transformation.

**Primary concern**
- training-serving misalignment
- weight conversion loss
- architecture-specific surgery errors

**What to check**
- pre-conversion accuracy
- post-conversion accuracy
- delta by category and by brittle sample
- whether the regression is concentrated in a known module family

### 6. Low-Min-Logprob Suspect Tasks
Examples: samples identified as having brittle low-confidence tokens or unstable trace segments.

**Primary concern**
- one-token collapse or localized instability

**What to check**
- whether the target weak token/segment improved
- whether reweighting helped the intended subset
- whether gains are localized or generalizable

### 7. Hard / Research-Like Tasks
Examples: cryptarithm-like tasks or categories known to require new insight rather than simple cleanup.

**Primary concern**
- avoid wasting early-cycle budget on expensive but slow-to-judge ideas

**What to check**
- only run after the evaluation OS is stable
- track separately from quick-win categories
- require explicit Planner approval before consuming Kaggle budget

## Failure Mode Taxonomy
Each failed sample should be tagged with one primary label and, if needed, one secondary label.

Primary labels:
- FORMAT_FAILURE
- EXTRACTION_FAILURE
- EASY_TASK_REGRESSION
- OPERATOR_CONFUSION
- BIT_RULE_FAILURE
- POSITIONAL_MISMATCH
- LOW_LOGPROB_COLLAPSE
- PREPOST_CONVERSION_REGRESSION
- OVERFITTING_TO_DIAGNOSTIC
- UNKNOWN

## Category-Level Decision Rules

### Keep / promote signals
- targeted category improves without harming easy / formatting categories
- failure modes become more concentrated and more explainable
- pre/post conversion gap shrinks for alignment experiments
- brittle samples become stable

### Reject / rollback signals
- easy-task or formatting regressions appear
- gains are visible only on a tiny cherry-picked subset
- target category improves but conversion gap worsens sharply
- diagnostic failure modes become more scattered and less explainable

## Required Reporting Block
Every experiment report must include a category table with:
- category
- baseline accuracy
- candidate accuracy
- delta
- key failure mode changes
- promotion judgment

## Planner Use
Planner should use this file to:
- decide whether a gain is strategically useful
- reject wins that come from the wrong tradeoff
- choose the next experiment based on the new dominant failure mode
