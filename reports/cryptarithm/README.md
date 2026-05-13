# Cryptarithm Diagnostics Workflow

This directory keeps human-authored plans and catalogs for cryptarithm diagnostics. Generated CSV/JSONL/Markdown outputs are intentionally not committed because they depend on the local challenge data snapshot.

## Recommended execution

```bash
python scripts/cryptarithm_inventory.py \
  --inputs data/raw/problems.jsonl data/raw/train.csv data/raw/corpus.jsonl \
  --corpus data/raw/corpus.jsonl \
  --output reports/cryptarithm/cryptarithm_problem_inventory.csv

python scripts/cryptarithm_solver.py \
  --inventory reports/cryptarithm/cryptarithm_problem_inventory.csv \
  --output reports/cryptarithm/cryptarithm_solver_coverage.csv

python scripts/cryptarithm_generate_verified_cot.py \
  --coverage reports/cryptarithm/cryptarithm_solver_coverage.csv \
  --output reports/cryptarithm/cryptarithm_generated_cot_sample.jsonl \
  --report reports/cryptarithm/cryptarithm_failure_report.md

python scripts/cryptarithm_build_corpus_patch.py \
  --input reports/cryptarithm/cryptarithm_generated_cot_sample.jsonl \
  --output reports/cryptarithm/cryptarithm_corpus_patch.jsonl \
  --strict

python scripts/cryptarithm_validate_corpus_patch.py \
  --patch reports/cryptarithm/cryptarithm_corpus_patch.jsonl \
  --require-rows
```

Only rows marked `verified=true` by the solver should be promoted to a corpus patch. The patch builder skips unsafe rows by default and fails on them when `--strict` is set. Missing-example fallback rows are diagnostic failures and are not eligible for CoT generation.

## Do not do yet

- Do not train LoRA adapters from unverified cryptarithm data.
- Do not change rank maps, target modules, SVD settings, adapter fusion, or `submission.zip` in the diagnostic phase.
- Do not commit local generated reports unless they correspond to a documented, reproducible data snapshot.
