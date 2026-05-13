# Cryptarithm A/B Plan

The first iteration intentionally stops at diagnosis, solver coverage, and solver-verified CoT generation. It does not alter adapters, ranks, target modules, SVD settings, fusion, or submission packaging.

| Experiment | Change | Purpose | Proceed criteria |
|---|---|---|---|
| C0 | Re-submit or re-evaluate the current baseline artifact unchanged. | Confirm the measured LB/reference baseline is reproducible. | Adapter SHA, zip SHA, rank map, target modules, and key counts are recorded. |
| C1 | Add only `cryptarithm_deduce` verified CoT samples. | Test the safest rule-inference improvement path. | Deduce coverage and local category metrics improve without broad category regression. |
| C2 | Add only `cryptarithm_guess` verified CoT samples. | Measure whether candidate-enumeration teaching helps. | Guess improves without overfitting or answer-format regressions. |
| C3 | Add deduce + guess verified samples. | Measure combined impact. | Combined cryptarithm lift exceeds either single-source patch. |
| C4 | Add deduce + guess + hard negatives. | Reduce default-concat overuse and wrong-rule generalization. | Hard negatives reduce mismatch types without lowering solved examples. |

## Initial training guardrails for later work

- Keep LoRA rank and target modules identical to the baseline configuration.
- Prefer a lower learning rate range such as `5e-5` to `1e-4` for the cryptarithm patch.
- Keep the existing corpus in the mix; start with roughly 80–90% existing corpus and 10–20% cryptarithm strengthening data.
- Use one-variable experiments only.
- Never include unverified LLM-only CoT in the patch.
