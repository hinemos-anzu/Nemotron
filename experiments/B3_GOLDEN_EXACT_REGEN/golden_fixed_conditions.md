# Golden Fixed Conditions

`B3_GOLDEN_EXACT_REGEN` is valid only if it keeps the Golden Baseline unchanged. The experiment changes **nothing** intentionally; it only checks whether the current workflow can reproduce the known `0.86` Public LB control.

| Condition | Required value | Allowed to change? | Notes |
|---|---|---:|---|
| Training data | Golden Baseline data | No | Do not add SFT rows, cryptarithm rows, format rows, or any synthetic patch. |
| Adapter rank map | Golden Baseline rank map | No | Do not tune per-module ranks or redistribute rank budget. |
| `target_modules` | Golden Baseline target modules | No | Adapter module coverage must match Golden. |
| Conversion script | Golden Baseline conversion path | No | Do not introduce fp64-only SVD or new conversion heuristics. |
| Gate/X split | `gate=16 / x=16` | No | No `15/17`, `17/15`, `14/18`, `18/14`, or other asymmetric split. |
| Generation config | Golden Baseline generation config | No | Keep deterministic Golden config, including `temperature=0.0` and `do_sample=False`. |
| Root packaging | Golden Baseline root `submission.zip` packaging | No | Archive contents and root layout must match the accepted Golden packaging convention. |

## Generation config source of truth

The Golden inference scripts define the fixed generation configuration as:

```text
max_new_tokens: 2048
temperature: 0.0
do_sample: False
repetition_penalty: 1.0
stop: ["<|endoftext|>", "<|im_end|>"]
```

Do not change prompt formatting, decoding parameters, answer extraction, stop tokens, or seed as part of this control.

## Structural source of truth

Before promotion, confirm that the regenerated candidate reports Golden-compatible structural diagnostics:

- Gate/X split is `16/16`.
- Rank violations are absent.
- NaN/Inf values are absent.
- Tensor count is Golden-compatible.
- `max_rank_seen` is Golden-compatible.
- Adapter keys and module names are consistent with Golden `target_modules` and rank map.
- The root archive contains exactly the expected submission payload layout when artifact generation is explicitly approved in a later phase.
