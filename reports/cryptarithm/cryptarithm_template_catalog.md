# Cryptarithm Template Catalog

This catalog defines the first solver-backed rule families to diagnose before any training run.

| rule_type | Description | Example transform |
|---|---|---|
| `forward_concat` | Concatenate left then right. | `AB ? CD -> ABCD` |
| `reverse_concat` | Concatenate right then left; this is the canonical label for side-swap/right-left output. | `AB ? CD -> CDAB` |
| `reverse_left` | Reverse the left token, then append the right token. | `AB ? CD -> BACD` |
| `reverse_right` | Append the reversed right token after the left token. | `AB ? CD -> ABDC` |
| `reverse_both` | Reverse each token independently, preserving left-right side order. | `AB ? CD -> BADC` |
| `interleave_lr` | Interleave characters starting from the left token. | `AB ? CD -> ACBD` |
| `interleave_rl` | Interleave characters starting from the right token. | `AB ? CD -> CADB` |
| `operator_conditioned_rule` | Infer a rule from examples with the same operator as the question when one global rule does not fit all examples. | `@` may be forward concat while `#` may be reverse concat. |
| `unknown_operator_fallback` | Diagnostic label for missing examples or unsupported operators; it is not promoted to verified CoT. | n/a |

## Masking recommendation

Generated examples should keep the loss focused on:

1. the selected rule decision,
2. the final application of the rule to the question, and
3. the boxed answer.

Avoid training on unverified free-form reasoning. Only samples marked `verified: true` by the solver should be promoted into a corpus patch.
