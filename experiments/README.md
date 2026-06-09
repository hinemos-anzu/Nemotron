# Experiments

This directory records leaderboard-facing experiment decisions, rollback notes, and control-run plans.

## Current control sequence

1. `b3_gate15_x17/` records the rejected asymmetric in-projection split candidate.
2. `B3_GOLDEN_EXACT_REGEN/` is the next control experiment and must reproduce the Golden Baseline with no intentional model, rank, data, conversion, generation, or packaging changes.

## Current policy

Rank / SVD / weight-surgery exploration is paused until the Golden exact control is reconfirmed. The next improvement family after that control should keep the Golden adapter structure fixed and use one-variable cryptarithm-focused SFT experiments.
