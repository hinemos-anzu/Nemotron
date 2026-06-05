# Claude Code Guidelines

1. Ask, don't assume — if something is unclear, ask before writing a single line. Never make silent assumptions.
2. Simplest solution first — always implement the simplest thing that could work. Don't add abstractions that weren't requested.
3. Don't touch unrelated code — if a file is not directly part of the current task, do not modify it. Ever.
4. Flag uncertainty explicitly — if you're not confident about an approach, say so before proceeding. Confidence without certainty causes more damage than admitting a gap.
5. Session start: user message overrides summary — if the user's opening message explicitly states a purpose, prohibitions, or task list, that is the sole current task definition. Never continue a carry-over task from the session summary without explicit confirmation from the user.
6. Existing working code is untouchable — if the user states they already have working code for a task, do not re-implement, modify, or create alternatives to it. Read and analyse only.
