# Day2 Codex Request

あなたは NVIDIA Nemotron Model Reasoning Challenge プロジェクトの Day2 Generator である。

## Role
Day1 で固定したベースAを比較基準として維持しながら、**B1: training-serving misalignment 修正** を 1変数だけ投入し、比較可能な成果物を作成することが役割である。

## Fixed baseline
- Repository: `hinemos-anzu/Nemotron`
- Branch: `planner/day1-logprob`
- Baseline SHA: `39f4bed90392567517b606d1301ae1c36a86a97c`

## Day2 objective
今回の目的は1つだけ。  
**B1: training-serving misalignment 修正を 1変数だけ投入し、A固定基準と比較可能な状態を作ること。**

## Must read first
1. `control/decision_log.md`
2. `control/day2_sprint_contract.md`
3. `control/day2_generator_instructions.md`
4. `control/day2_evaluator_instructions.md`
5. `reports/day1/reproduction_baseline.md`
6. `reports/day1/submission_asset_verification.md`

## Fixed rules
1. B1 以外の施策を入れない
2. 1実験1変数を厳守する
3. Baseline SHA を変更しない
4. 提出資産部を壊さない
5. Kaggle 正本で記録する
6. 悪化した場合は隠さない
7. 曖昧語を使わない
8. silent fallback をしない

## In scope
1. B1 のみを反映した変更
2. A固定基準との差分整理
3. Kaggle 上での実行
4. submission.zip 成立確認
5. Day2 成果物作成
6. GitHub 正本への push

## Out of scope
1. B2 以降
2. 多変数変更
3. A基準の再編集
4. router / 多数決 / SymPy 改善
5. 無関係な提出資産ロジック改変

## Required outputs
1. `reports/day2/change_manifest.md`
2. `reports/day2/experiment_result.md`
3. `reports/day2/submission_asset_check.md`
4. `reports/day2/adoption_decision.md`

## Required output rules

### reports/day2/change_manifest.md
- baseline SHA
- modified SHA
- 変更変数は1つだけ、という明示
- 変更内容要約
- 変更していない施策一覧

### reports/day2/experiment_result.md
- 実行環境 = Kaggle
- 比較対象 = baseline SHA 39f4bed9
- 実行条件
- 観測結果
- A比で悪化したか否か
- 0.86超えに向かう根拠の有無

### reports/day2/submission_asset_check.md
- 提出資産部が壊れていないか
- submission.zip の生成可否
- path
- size
- file list
- 判定: PASS / FAIL / BLOCKED

### reports/day2/adoption_decision.md
- 最終判定: ADOPT / REJECT / HOLD
- 判定理由
- A比で悪化していないか
- 次に進む条件

## GitHub reflection
- 必ず `planner/day1-logprob` に push する
- local only で終えない
- remote HEAD SHA を最終報告に入れる

## Final output format
1. Branch confirmation
2. Updated files
3. Change summary
4. Git operation summary
5. Final status: READY FOR EVALUATION / NOT READY
