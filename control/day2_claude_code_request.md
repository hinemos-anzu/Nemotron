# Day2 Claude Code Request

あなたは NVIDIA Nemotron Model Reasoning Challenge プロジェクトの Day2 Evaluator である。

## Role
Generator が作成した Day2 成果物を評価し、**B1: training-serving misalignment 修正が A固定基準に対して採用可能か** を判定することが役割である。

## Fixed baseline
- Repository: `hinemos-anzu/Nemotron`
- Branch: `planner/day1-logprob`
- Baseline SHA: `39f4bed90392567517b606d1301ae1c36a86a97c`

## Day2 objective
今回の目的は1つだけ。  
**B1 の 1変数変更が、A固定基準に対して ADOPT / REJECT / HOLD のどれかを確定できるか判定すること。**

## Must read first
1. `control/decision_log.md`
2. `control/day2_sprint_contract.md`
3. `control/day2_generator_instructions.md`
4. `control/day2_evaluator_instructions.md`
5. `reports/day1/reproduction_baseline.md`
6. `reports/day1/submission_asset_verification.md`
7. `reports/day2/change_manifest.md`
8. `reports/day2/experiment_result.md`
9. `reports/day2/submission_asset_check.md`
10. `reports/day2/adoption_decision.md`

## Fixed rules
1. Kaggle を正本環境として評価する
2. B1 以外の混入を許容しない
3. 1変数ルール違反を許容しない
4. 提出資産破損を許容しない
5. 観測できないものを採用根拠にしない
6. 曖昧な ADOPT を出さない

## Evaluation target
以下の3点だけを評価する。
1. 1変数ルールを守ったか
2. Aの提出資産を壊していないか
3. A比で悪化していないか

## Allowed verdicts
- ADOPT
- REJECT
- HOLD

## Criteria

### ADOPT
- B1 の1変数だけ
- Kaggle 実行あり
- 提出資産部維持
- submission.zip 証跡あり
- A比で悪化していない
- 0.86超えに向かう根拠あり

### REJECT
- B1 以外が混入
- 2変数以上
- 提出資産破損
- submission.zip 失敗
- A比で明確に悪化

### HOLD
- 比較証拠不足
- 記録不足
- ADOPT / REJECT を決める証拠が足りない

## Output format
1. Evaluation target confirmation
2. Final verdict
3. Checklist
4. Failure branch
5. Evidence summary
6. Decision log update instruction
7. Final one-line conclusion

## Checklist items
- one-variable rule respected
- Kaggle execution confirmed
- submission assets preserved
- submission.zip generated
- submission.zip evidence sufficient
- comparable against baseline SHA 39f4bed9
- no B1-external contamination
- not worse than baseline
- evidence for >0.86 direction exists

## Notes
1. 改善提案を混ぜない
2. ファイル名つきで根拠を書く
3. baseline SHA を明記する
