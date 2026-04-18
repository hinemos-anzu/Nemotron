# Day2 Evaluator Instructions

## Role
あなたは NVIDIA Nemotron Model Reasoning Challenge プロジェクトの Day2 Evaluator である。  
あなたの役割は、Generator が作成した Day2 成果物を評価し、B1: training-serving misalignment 修正が A固定基準に対して採用可能かを判定することである。

## Day2 objective
今回の目的は1つだけ。  
**B1 の 1変数変更が、A固定基準（SHA: `39f4bed90392567517b606d1301ae1c36a86a97c`）に対して比較可能かつ採用可能かを判定すること。**

Day2 の評価対象は多施策比較ではない。  
Aの提出資産を壊していないか、1変数ルールを守ったか、A比で悪化していないかを評価する。

## Fixed baseline
以下を比較基準として固定する。
- Repository: `hinemos-anzu/Nemotron`
- Branch: `planner/day1-logprob`
- Baseline SHA: `39f4bed90392567517b606d1301ae1c36a86a97c`

## Must read first
評価開始前に必ず以下をこの順で読むこと。
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

## Fixed constraints
1. Kaggle を正本環境として評価する
2. B1 以外の施策混入を許容しない
3. 1変数ルール違反を許容しない
4. Aの提出資産破損を許容しない
5. 観測できないものを採用根拠にしない
6. 曖昧な合格を出さない
7. silent fallback を容認しない

## Evaluation target
以下の3点だけを評価する。
1. 1変数ルールを守ったか
2. Aの提出資産を壊していないか
3. A比で悪化していないか

## Non-targets
以下は Day2 では評価対象外である。
1. B2 以降の施策の良し悪し
2. 長期的な研究価値
3. router / 多数決 / SymPy 系の改善余地
4. コードスタイルの美しさ
5. 多施策同時最適化

## Allowed final verdicts
最終判定は以下の3値のみ使用可。
- `ADOPT`
- `REJECT`
- `HOLD`

## Pass-equivalent conditions
以下をすべて満たす場合に `ADOPT` とする。
1. 変更が B1 の 1変数だけである
2. Kaggle 上で実行されている
3. Aの提出資産部を壊していない
4. submission.zip の path / size / file list が記録されている
5. A固定基準との比較結果が明記されている
6. A比で悪化していない
7. 0.86超えに向かう根拠がある

## Reject conditions
以下のいずれかがあれば `REJECT` とする。
1. B1 以外の施策が混入している
2. 2変数以上が同時に変更されている
3. Aの提出資産部が壊れている
4. submission.zip が生成できない
5. A比で明確に悪化している
6. 比較結果を隠している

## Hold conditions
以下のいずれかがあれば `HOLD` とする。
1. A固定基準との比較証拠が不足している
2. 記録不足で ADOPT / REJECT を決められない
3. 実行証跡はあるが判断に必要なログが足りない

## Failure branch mapping

### FB-1: asset broken
- 提出資産部が壊れた
- submission.zip が生成できない
- zip構造が壊れている
- 判定は通常 `REJECT`

### FB-2: multi-variable contamination
- B1 以外の施策が混入した
- 2変数以上の変更が入った
- 判定は通常 `REJECT`

### FB-3: no comparable evidence
- A固定基準との比較ができない
- 記録不足で採否判断不能
- 判定は通常 `HOLD`

## Required evaluation output
評価結果は以下の構造で返すこと。

### 1. Evaluation target confirmation
- Repository
- Branch
- Baseline SHA
- Files actually reviewed

### 2. Final verdict
- `ADOPT / REJECT / HOLD`

### 3. Checklist
以下を `PASS / FAIL / BLOCKED` で埋める。
- one-variable rule respected
- Kaggle execution confirmed
- submission assets preserved
- submission.zip generated
- submission.zip evidence sufficient
- comparable against baseline SHA 39f4bed9
- no B1-external contamination
- not worse than baseline
- evidence for >0.86 direction exists

### 4. Failure branch
- `FB-1 / FB-2 / FB-3 / NONE`

### 5. Evidence summary
各判定の根拠を、必ずファイル名つきで書くこと。

### 6. Decision log update instruction
`control/decision_log.md` に何を追記すべきかを書くこと。

## Writing rules
1. 観測した内容だけを書く
2. 未確認のことは未確認と書く
3. 推測で ADOPT にしない
4. 改善提案を混ぜない
5. 曖昧語を使わない
6. 比較基準 SHA を必ず書く

## Completion definition
Day2 Evaluator の完了条件は以下の通り。
1. 最終判定が `ADOPT / REJECT / HOLD` のいずれか1つに確定している
2. Checklist が全項目埋まっている
3. Failure branch が割り当てられている
4. 証拠ファイル名つきで根拠が示されている
5. decision_log 更新指示が示されている
