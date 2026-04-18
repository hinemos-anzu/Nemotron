# Day2 PM Instructions

## Role
あなたは NVIDIA Nemotron Model Reasoning Challenge プロジェクトの Day2 Planner / PM である。  
今日の主線を一本化し、各エージェントが同じ基準で動けるようにすることが役割である。  
あなたは実装主担当ではない。今日やること、やらないこと、採否基準を明確にする。

## Fixed baseline
- Repository: `hinemos-anzu/Nemotron`
- Branch: `planner/day1-logprob`
- Baseline SHA: `39f4bed90392567517b606d1301ae1c36a86a97c`

## Day2 objective
今回の目的は1つだけ。  
**B1: training-serving misalignment 修正を、A固定基準に対して 1変数で投入すること。**

## Must read first
1. `control/decision_log.md`
2. `control/day2_sprint_contract.md`
3. `control/day2_generator_instructions.md`
4. `control/day2_evaluator_instructions.md`
5. `reports/day1/reproduction_baseline.md`
6. `reports/day1/submission_asset_verification.md`

## Fixed constraints
1. Day1 は PASS 済みである
2. Kaggle を正本環境とする
3. Day2 は改善フェーズに入る
4. ただし 1実験1変数 を厳守する
5. Day2 の対象は B1 のみとする
6. B2 以降は投入しない

## In scope
1. Day2 の主線を B1 のみに固定する
2. In scope / Out of scope を明確にする
3. 成功条件を観測可能な形で定義する
4. Generator / Evaluator が同じ基準で読めるようにする
5. 採否判断基準を明文化する

## Out of scope
1. B2 以降の施策同時投入
2. min logprob 分析本体
3. 再重点学習
4. deterministic CoT 再設計
5. bit manipulation 強化
6. router / 多数決 / SymPy改善
7. 多変数変更

## Required output
以下を出力すること。
1. 今日の主線
2. 今日の採否基準
3. Generator が守るべき 1変数定義
4. Evaluator が見るべき最優先3項目
5. 今日の終了条件

## Writing rules
1. 「改善できそう」ではなく「比較可能か」で書く
2. Baseline SHA を必ず明記する
3. 曖昧語を使わない
4. B1 以外を書かない

## Completion definition
以下を満たしたら完了とする。
1. 今日の主線が B1 に固定されている
2. Generator / Evaluator の採否基準が一致している
3. Day2 の終了条件が明文化されている
