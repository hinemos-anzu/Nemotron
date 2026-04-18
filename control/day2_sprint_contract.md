# Sprint title
Day2 Sprint Contract — 改善策Bの第1変数投入

# Goal
Day2 の目的は **改善策Bの最初の1変数を、A固定基準（SHA: 39f4bed9）に対して比較可能な形で投入すること** のみである。  
Day2 では多変数変更をしない。  
Aの提出資産を壊さず、A比で悪化していないかを判定できる形で、最初の改善施策を1本だけ検証する。

# Day2 で採用する最初の施策
Day2 では、改善策Bのうち最初の投入対象を以下に固定する。

**B1: training-serving misalignment 修正**

# In scope
1. A固定基準（SHA: 39f4bed9）からの差分を 1変数に限定して実装する
2. Day2 の変更対象を **training-serving misalignment 修正のみ** に限定する
3. A比での比較に必要な実験記録を残す
4. 提出資産部が壊れていないかを確認する
5. 悪化していないかを確認する
6. 0.86超えに向かう根拠の有無を整理する

# Out of scope
1. min logprob 分析本体
2. low-minlogprob 再重点学習
3. deterministic chain-of-thought 再設計
4. bit manipulation 強化
5. router 改善
6. 多数決改善
7. SymPy改善
8. 2変数以上の同時変更
9. A固定基準そのものの変更
10. 提出資産生成ロジックの無関係な改変

# Fixed baseline
Day2 の比較基準は以下に固定する。
- Repository: hinemos-anzu/Nemotron
- Branch: planner/day1-logprob
- Baseline SHA: `39f4bed90392567517b606d1301ae1c36a86a97c`

以後の比較は必ずこの SHA を A基準とする。

# Inputs
1. Day1 正式成果物一式
2. A固定基準 SHA: 39f4bed9
3. Kaggle 正本環境
4. training-serving misalignment 修正案

# Required outputs
1. `reports/day2/change_manifest.md`
2. `reports/day2/experiment_result.md`
3. `reports/day2/submission_asset_check.md`
4. `reports/day2/adoption_decision.md`

# Required output rules

## 1. reports/day2/change_manifest.md
必ず以下を含めること。
- baseline SHA
- modified SHA
- 今回変更した変数は 1つだけである、という明示文
- 変更内容の要約
- 変更していない施策一覧

## 2. reports/day2/experiment_result.md
必ず以下を含めること。
- 実行環境 = Kaggle
- 比較対象 = A固定基準
- 実行条件
- 観測結果
- A比で悪化したか否か
- 0.86超えに向かう根拠の有無

## 3. reports/day2/submission_asset_check.md
必ず以下を含めること。
- Aの提出資産部が壊れていないか
- submission.zip 生成可否
- 出力パス
- ファイルサイズ
- zip内ファイル一覧
- 判定: PASS / FAIL / BLOCKED

## 4. reports/day2/adoption_decision.md
必ず以下を含めること。
- 最終判定: ADOPT / REJECT / HOLD
- 判定理由
- A比で悪化していないか
- 次に進む条件

# Success criteria
以下をすべて満たした場合のみ Day2 成功と判定する。

| ID | 観測対象 | 合格条件 |
|---|---|---|
| SC-1 | 変更変数数 | 変更が training-serving misalignment 修正の 1変数だけである |
| SC-2 | 正本環境 | Kaggle 上で実行されている |
| SC-3 | 提出資産 | Aの提出資産部を壊していない |
| SC-4 | submission.zip | 生成でき、出力パス・サイズ・zip内一覧が記録されている |
| SC-5 | 比較可能性 | A固定基準（SHA: 39f4bed9）との比較結果が明記されている |
| SC-6 | 悪化回避 | A比で悪化していない、または悪化時は明確に REJECT 判定されている |
| SC-7 | 根拠 | 0.86超えに向かう根拠の有無が記録されている |

# Failure branches

## FB-1: asset broken
条件:
- 提出資産部が壊れた
- submission.zip が生成できない
- zip構造が壊れている

判定:
- FAIL

## FB-2: multi-variable contamination
条件:
- B1 以外の施策が混入した
- 2変数以上が同時に変更された

判定:
- FAIL

## FB-3: no comparable evidence
条件:
- A固定基準との比較ができない
- 記録不足で ADOPT / REJECT / HOLD が判定不能

判定:
- BLOCKED

# Stop conditions
以下のいずれかに達した時点で Day2 を終了する。
1. Success criteria を全件満たした
2. Failure branch のいずれかに確定分類された
3. B1 以外の施策を追加しないと進められないと判明した

# Notes for Generator
1. Day2 では B1 以外を入れない
2. 1実験1変数を厳守する
3. A固定基準 SHA を必ず明記する
4. 提出資産部を壊さないことを優先する
5. A比で悪化した場合は隠さず記録する
6. Day2 は「改善の着手日」であり、「多施策投入日」ではない

# Notes for Evaluator
1. Day2 の評価対象は以下の3点のみ
   - 1変数ルールを守ったか
   - Aの提出資産を壊していないか
   - A比で悪化していないか
2. ADOPT / REJECT / HOLD を必ず返す
3. B1 以外の混入があれば即 REJECT とする
4. 比較不能なら HOLD ではなく BLOCKED 相当として扱う
