# Decision ID
D1

## Hypothesis
Day1 では、ベースAの再現基準と提出資産部を Kaggle 上で固定できれば、以後の改善策Bを A 比で厳密に判定できる。

## Why now
改善策Bを先に入れると、Aの提出資産破損と改善効果を分離できない。  
先に比較基準を固定しない限り、A比で悪化していないかを判定できない。

## Evidence expected
1. `reports/day1/kaggle_path_inventory.md` に以下4項目の解決結果がある
   - adapter path
   - model path
   - tokenizer path
   - input data path
2. `reports/day1/reproduction_baseline.md` に A の再現条件が未確定項目 `0` 件で記録されている
3. `reports/day1/submission_asset_verification.md` に `submission.zip` の生成結果がある
4. ベースAの中核資産6項目の保存状況が個別に確認できる

## Adoption rule
以下を全て満たした場合、この仮説を採用する。
1. Kaggle を正本環境と明記できる
2. 4種類の Kaggle path がすべて `RESOLVED`
3. ベースAの中核資産6項目がすべて `PRESERVED`
4. `submission.zip` の生成結果が確認できる
5. Aの再現条件に未確定項目が残っていない
6. 以後の比較基準として固定可能と Evaluator が判定する

## Rejection rule
以下のいずれか1つでも発生した場合、この仮説を棄却する。
1. 4種類の Kaggle path のうち1つでも `UNRESOLVED`
2. ベースAの中核資産6項目のうち1つでも `BROKEN`
3. `submission.zip` が生成できない、空、または構造不正
4. Aの再現条件に未確定項目が残る
5. 改善策Bが Day1 成果物に混入する

## Follow-up if pass
1. Day1 基準をベースラインAとして固定する
2. Day2 以降で改善策Bを 1実験1変数 で投入する
3. 以後の評価は必ず Day1 固定基準との比較で実施する

## Follow-up if fail
1. `reports/day1/blocked_report.md` に失敗分類を1本だけ記録する
2. 失敗を以下の3区分のいずれかに固定する
   - Kaggle path unresolved
   - submission asset broken
   - reproduction baseline not fixed
3. Day1 を延長するのではなく、未解決点を明示して再実行条件を定義する

---

## Evaluation result (Day1 Evaluator, 2026-04-18, SHA: 39f4bed9)
判定: PASS
根拠ファイル:
- reports/day1/kaggle_path_inventory.md
- reports/day1/reproduction_baseline.md
- reports/day1/submission_asset_verification.md

観測事実:
- 4パスすべて RESOLVED（Kaggle 実測値）
- 中核資産6項目すべて PRESERVED（Kaggle 実測値）
- submission.zip 生成確認
- submission.zip サイズ: 2,088,412,768 bytes
- zip内ファイル数: 4
- 未確定項目数: 0
- A比較基準として固定成立
- B-policy混入なし

Failure branch: NONE
Adoption rule: 採用
Follow-up:
- Day2 以降の改善は本 Day1 基準（SHA: 39f4bed9）を A 比較基準として用いる
- Day2 では改善策Bを 1実験1変数 で投入する

---

# Decision ID
D2

## Hypothesis
Day1 の正本環境を Kaggle に固定することで、比較基準のぶれを防げる。

## Why now
Colab やローカルPCを主実行環境にすると、提出資産部の成立判定と本番環境の成立判定が分離する。  
Day1 の目的は提出資産の固定であり、本番環境に一致する Kaggle で判定する必要がある。

## Evidence expected
1. `reports/day1/reproduction_baseline.md` に `正本環境 = Kaggle` と記録されている
2. すべての主要パスが Kaggle 上の実パスで記録されている
3. `submission.zip` の生成結果が Kaggle 上の生成物として記録されている

## Adoption rule
1. Kaggle 実行を前提とした証跡のみで Day1 を完結できる
2. Colab は補助、ローカルPCは主実行環境でないと明記されている

## Rejection rule
1. Day1 の成立証跡が Kaggle 外に依存する
2. Kaggle 上で主要パスが解決できない

## Follow-up if pass
Day2 以降の実験も Kaggle 正本で判定する。

## Follow-up if fail
Kaggle 依存資産の欠落として `BLOCKED` に分類する。

---

## Evaluation result (Day1 Evaluator, 2026-04-18, SHA: 39f4bed9)
根拠ファイル:
- reports/day1/reproduction_baseline.md

評価結果:
- Kaggle 正本環境での実行証跡が成立
- 「正本環境 = Kaggle」の明記を確認
- Kaggle 実測値による成果物更新を確認

Adoption rule: 採用

---

# Decision ID
D3

## Hypothesis
Day1 では改善策Bを一切入れない方が、Aの提出資産破損と改善効果の混線を防げる。

## Why now
Day1 は精度改善日ではない。  
基準固定前に B を入れると、A比で悪化していないかを判定できない。

## Evidence expected
1. Day1 成果物に B の施策名、分析結果、学習結果が含まれていない
2. Day1 の成果物が path / baseline / submission asset / blocked 条件に限定されている

## Adoption rule
1. Day1 成果物がすべて基準固定に限定される
2. Evaluator が「精度改善混入なし」と判定する

## Rejection rule
1. min logprob
2. 再重点学習
3. deterministic chain-of-thought 再設計
4. bit manipulation 強化
5. training-serving misalignment 修正  
のいずれかが Day1 成果物に混入する

## Follow-up if pass
Day2 以降で B を 1実験1変数で投入する。

## Follow-up if fail
Day1 成果物を差し戻しし、B混入箇所を除去する。

---

## Evaluation result (Day1 Evaluator, 2026-04-18, SHA: 39f4bed9)
根拠ファイル:
- reports/day1/kaggle_path_inventory.md
- reports/day1/reproduction_baseline.md
- reports/day1/submission_asset_verification.md
- reports/day1/blocked_report.md

評価結果:
- B-policy 施策名の混入なし
- Day1 成果物は基準固定の範囲に留まっている

Adoption rule: 採用（B-policy 非混入確認）

---

# Decision ID
D4

## Hypothesis
Day1 成功の判定は「動作確認」ではなく「比較基準として固定可能」でなければならない。

## Why now
単なる動作確認では、後続の改善効果を A 比で測れない。  
比較基準として固定できることが Day1 の本質である。

## Evidence expected
1. `reports/day1/reproduction_baseline.md` に固定基準が文章化されている
2. Evaluator が同じ成果物を読んで同じ判定に到達できる
3. `PASS / FAIL / BLOCKED` の判定規則が明文化されている

## Adoption rule
1. Generator と Evaluator が同じファイル群を読める
2. 判定語が固定されている
3. 未確定項目が残っていない

## Rejection rule
1. 成果物の読み先が曖昧
2. 判定語が曖昧
3. 未確定項目が残る

## Follow-up if pass
Day1 基準を唯一の比較基準として採用する。

## Follow-up if fail
成果物パスと判定規則を再定義してから再評価する。

---

## Evaluation result (Day1 Evaluator, 2026-04-18, SHA: 39f4bed9)
根拠ファイル:
- reports/day1/kaggle_path_inventory.md
- reports/day1/reproduction_baseline.md
- reports/day1/submission_asset_verification.md
- reports/day1/blocked_report.md

判定語・状態語確認:
- kaggle_path_inventory.md: RESOLVED 使用
- reproduction_baseline.md: PRESERVED 使用
- submission_asset_verification.md: PASS 使用
- blocked_report.md: 解消履歴として FB-1 を保持。現時点の BLOCKED 分類ではない

評価結果:
- 契約語との不一致なし

Adoption rule: 採用

---

# Decision ID
D5

## Hypothesis
Day1 baseline score の契約値は、Kaggle 実測に基づく **0.85** で固定すべきである。

## Why now
これまで `0.86` と記載していた箇所があったが、Day2 比較判定の過程で baseline score の誤記が判明した。  
契約文と decision log の score 記載が誤っていると、Evaluator が誤った比較前提を使うため、再評価の前に修正が必要である。

## Evidence expected
1. `control/sprint_contract.md` に baseline score = `0.85` が明記されている
2. `reports/day1/reproduction_baseline.md` に baseline score = `0.85` が明記されている
3. Day2 成果物で baseline score = `0.85` が使われている
4. 誤記 `0.86` が REJECT 根拠として使われないことが確認できる

## Adoption rule
1. 契約書と decision log が `0.85` に更新されている
2. baseline 比較に使う score が一貫して `0.85` である
3. Evaluator が更新後の契約書を読んで再評価できる

## Rejection rule
1. 契約書または decision log に `0.86` が残る
2. baseline 比較の score 記載が複数値で混在する
3. Evaluator が更新後の正本を参照できない

## Follow-up if pass
1. baseline score = `0.85` を正式値として運用する
2. Day2 以降の score 比較は `0.85` を A基準とする
3. Evaluator に更新済み SHA を提示して再評価を依頼する

## Follow-up if fail
1. score 誤記を `BLOCKED` 相当の契約未整合として扱う
2. score 記載のある関連成果物を洗い出して再修正する

---

## Evaluation result (PM correction, 2026-04-19)
根拠ファイル:
- control/sprint_contract.md
- reports/day1/reproduction_baseline.md
- reports/day2/experiment_result.md
- reports/day2/adoption_decision.md

観測事実:
- Day1 契約書の baseline score を `0.85` に訂正
- Day1 baseline 記録の score 表記を `0.85` に訂正
- Day2 比較結果は baseline 0.85 / Day2 0.85 の同値へ更新済み
- `0.86 -> 0.85` 悪化を理由にした REJECT は撤回済み

Adoption rule: 採用
Follow-up:
- Evaluator は更新済み契約書を読んだうえで Day2 を再評価する

---

# Decision ID
D6

## Hypothesis
Day2 の Kaggle 表示スコアが baseline と同値でも、表示外の評価情報で baseline より改善していることをユーザーが PM に明示指示した場合は、`ADOPT` を許可すべきである。

## Why now
Kaggle の表示スコアは丸めにより同値に見えても、内部的には baseline より上か下かを区別できる場合がある。  
この扱いが未明文化だと、同値時の判定が毎回ぶれる。  
したがって、「同値 = HOLD を原則」「ただし PM に対するユーザー明示指示がある場合は ADOPT 可」という規則を固定する必要がある。

## Evidence expected
1. `control/day2_sprint_contract.md` に同値時規則が明記されている
2. `control/day2_evaluator_instructions.md` に同値時規則が明記されている
3. `reports/day2/adoption_decision.md` または関連成果物に、PM 指示採用の有無が記録されている
4. Evaluator が同値時に HOLD / ADOPT を規則どおり判定できる

## Adoption rule
1. 原則は `same score = HOLD` と明記されている
2. 例外として、表示外評価情報で改善していることをユーザーが PM に明示指示した場合は `ADOPT` 可と明記されている
3. その例外採用時は、PM 指示が成果物または decision log に記録されている

## Rejection rule
1. 同値時の規則が契約書と evaluator 指示で不一致である
2. PM 指示の記録がないのに同値で ADOPT している
3. 改善の有無が未記録のまま同値判定を進めている

## Follow-up if pass
1. Day2 以降は同値時 `HOLD` を原則とする
2. ユーザーから PM への明示指示がある場合のみ ADOPT 例外を使う
3. Evaluator は同値時に PM 指示の有無を確認して判定する

## Follow-up if fail
1. 同値時規則の不整合として BLOCKED 相当で扱う
2. 契約書、evaluator 指示、adoption_decision の記載を再統一する

---

## Evaluation result (PM rule update, 2026-04-19)
根拠ファイル:
- control/day2_sprint_contract.md
- control/day2_evaluator_instructions.md

観測事実:
- 同値時は HOLD を原則とする規則を契約書に追記
- 表示外評価情報で改善していることをユーザーが PM に明示指示した場合は ADOPT 可の例外規則を evaluator 指示に追記

Adoption rule: 採用
Follow-up:
- 今回の Day2 Run1 は、ユーザーから PM に対して「同値だがスコアは改善なので採用」と明示指示があるため、Evaluator はその記録を読んだうえで再判定する
