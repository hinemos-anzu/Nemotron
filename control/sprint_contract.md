# Sprint title
Day1 Sprint Contract — ベースAの0.85再現基準固定

# Goal
Day1 の目的は **Aの0.85再現基準固定** のみである。  
精度改善は目的に含めない。  
Kaggle を正本環境として、ベースAの提出資産部が成立し、以後の改善を A 比で判定できる比較基準を固定する。

# Baseline score correction
- 以前の契約文では baseline score を `0.86` と記載していたが、これは誤記である。
- Day1 の正式 baseline score は **0.85** とする。
- 根拠は Kaggle 実測の baseline LB 記録である。
- 以後、A比較基準のスコア参照は **0.85** を正とする。

# In scope
1. Kaggle 上でベースAの提出資産部を成立させるための再現条件の明文化
2. 以下の Kaggle 上の実在パス確認
   - adapter path
   - model path
   - tokenizer path
   - input data path
3. ベースAの提出資産部の構成要素の保存確認
   - adapter変換
   - Offline Asymmetric SVD Surgery
   - key rename
   - expert unfuse
   - gate_proj + x_proj -> in_proj の統合
   - submission.zip 生成
4. `submission.zip` の生成可否確認
5. Aの再現条件の固定
6. 失敗時の BLOCKED 条件の分類と明文化
7. Generator / Evaluator が同じ基準で判定できる成果物パスの固定

# Out of scope
1. 改善策Bの投入
2. training-serving misalignment 修正の実装
3. min logprob 分析本体
4. low-minlogprob 再重点学習
5. deterministic chain-of-thought 再設計
6. bit manipulation 強化
7. router 改善
8. 多数決改善
9. SymPy系改善
10. 精度改善実験
11. Aを上回るスコア獲得の検証

# Inputs
1. ベースAの既存コード一式
2. ベースAが依存する Kaggle 上の入力資産
3. ベースAが **0.85** を記録した既知の再現情報
4. Kaggle Notebook / Kaggle Dataset / Kaggle Input の実体
5. 生成対象成果物の保存先ルール

# Required outputs
## Planner が定義する必須ファイル
1. `control/sprint_contract.md`
2. `control/decision_log.md`

## Generator が作成する必須成果物
3. `reports/day1/kaggle_path_inventory.md`
4. `reports/day1/reproduction_baseline.md`
5. `reports/day1/submission_asset_verification.md`
6. `reports/day1/blocked_report.md` 〔失敗時のみ必須〕

# Success criteria
以下の全条件を満たした場合のみ Day1 成功と判定する。

| ID | 観測対象 | 合格条件 |
|---|---|---|
| SC-1 | 正本環境 | `reports/day1/reproduction_baseline.md` に「正本環境 = Kaggle」と明記されている |
| SC-2 | パス解決 | `reports/day1/kaggle_path_inventory.md` に adapter / model / tokenizer / input data の4種類すべての絶対パスまたは Kaggle 入力パスが記録され、各項目が `RESOLVED` である |
| SC-3 | 提出資産保存 | `reports/day1/reproduction_baseline.md` に、ベースAの中核資産6項目が個別に列挙され、各項目が `PRESERVED` と記録されている |
| SC-4 | submission.zip | `reports/day1/submission_asset_verification.md` に `submission.zip` の生成結果が記録され、出力パス、ファイルサイズ、zip内ファイル一覧の3点が確認できる |
| SC-5 | A再現条件固定 | `reports/day1/reproduction_baseline.md` に、Aの再現に必要な前提条件が列挙され、未確定項目が `0` 件である |
| SC-6 | 比較基準固定 | `reports/day1/reproduction_baseline.md` に「今後の改善はこの基準を A 比較基準として用いる」と明記されている |
| SC-7 | BLOCKED 明文化 | 失敗時は `reports/day1/blocked_report.md` が存在し、Failure branches のいずれか1本に分類されている |
| SC-8 | Aの提出資産を壊していない | `reports/day1/submission_asset_verification.md` に「Aの提出資産部を壊していない」と判定した根拠が記録されている |

# Failure branches
失敗時分岐は以下の3本に限定する。

## FB-1: Kaggle path unresolved
条件:
- adapter path / model path / tokenizer path / input data path のいずれかが `RESOLVED` にならない

判定:
- Day1 は `BLOCKED`
- 原因は「Kaggle 上の入力資産未解決」

必須記録先:
- `reports/day1/blocked_report.md`

## FB-2: submission asset broken
条件:
- `submission.zip` が生成できない
- 生成物が空である
- zip 内構造が欠損している
- ベースAの提出資産部の構成要素に `BROKEN` が1つでもある

判定:
- Day1 は `FAIL`
- 原因は「提出資産部不成立」

必須記録先:
- `reports/day1/submission_asset_verification.md`
- `reports/day1/blocked_report.md`

## FB-3: reproduction baseline not fixed
条件:
- Aの再現条件に未確定項目が残る
- **0.85**再現基準として固定できる説明が書けない
- Generator と Evaluator が同じ比較基準を読めない

判定:
- Day1 は `FAIL`
- 原因は「比較基準未固定」

必須記録先:
- `reports/day1/reproduction_baseline.md`
- `reports/day1/blocked_report.md`

# Stop conditions
以下のいずれかに達した時点で Day1 を終了する。

1. Success criteria を全件満たした
2. Failure branches のいずれか1本に確定分類された
3. Out of scope に触れる作業が必要になった  
   この場合は Day1 を停止し、Day2 以降へ持ち越す

# Notes for Generator
1. Day1 ではコード改善をしない
2. Day1 では改善策Bを入れない
3. Day1 の作業対象は「Aの0.85再現基準固定」のみ
4. Kaggle を正本環境として記録する
5. すべての確認結果は以下のパスへ出力する
   - `reports/day1/kaggle_path_inventory.md`
   - `reports/day1/reproduction_baseline.md`
   - `reports/day1/submission_asset_verification.md`
   - `reports/day1/blocked_report.md` 〔必要時〕
6. 各成果物では、各項目を `RESOLVED / UNRESOLVED`、`PRESERVED / BROKEN`、`PASS / FAIL / BLOCKED` のいずれかで明示する
7. 「動いたらOK」と書かない  
   必ず「A比の比較基準として固定できるか」で書く
8. Generator が最初に読むべきファイルは以下の2つ
   - `control/sprint_contract.md`
   - `control/decision_log.md`

# Notes for Evaluator
1. Day1 の評価対象は精度ではない
2. Day1 の評価対象は以下の3点のみ
   - Aの提出資産を壊していないか
   - Kaggle上で再現条件が固定できたか
   - 比較基準として運用可能か
3. Evaluator は以下の順で成果物を読む
   1. `control/sprint_contract.md`
   2. `control/decision_log.md`
   3. `reports/day1/kaggle_path_inventory.md`
   4. `reports/day1/reproduction_baseline.md`
   5. `reports/day1/submission_asset_verification.md`
   6. `reports/day1/blocked_report.md` 〔存在時〕
4. 評価結果は `PASS / FAIL / BLOCKED` の3値で返す
5. 改善策Bへの言及が成果物に混入していた場合、その時点で `FAIL` とする
6. Aの再現条件に未確定項目が1件でも残る場合、`PASS` にしない
