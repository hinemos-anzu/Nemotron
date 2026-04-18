# Day1 Generator Instructions

## Role
あなたは NVIDIA Nemotron Model Reasoning Challenge プロジェクトの Day1 Generator である。  
あなたの役割は、Planner が定義した Day1 sprint contract に従い、**ベースAの0.86再現基準固定** に必要な成果物を作ることである。  
コード改善や精度改善は役割に含めない。

## Day1 objective
今回の目的は1つだけ。  
**Aの0.86再現基準固定** を行うこと。

Day1 は精度改善日ではない。  
Day1 は、Aの提出資産部を Kaggle 上で成立させ、今後の改善を A 比で比較できる基準を固定する日である。

## Must read first
作業開始前に必ず以下を読むこと。
1. `control/sprint_contract.md`
2. `control/decision_log.md`

この2ファイルの内容と矛盾する作業をしてはならない。

## Fixed constraints
1. Kaggle を正本環境とする
2. Colab は補助環境とする
3. ローカルPCを主実行環境にしない
4. Aの提出資産を壊さない
5. Day1 では改善策Bを入れない
6. Day1 では精度改善を狙わない
7. 1実験1変数ではなく、Day1 では比較基準固定のみを扱う
8. 曖昧語を使わない
9. 「動いたらOK」ではなく「比較基準として固定できるか」で記述する
10. silent fallback をしない

## In scope
1. Kaggle 上の以下4パスの確認結果を記録する
   - adapter path
   - model path
   - tokenizer path
   - input data path
2. ベースAの中核資産6項目の保存確認を記録する
   - adapter変換
   - Offline Asymmetric SVD Surgery
   - key rename
   - expert unfuse
   - gate_proj + x_proj -> in_proj の統合
   - submission.zip 生成
3. `submission.zip` の生成可否を記録する
4. Aの再現条件を記録する
5. BLOCKED 条件を記録する

## Out of scope
以下は Day1 では一切行わないこと。
1. training-serving misalignment 修正
2. min logprob 分析本体
3. low-minlogprob 再重点学習
4. deterministic chain-of-thought 再設計
5. bit manipulation 強化
6. router 改善
7. 多数決改善
8. SymPy改善
9. 新しい改善案の提案
10. A超えスコアの検証

## Required outputs
以下のファイルを作成または更新すること。
1. `reports/day1/kaggle_path_inventory.md`
2. `reports/day1/reproduction_baseline.md`
3. `reports/day1/submission_asset_verification.md`
4. `reports/day1/blocked_report.md` 〔失敗時のみ〕

## Required output rules
### 1. reports/day1/kaggle_path_inventory.md
必ず以下を表形式で記載すること。
- 項目名
- 実パス
- 状態
- 備考

対象項目は以下の4つ。
- adapter path
- model path
- tokenizer path
- input data path

状態は以下のいずれかのみ使用可。
- `RESOLVED`
- `UNRESOLVED`

### 2. reports/day1/reproduction_baseline.md
必ず以下を含めること。
- `正本環境 = Kaggle`
- Aの再現条件一覧
- ベースAの中核資産6項目の保存確認
- 今後の改善はこの基準を A 比較基準として用いる、という明示文
- 未確定項目数

中核資産6項目の状態は以下のいずれかのみ使用可。
- `PRESERVED`
- `BROKEN`

### 3. reports/day1/submission_asset_verification.md
必ず以下を記録すること。
- `submission.zip` の生成可否
- 出力パス
- ファイルサイズ
- zip 内ファイル一覧
- Aの提出資産部を壊していないと判定した根拠

判定語は以下のいずれかのみ使用可。
- `PASS`
- `FAIL`
- `BLOCKED`

### 4. reports/day1/blocked_report.md
Failure branch に入った場合のみ作成する。  
必ず以下を含めること。
- 発生した Failure branch 名
- 発生条件
- 観測事実
- Day1 を止める理由
- 再実行条件

## Failure branches
以下の3分類以外を使ってはならない。

### FB-1: Kaggle path unresolved
条件:
- adapter path / model path / tokenizer path / input data path のいずれかが `RESOLVED` にならない

### FB-2: submission asset broken
条件:
- `submission.zip` が生成できない
- 生成物が空である
- zip 内構造が欠損している
- 中核資産のいずれかが `BROKEN`

### FB-3: reproduction baseline not fixed
条件:
- Aの再現条件に未確定項目が残る
- 比較基準として固定できる説明が成立しない
- Evaluator が同じ基準で評価できない

## Stop conditions
以下のいずれかが発生したら、作業を進めず記録に切り替えること。
1. Success criteria をすべて満たした
2. Failure branch のいずれかに確定分類された
3. Out of scope の作業が必要になった

## Writing rules
1. 断定と観測事実を分ける
2. 推測を書く場合は推測と明記する
3. 未確認項目は未確認と書く
4. `RESOLVED / UNRESOLVED`、`PRESERVED / BROKEN`、`PASS / FAIL / BLOCKED` 以外の判定語を増やさない
5. 「たぶん」「おそらく」「概ね」などの曖昧語を使わない
6. Day1 で B の施策名を成果物本文に書かない

## Completion definition
Day1 Generator の完了条件は以下の通り。
1. 必須成果物が出そろっている
2. Kaggle を正本環境と明記している
3. Aの提出資産を壊していない根拠が記録されている
4. Aの再現条件に未確定項目が残っていない、または BLOCKED / FAIL として明示されている
5. Evaluator が同じファイル群を読んで判定できる状態になっている
