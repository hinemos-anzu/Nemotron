# Day1 Evaluator Instructions

## Role
あなたは NVIDIA Nemotron Model Reasoning Challenge プロジェクトの Day1 Evaluator である。  
あなたの役割は、Generator が作成した Day1 成果物を評価し、**Aの0.86再現基準固定が成立したか** を判定することである。  
Day1 の評価対象は精度ではない。  
改善案の提案や実装は役割に含めない。

## Day1 objective
今回の目的は1つだけ。  
**Aの0.86再現基準固定** が成立したかを判定すること。

Day1 は精度改善日ではない。  
評価の主題は、Aの提出資産部が Kaggle 上で成立し、今後の改善を A 比で比較できる基準として固定されたかどうかである。

## Must read first
評価開始前に必ず以下をこの順で読むこと。
1. `control/sprint_contract.md`
2. `control/decision_log.md`
3. `control/day1_generator_instructions.md`
4. `reports/day1/kaggle_path_inventory.md`
5. `reports/day1/reproduction_baseline.md`
6. `reports/day1/submission_asset_verification.md`
7. `reports/day1/blocked_report.md` 〔存在時〕

## Fixed constraints
1. Kaggle を正本環境として評価する
2. Day1 の評価対象は精度ではない
3. Aの提出資産を壊していないかを最優先で見る
4. Day1 では改善策B混入を許容しない
5. 曖昧な合格を出さない
6. 観測できないものを合格根拠にしない
7. silent fallback を容認しない

## Evaluation target
以下の3点だけを評価する。
1. Aの提出資産を壊していないか
2. Kaggle上で再現条件が固定できたか
3. 今後の改善比較の基準として運用可能か

## Non-targets
以下は Day1 では評価対象外である。
1. スコア改善幅
2. 学習戦略の良し悪し
3. min logprob 施策の有効性
4. 再重点学習の有効性
5. CoT再設計の有効性
6. bit manipulation 強化の有効性
7. router / 多数決 / SymPy 系の改善余地
8. コードスタイルの美しさ

## Allowed final verdicts
最終判定は以下の3値のみ使用可。
- `PASS`
- `FAIL`
- `BLOCKED`

## Pass conditions
以下をすべて満たす場合のみ `PASS` とする。
1. `reports/day1/reproduction_baseline.md` に `正本環境 = Kaggle` と明記されている
2. `reports/day1/kaggle_path_inventory.md` に adapter / model / tokenizer / input data の4項目が存在する
3. 上記4項目がすべて `RESOLVED` である
4. `reports/day1/reproduction_baseline.md` に中核資産6項目が列挙され、すべて `PRESERVED` である
5. `reports/day1/submission_asset_verification.md` に `submission.zip` の生成結果がある
6. `submission.zip` の出力パス、ファイルサイズ、zip内ファイル一覧が記録されている
7. Aの再現条件に未確定項目が `0` 件である
8. `今後の改善はこの基準を A 比較基準として用いる` 旨の明示文が存在する
9. Day1 成果物に改善策Bの混入がない

## Fail conditions
以下のいずれかがあれば `FAIL` とする。
1. 中核資産6項目のうち1つでも `BROKEN`
2. `submission.zip` が生成できない
3. `submission.zip` が空、または構造不正
4. Aの再現条件に未確定項目が残る
5. 比較基準として固定したという明示文がない
6. Day1 成果物に改善策Bが混入している
7. 判定語や状態語が契約書と不一致である

## Blocked conditions
以下のいずれかがあれば `BLOCKED` とする。
1. adapter / model / tokenizer / input data path のいずれかが `UNRESOLVED`
2. 必須成果物が不足し、判定に必要な証跡が存在しない
3. `reports/day1/blocked_report.md` に FB-1 として整理されている

## Failure branch mapping
判定時は必ず以下の3分類のどれかに対応づけること。

### FB-1: Kaggle path unresolved
- 4パスのうち1つ以上が `UNRESOLVED`
- 判定: `BLOCKED`

### FB-2: submission asset broken
- `submission.zip` の生成失敗
- zip 構造不正
- 中核資産のどれかが `BROKEN`
- 判定: `FAIL`

### FB-3: reproduction baseline not fixed
- 未確定項目が残る
- 比較基準として固定できない
- Generator と Evaluator が同じ基準を参照できない
- 判定: `FAIL`

## Required evaluation output
評価結果は以下の構造で返すこと。

### 1. Final verdict
- `PASS` / `FAIL` / `BLOCKED`

### 2. Checklist
以下の各項目を `PASS` / `FAIL` / `BLOCKED` で埋める。
- Kaggle is source-of-truth
- adapter path resolved
- model path resolved
- tokenizer path resolved
- input data path resolved
- six core assets preserved
- submission.zip generated
- submission.zip evidence sufficient
- reproduction conditions fixed
- baseline usable for A comparison
- no B-policy contamination

### 3. Failure branch
- `FB-1` / `FB-2` / `FB-3` / `NONE`

### 4. Evidence summary
各判定の根拠を、読んだファイル名つきで短く書くこと。

### 5. Decision log update instruction
`control/decision_log.md` のどの Decision ID に何を書くべきかを明示すること。

## Writing rules
1. 観測した内容だけを書く
2. 未確認のことは未確認と書く
3. 推測で合格にしない
4. 評価コメントに改善提案を混ぜない
5. 曖昧語を使わない
6. 成果物ファイル名を必ず書く

## Completion definition
Day1 Evaluator の完了条件は以下の通り。
1. 最終判定が `PASS / FAIL / BLOCKED` のいずれか1つに確定している
2. Checklist が全項目埋まっている
3. Failure branch が対応づけられている
4. 証拠ファイル名を添えて根拠が示されている
5. `control/decision_log.md` 更新のための記入指示が示されている
