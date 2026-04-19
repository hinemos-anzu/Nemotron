# Day2 Generator Instructions

## Role
あなたは NVIDIA Nemotron Model Reasoning Challenge プロジェクトの Day2 Generator である。  
あなたの役割は、Day1 で固定したベースA（SHA: `39f4bed90392567517b606d1301ae1c36a86a97c`）を比較基準として維持しながら、改善策Bの最初の1変数を **元コードを保持したまま** 投入し、比較可能な成果物を作成することである。

## Day2 objective
今回の目的は1つだけ。  
**B1: training-serving misalignment 修正を、1実験1変数で A固定基準に対して投入すること。**

Day2 は多施策投入日ではない。  
Day2 は、Aの提出資産を壊さず、**指定された元コードを GitHub 上で保持したうえで、その同一コードに B1 単独の差分を加え、A 比で判定できる状態を作る日** である。

## Fixed baseline
以下を比較基準として固定すること。
- Repository: `hinemos-anzu/Nemotron`
- Branch: `planner/day1-logprob`
- Baseline SHA: `39f4bed90392567517b606d1301ae1c36a86a97c`

## Must read first
作業開始前に必ず以下を読むこと。
1. `control/decision_log.md`
2. `control/day2_sprint_contract.md`
3. `control/day2_generator_instructions.md`
4. `control/day2_evaluator_instructions.md`
5. `control/day2_b1_implementation_spec.md`
6. `reports/day1/kaggle_path_inventory.md`
7. `reports/day1/reproduction_baseline.md`
8. `reports/day1/submission_asset_verification.md`

## Start conditions
以下をすべて満たした場合のみ着手してよい。
1. 実装対象コードが指定されている
2. `control/day2_b1_implementation_spec.md` が存在する
3. B1 の具体変更仕様が定義されている
4. 元コード保持条件が明示されている
5. 作業ブランチが `planner/day1-logprob` に固定されている

いずれか1つでも欠ける場合は着手せず、`NOT READY` を返すこと。

## Fixed constraints
1. Kaggle を正本環境とする
2. Day2 の変更対象は **B1: training-serving misalignment 修正のみ** とする
3. 1実験1変数を厳守する
4. A固定基準 SHA を変更しない
5. Aの提出資産を壊さない
6. B1 以外の B施策を入れない
7. router / 多数決 / SymPy 系は触らない
8. 曖昧語を使わない
9. silent fallback をしない
10. A比で悪化した場合は隠さず記録する
11. **指定された元コードを GitHub 上の対象ファイルとして反映し、その同一コードに対して B1 を最小差分で追加する**
12. **補助ファイルの新規作成だけで代替してはならない**
13. **元コードの主要機能（推論、`submission.csv` 生成、`submission.zip` 生成）を保持する**

## In scope
以下のみ実施してよい。
1. 元コードの GitHub 反映
2. training-serving misalignment 修正の 1変数変更
3. A固定基準との比較記録
4. `submission.zip` の成立確認
5. Day2 採否判断に必要な成果物作成
6. Day2 evidence 採取コード追加

## Out of scope
以下は一切行わないこと。
1. min logprob 分析本体
2. low-minlogprob 再重点学習
3. deterministic chain-of-thought 再設計
4. bit manipulation 強化
5. router 改善
6. 多数決改善
7. SymPy改善
8. 2変数以上の同時変更
9. A固定基準の再編集
10. 無関係な提出資産ロジック変更
11. 元コード主要機能の削除または省略
12. evidence 追加だけでの完了扱い

## Required outputs
以下のファイルを作成または更新すること。
1. 実装対象コードの GitHub 反映版
2. `reports/day2/change_manifest.md`
3. `reports/day2/experiment_result.md`
4. `reports/day2/submission_asset_check.md`
5. `reports/day2/adoption_decision.md`

## Required output rules

### 1. 実装対象コード
必ず以下を満たすこと。
- 指定された元コードが GitHub 上の対象ファイルとして存在する
- 元コードの主要機能（推論、`submission.csv` 生成、`submission.zip` 生成）が保持されている
- その同一コードに対して B1 が最小差分で追加されている

### 2. reports/day2/change_manifest.md
必ず以下を含めること。
- baseline SHA
- modified SHA
- 今回変更した変数は 1つだけである、という明示文
- **元コードを保持したうえで B1 を追加した** という明示文
- 変更内容の要約
- 変更していない施策一覧
- どの config / metadata / serving 前提を修正したか

### 3. reports/day2/experiment_result.md
必ず以下を含めること。
- 実行環境 = Kaggle
- 比較対象 = A固定基準
- 実行条件
- 観測結果
- A比で悪化したか否か
- 0.86超えに向かう根拠の有無

### 4. reports/day2/submission_asset_check.md
必ず以下を含めること。
- Aの提出資産部が壊れていないか
- `submission.zip` 生成可否
- 出力パス
- ファイルサイズ
- zip内ファイル一覧
- 判定: PASS / FAIL / BLOCKED

### 5. reports/day2/adoption_decision.md
必ず以下を含めること。
- 最終判定: ADOPT / REJECT / HOLD
- 判定理由
- A比で悪化していないか
- 次に進む条件

## Success criteria
以下をすべて満たした場合のみ Day2 Generator 完了とする。
1. 指定された元コードが GitHub 上の対象ファイルとして反映されている
2. 変更が B1 の1変数だけである
3. Kaggle 上で実行されている
4. 提出資産部が壊れていない
5. `submission.csv` と `submission.zip` の生成経路が保持されている
6. `submission.zip` の path / size / file list が記録されている
7. A固定基準との比較結果が明記されている
8. ADOPT / REJECT / HOLD のいずれかに到達している

## Failure branches

### FB-1: asset broken
条件:
- 提出資産部が壊れた
- `submission.zip` が生成できない
- zip構造が壊れている
- 元コードの提出経路が失われている

### FB-2: multi-variable contamination
条件:
- B1 以外の施策が混入した
- 2変数以上を同時に変更した
- 元コード保持ではなく別実装へ置換している

### FB-3: no comparable evidence
条件:
- A固定基準との比較ができない
- 記録不足で ADOPT / REJECT / HOLD が判定不能

## NOT READY conditions
以下のいずれかなら `NOT READY` を返すこと。
1. 実装対象コードが未指定
2. `control/day2_b1_implementation_spec.md` が未存在
3. B1 の具体修正が未定義
4. 元コード本体が GitHub 反映版として存在しない
5. 元コード主要処理を含まない補助ファイルだけを追加している
6. `submission.csv` / `submission.zip` の元コード相当経路が保持されていない
7. evidence 追加だけで終わっている
8. router / sampling / SymPy 等に手を入れている
9. 1変数性を示せない

## Stop conditions
以下のいずれかが発生したら、作業を止めて記録へ切り替えること。
1. Success criteria を満たした
2. Failure branch のいずれかに確定分類された
3. B1 以外の施策が必要になった

## Writing rules
1. 観測事実と判断を分ける
2. 推測を書く場合は推測と明記する
3. 未確認項目は未確認と書く
4. B1 以外の施策名を書かない
5. A固定基準 SHA を必ず書く
6. 曖昧語を使わない
7. 元コード保持の有無を必ず明記する

## Completion definition
Day2 Generator の完了条件は以下の通り。
1. 必須成果物が出そろっている
2. baseline SHA と modified SHA が明記されている
3. 元コードの GitHub 反映版が存在する
4. `submission.csv` / `submission.zip` 証跡がある
5. A 比較結果がある
6. ADOPT / REJECT / HOLD が記録されている
7. B1 本体が入っている
