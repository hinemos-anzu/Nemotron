# Issue Report

- date: 2026-04-19
- issue_id: ISSUE-20260419-01
- title: 元コード＋B1 の意図が指示文で十分に固定されていなかった
- status: RESOLVED
- related_run: Day2 Run1
- baseline_sha: 39f4bed90392567517b606d1301ae1c36a86a97c
- target_sha: f67e8f5c6031b0fa8c64ca6b1c938f33cb76fbff

## 1. Problem
PM の意図は「指定された元コードを土台として保持し、その上に B1 を 1変数で追加すること」だったが、Generator 向け指示文ではその意図が十分に固定されていなかった。

結果として、Generator は元コード本体の GitHub 反映ではなく、Day2 evidence 採取用の補助実装ファイル追加で Step 2 要件を満たせると解釈した。これにより、「元コード＋B1」ではなく「補助実装＋evidence」に寄った成果物が一度生成された。

## 2. Impact
- Step 2 完了判定が一度誤って前進方向に解釈された。
- Kaggle 実行役の起動条件判断が不安定になった。
- 「submission.zip 生成を含む元コード主要機能が保持されているか」という重要観点が、指示文上で十分に拘束されていないことが露呈した。
- PM / Generator 間で「B1 本体」と「B1 周辺の補助実装」の境界が曖昧になった。

## 3. Root Cause
原因は 2 点である。

1. PM 指示文において、「元コードをまず GitHub 上の対象ファイルとして完全反映し、その同一ファイルに対して B1 を最小差分で追加すること」が明文化されていなかった。
2. Step 2 完了条件に、「元コードの主要機能（推論、submission.csv 生成、submission.zip 生成）が保持されていること」が入っていなかった。

## 4. Resolution
以下をもって解決した。

1. PM が本 issue を正式化し、意図と指示文の差分を明文化した。
2. `control/day2_b1_implementation_spec.md` を追加し、B1 の意味、変更対象、変更禁止、1変数定義、実装完了条件を固定した。
3. 今後 Generator には、「元コード完全保持＋その上に B1 最小差分追加」で再指示する方針に切り替えた。

## 5. Recurrence Prevention
今後は次を必須ルールとする。

1. 実装対象コードが指定されていても、「元コード完全保持」が明文化されていなければ Step 2 を開始しない。
2. Generator 指示文には必ず以下を入れる。
   - 指定された元コードを GitHub 上の対象ファイルとして反映すること
   - 元コードの主要機能を保持すること
   - その同一ファイルに B1 を最小差分で追加すること
   - 補助ファイル新規作成だけで代替してはならないこと
3. Step 2 完了条件には必ず以下を入れる。
   - 推論実行が保持されていること
   - `submission.csv` 生成が保持されていること
   - `submission.zip` 生成が保持されていること
4. NOT READY 条件には必ず以下を入れる。
   - 元コード本体が GitHub 反映版として存在しない
   - 元コード主要処理を含まない補助ファイルだけを追加している
   - `submission.csv` / `submission.zip` の元コード相当経路が保持されていない

## 6. Required Instruction Updates
以下の指示文に追記が必要である。

### A. `control/day2_pm_instructions.md`
追記内容:
- 実装対象コードが指定されていても、変更仕様と元コード保持条件が未確定なら Generator を起動してはならない。
- PM は「対象コードの特定」と「変更仕様の確定」を分けて管理すること。

### B. `control/day2_generator_instructions.md`
追記内容:
- 実装対象コードの指定だけでは着手してはならない。
- 指定された元コードを GitHub 上の対象ファイルとして反映したうえで、その同一ファイルに B1 を最小差分で追加すること。
- evidence 追加だけで B1 実装完了と見なしてはならない。
- 補助ファイル新規作成のみで代替してはならない。

### C. `control/day2_codex_request.md`
追記内容:
- 元コードの主要機能（推論、submission.csv 生成、submission.zip 生成）を保持すること。
- NOT READY 条件として、元コード未反映と主要機能欠落を明記すること。

## 7. Owner
- PM / まとめ役: ChatGPT
- 実装担当: Generator
- 実行担当: ユーザー（Kaggle 実行役）

## 8. Notes
この issue は、PM の意図が誤っていたのではなく、指示文の拘束力が不足していたことに起因する。したがって、再発防止の中心は「意図の修正」ではなく、「指示文の強制力向上」である。
