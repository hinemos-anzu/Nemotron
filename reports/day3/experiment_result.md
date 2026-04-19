# Day3 Experiment Result

## 実行環境
- Kaggle（正本環境）

## 比較対象
- A固定基準（baseline SHA: 39f4bed90392567517b606d1301ae1c36a86a97c）

## 実行条件
- 実装対象: `original-nemotron-asymmetric-svd-26041602`
- 1実験1変数: `temperature` のみ変更
- 変更値: `temperature=0.6`（他の推論パラメータは据え置き）

## 観測結果（Kaggle実測）
- timestamp_utc: 2026-04-19T12:57:46.120026Z
- source_of_truth: Kaggle
- one_variable_rule: True
- submission_zip.exists: True
- submission_zip.status: PASS
- submission_zip.size_bytes: 2088413108
- submission_zip.file_count: 5
- submission_assets_preserved: True
- comparable_against_baseline: False
- worse_than_baseline: UNCONFIRMED
- evidence_for_gt_086: UNCONFIRMED
- provisional_verdict: HOLD

## baseline 比
- BLOCKED（`comparable_against_baseline=False` のため、改善 / 同値 / 悪化の確定判定は不可）

## 補足メモ
- `submission.zip` は正常生成（PASS）。
- baseline 比較可能性を満たす追加証跡が必要。
