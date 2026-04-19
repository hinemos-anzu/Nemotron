# Day2 Experiment Result

## 実行環境
- Kaggle（正本環境）

## 比較対象
- baseline SHA: 39f4bed90392567517b606d1301ae1c36a86a97c
- baseline score (A): 0.86

## 実行条件
- 1実験1変数: B1 training-serving misalignment 修正のみ
- 実装対象: `original-nemotron-asymmetric-svd-26041602`
- 元コード主要機能（推論 / submission.csv / submission.zip）を保持

## 観測結果（Kaggle 実測）
- timestamp_utc: 2026-04-19T02:21:47.227365Z
- source_of_truth: Kaggle
- one_variable_rule: True
- submission_zip.exists: True
- submission_zip.status: PASS
- submission_zip.size_bytes: 2088413107
- submission_zip.file_count: 5
- submission_assets_preserved: True
- LB score (Day2 B1): 0.85
- comparable_against_baseline: True
- provisional_verdict: REJECT

## A比で悪化したか否か
- 悪化した（0.86 -> 0.85）

## 0.86超えに向かう根拠の有無
- なし（LB実測で baseline を下回った）

## 判定上の分類
- worse than baseline
