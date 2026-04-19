# Day1 Reproduction Baseline (A=0.85 固定)

## 正本環境
- 正本環境 = Kaggle
- 補助環境 = Colab
- ローカルPCは主実行環境にしない

## Baseline score
- baseline score (A): 0.85
- 以前の 0.86 記載は誤記であり、Kaggle 実測に基づく正式値は 0.85 とする

## Aの再現条件一覧（Kaggle 実測反映後）
1. adapter/model/tokenizer/input data の4パスがすべて RESOLVED であること。
2. 中核資産6項目がすべて PRESERVED であること。
3. `submission.zip` の生成が確認され、パス・サイズ・内容一覧が記録されていること。
4. Evaluator が同一条件で再読できる記録順が明示されること。

## 中核資産6項目の保存確認
| 項目 | 状態 | 根拠 |
|---|---|---|
| adapter_conversion | PRESERVED | Kaggle 実測値 |
| offline_asymmetric_svd_surgery | PRESERVED | Kaggle 実測値 |
| key_rename | PRESERVED | Kaggle 実測値 |
| expert_unfuse | PRESERVED | Kaggle 実測値 |
| gate_x_to_in_proj_merge | PRESERVED | Kaggle 実測値 |
| submission_zip_generation_path | PRESERVED | `/kaggle/working/submission.zip` 生成確認 |

## Day1 集計
- unresolved_count: 0
- unresolved_items: none
- provisional_verdict: PASS
- failure_branch: NONE

## A比較基準の明示
- 今後の改善はこの基準を A 比較基準として用いる。
- baseline score は 0.85 を正式値とする。
- 比較基準として固定成立。

## 未確定項目数
- 0

## Evaluator に渡す正式な読み順（契約準拠）
1. `control/sprint_contract.md`
2. `control/decision_log.md`
3. `control/day1_generator_instructions.md`
4. `control/day1_evaluator_instructions.md`
5. `reports/day1/kaggle_path_inventory.md`
6. `reports/day1/reproduction_baseline.md`
7. `reports/day1/submission_asset_verification.md`
8. `reports/day1/blocked_report.md` 〔存在時〕
