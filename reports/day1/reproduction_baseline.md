# Day1 Reproduction Baseline (A=0.86 固定)

## 正本環境
- 正本環境 = Kaggle
- 補助環境 = Colab
- ローカルPCは主実行環境にしない

## 観測事実
- Kaggle 上4パスはすべて UNRESOLVED。
- `control/sprint_contract.md`、`control/decision_log.md`、`control/day1_generator_instructions.md`、`control/day1_evaluator_instructions.md` はこの実行時点で未読（ファイル未存在）だった。
- Kaggle 正本環境に未到達のため、中核資産6項目の保存検証に進めていない。

## Aの再現条件一覧
1. Kaggle 上で adapter/model/tokenizer/input data の4パスが RESOLVED であること。
2. 中核資産6項目（adapter変換 / Offline Asymmetric SVD Surgery / key rename / expert unfuse / gate_proj+x_proj->in_proj 統合 / submission.zip 生成）が、Kaggle 上の実物で検証されること。
3. `submission.zip` が Kaggle 上で生成され、出力パス・サイズ・zip内一覧が観測記録されること。
4. Evaluator が同一条件で再読できる記録順が明示されること。

## 中核資産6項目の保存確認（FB-1時点）
| 項目 | 現在の検証状態 | 根拠 |
|---|---|---|
| adapter変換 | 未検証（FB-1） | Kaggle path unresolved のため検証未到達 |
| Offline Asymmetric SVD Surgery | 未検証（FB-1） | Kaggle path unresolved のため検証未到達 |
| key rename | 未検証（FB-1） | Kaggle path unresolved のため検証未到達 |
| expert unfuse | 未検証（FB-1） | Kaggle path unresolved のため検証未到達 |
| gate_proj + x_proj -> in_proj の統合 | 未検証（FB-1） | Kaggle path unresolved のため検証未到達 |
| submission.zip 生成 | 未検証（FB-1） | 生成前提（4パス解決）が未充足 |

## 判断
- 本日の失敗分類は FB-1: Kaggle path unresolved のみ。
- 上表は破損断定ではない。実証未了のため保存性判定は保留。

## A比較基準の明示
- 今後の改善はこの基準を A 比較基準として用いる。
- ただし本時点では未確定項目が残存するため、比較基準として固定は未成立。

## 未確定項目数
- 11
  - Kaggle 4パス: 4
  - 中核資産6項目の保存検証: 6
  - Evaluator 同一基準評価成立確認: 1

## Evaluator に渡す正式な読み順（契約準拠）
1. `control/sprint_contract.md`
2. `control/decision_log.md`
3. `control/day1_generator_instructions.md`
4. `control/day1_evaluator_instructions.md`
5. `reports/day1/kaggle_path_inventory.md`
6. `reports/day1/reproduction_baseline.md`
7. `reports/day1/submission_asset_verification.md`
8. `reports/day1/blocked_report.md` 〔存在時〕
