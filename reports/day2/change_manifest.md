# Day2 Change Manifest

- baseline_sha: 39f4bed90392567517b606d1301ae1c36a86a97c
- modified_sha: aa235d36e5b0d1e81707bf91b83161adcd8747e0
- changed_variable_count: 1
- changed_variable: B1 training-serving misalignment 修正

## 元コード保持の明示
- `original-nemotron-asymmetric-svd-26041602` の主要処理（推論実行 / `submission.csv` 生成 / `submission.zip` 生成）を保持したまま、同一コードへ B1 を最小差分で追加した。

## B1 変更内容（1変数）
1. surgery 後 `adapter_model.safetensors` の key から `target_modules` を再推定。
2. `adapter_config.json` の `target_modules` を serving 実体に整合。
3. `inference_mode=True` に補正。
4. `serving_alignment.json` を出力して before/after を記録。
5. `WORKING_ADAPTER_DIR` に必要ファイルがない場合、入力 adapter から最小コピーして load 前提を補正。
6. Day2 evidence (`day2_evidence.json` / `day2_evidence.md`) を生成。

## 変更していない施策一覧
- router ロジック
- 多数決ロジック
- SymPy ロジック
- sampling parameters
- model / adapter / data path
- 再学習
- min logprob 系
- deterministic CoT 再設計
- bit manipulation 強化
- submission.zip 中身変更
