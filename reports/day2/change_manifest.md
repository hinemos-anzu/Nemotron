# Day2 Change Manifest

- baseline_sha: 39f4bed90392567517b606d1301ae1c36a86a97c
- modified_sha: 9295ccac05dcbde28aa606a9d8e15d773515a343
- changed_variable_count: 1
- changed_variable: B1 training-serving misalignment 修正

## 1変数明示
- 今回の意味上の1変数は、surgery 後 adapter 実体に合わせて serving metadata / serving 前提を整合化すること。

## 変更内容要約
1. `kaggle/original-nemotron-asymmetric-svd-26041602.py` に B1 本体を実装。
2. `reconcile_serving_metadata()` を追加し、`adapter_config.json` の `target_modules` を surgery 後 tensor key から再構成。
3. `inference_mode` を serving 用に `True` へ補正。
4. `serving_alignment.json` を出力し、before/after を記録。
5. `WORKING_ADAPTER_DIR` に adapter 実体がない場合、`ADAPTER_PATH` / Kaggle input 既定パスから最小コピーして整合化処理を継続。
6. adapter 実体が依然不足する場合は例外停止せず `status=BLOCKED` を返し、evidence に理由を記録。
7. `day2_evidence.json` / `day2_evidence.md` 採取コードを維持。

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
- submission.zip 中身
