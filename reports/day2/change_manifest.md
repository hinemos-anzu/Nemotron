# Day2 Change Manifest

- baseline_sha: 39f4bed90392567517b606d1301ae1c36a86a97c
- modified_sha: d0da87a7d2d9893d3509d61a956f2905490b112f
- changed_variable_count: 1
- changed_variable: B1 training-serving misalignment 修正

## 変更内容要約
- Day2 では B1 のみを実験変数として定義した。
- ただし Kaggle 実行証跡が未取得のため、B1 変更の実行結果は未確定。

## 変更していない施策一覧
- B2 以降の全施策
- min logprob 分析本体
- low-minlogprob 再重点学習
- deterministic chain-of-thought 再設計
- bit manipulation 強化
- router 改善
- 多数決改善
- SymPy 改善
