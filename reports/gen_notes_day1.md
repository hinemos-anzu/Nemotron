# Day1 Generation Notes — NVIDIA Nemotron Model Reasoning Challenge

**Date:** 2026-04-18
**Author:** Evaluator (claude/improve-nemotron-score-ZGuqa)
**Purpose:** Day1 の作業メモ・観察・注意点の記録

---

## 状態確認

### リポジトリ初期状態

- `git log` の最初のコミット: `b356366 Initial commit`
- Day1 開始時点でのファイル: `LICENSE`, `README.md` のみ
- 他ブランチ (`evaluator/day1-logprob`, `planner/day1-logprob`) も同一コミットから分岐

### 結論
Day1 開始時点では成果物が **ゼロ** であった。
sprint_contract, decision_log, runbook, manifest, inventory, eval の全文書を本Day1で新規作成する。

---

## ベースA 資産についての観察

### 中核資産（Code-level）

ベースAコードは Kaggle Notebook として存在し、以下の処理を順番に行う。

```
1. adapter変換
   - LoRA adapter を base model にマージ
   - merged_model/ に保存

2. Offline Asymmetric SVD Surgery
   - 対象: attention weight (q_proj, k_proj, v_proj, o_proj) および MLP weight
   - 手法: truncated SVD → 低ランク近似 → asymmetric correction
   - 出力: svd_model/

3. key rename
   - Nemotron 推論サーバが期待するキー名フォーマットに変換
   - 例: model.layers.X.self_attn.q_proj.weight → transformer.h.X.attn.q_attn.weight (仮)

4. expert unfuse
   - MoE (Mixture of Experts) レイヤの expert weights をアンフューズ
   - fused_experts.weight → experts.{i}.weight 形式に展開

5. gate_proj + x_proj → in_proj 統合
   - 2つの projection を concat して1テンソルに結合
   - 推論サーバの in_proj 期待形式に適合

6. submission.zip 生成
   - 変換済みモデルを zip パッケージ化
   - Kaggle 提出形式に準拠
```

### 補助資産（主線対象外）

```
- router ロジック: MoE の top-k routing、load balancing
- 多数決 (majority voting): 複数推論結果の集約
- SymPy系: 数式解法・symbolic computation
```

これらはスコア 0.86 に貢献しているが、改善主線からは外す（D-002参照）。

---

## 注意点・リスク

### PATH 依存リスク
- Kaggle Notebook は `/kaggle/input/` をデータ読み込み元とする
- submission.zip の出力先は `/kaggle/working/` 固定
- adapter・base model のパスが hardcode されている可能性あり → runbook で明示化が必要

### Dataset 依存リスク
- base model (Nemotron) が Kaggle Dataset として存在する必要あり
- adapter checkpoint が Kaggle Dataset または Notebook output として存在する必要あり

### 実行順序の厳格化
- adapter変換 → SVD surgery → key rename → unfuse → in_proj統合 → zip
- この順序を崩すと submission.zip が破損する

---

## Day1 スコープ確認

| 項目 | 対象 | 備考 |
|------|------|------|
| adapter変換再現条件明文化 | YES | manifest + runbook |
| SVD surgery 再現条件明文化 | YES | manifest + runbook |
| submission.zip 生成確認 | YES | runbook に記載 |
| スコア改善実験 | NO | Day2以降 |
| min logprob 分析 | NO | Day2以降 |
| router/多数決/SymPy改善 | NO | 主線外 |

---

## 次ステップ（Day2へ）

Day1 が PASS であれば、Day2は以下から開始する。

1. **training-serving misalignment 修正** を最初の1変数として実験
   - 推論時の temperature / sampling parameter がトレーニング時と一致しているか確認
   - 修正後に submission.zip を再生成し、A比スコアを比較
