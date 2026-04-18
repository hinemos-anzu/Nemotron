# Sprint Contract — NVIDIA Nemotron Model Reasoning Challenge

## Sprint: Day1
**Date:** 2026-04-18
**Status:** ACTIVE

---

## Goal

Day1 は精度改善日ではない。
Aの 0.86 再現基準と提出資産を固定し、Day2以降の改善を A比で判定できる状態にする。

---

## In Scope (Day1)

| タスク | 目的 |
|--------|------|
| ベースA中核資産の明文化 | adapter変換・SVD surgery・submission.zip生成の再現条件を文書化 |
| Kaggle runbook 作成 | path / dataset / model / tokenizer / 実行順を明確化 |
| day1_baseline_manifest 作成 | A比較基準として固定するアーティファクトを列挙 |
| eval_day1.md 作成 | Day1成果物の監査・Verdict発行 |

## Out of Scope (Day1)

- 改善策Bの本格投入
- min logprob 分析本体
- 再重点学習
- CoT再設計
- bit manipulation 強化
- router 改善
- 多数決改善
- SymPy改善

---

## ベースA 中核資産（保護対象）

1. **adapter変換** — LoRA adapter → base model マージ変換
2. **Offline Asymmetric SVD Surgery** — 重み行列のSVD分解による軽量化・非対称補正
3. **key rename** — チェックポイント内のキー名を推論サーバ互換形式にリネーム
4. **expert unfuse** — MoE層の expert weights をアンフューズ
5. **gate_proj + x_proj → in_proj 統合** — projection レイヤの結合
6. **submission.zip 生成** — Kaggle提出形式への最終パッケージング

## ベースA 補助資産（主線対象外）

- router ロジック
- 多数決 (majority voting) アンサンブル
- SymPy系 symbolic solver

---

## 成功条件

- [ ] Aの中核資産の再現条件が明文化される
- [ ] Kaggle上でAの提出資産部が成立する
- [ ] Day2以降の比較基準として固定できる
- [ ] 失敗時の BLOCKED 要因が明文化される

---

## Day2 開始条件

- eval_day1.md の Verdict が PASS または SOFT FAIL
- day1_baseline_manifest.md が確定している
- day1_kaggle_runbook.md が使える状態にある
