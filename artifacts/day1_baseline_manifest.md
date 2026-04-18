# Day1 Baseline Manifest — NVIDIA Nemotron Model Reasoning Challenge

**Date:** 2026-04-18
**Baseline Score:** 0.86
**Status:** FIXED (Day1比較基準)

---

## 概要

このファイルはベースA（スコア0.86）の提出資産を比較基準として固定するためのマニフェストである。
Day2以降のすべての改善は本マニフェストを基準として A比で評価する。

---

## 中核資産（保護対象・変更禁止）

### 1. adapter変換

| 項目 | 値 |
|------|-----|
| 処理 | LoRA adapter → base model マージ |
| 入力 | `BASE_MODEL_PATH`, `ADAPTER_PATH` |
| 出力 | `MERGED_OUTPUT_PATH/` (safetensors形式) |
| ライブラリ | `peft` (バージョン固定必須) |
| 重要度 | CRITICAL — これがなければSVD surgeryが動かない |

### 2. Offline Asymmetric SVD Surgery

| 項目 | 値 |
|------|-----|
| 処理 | 重み行列のSVD分解 → 低ランク近似 → 非対称補正 |
| 入力 | `MERGED_OUTPUT_PATH/` |
| 出力 | `SVD_OUTPUT_PATH/` |
| 対象レイヤ | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| rank_ratio | 0.5 (ベースA固定値) |
| asymmetric | True (ベースA固定値) |
| 重要度 | CRITICAL — スコア0.86の主要因 |

### 3. key rename

| 項目 | 値 |
|------|-----|
| 処理 | checkpoint キー名を推論サーバ互換形式にリネーム |
| 入力 | `SVD_OUTPUT_PATH/` |
| 出力 | `SVD_OUTPUT_PATH/` (in-place または別ファイル) |
| マッピング | `NEMOTRON_KEY_MAP` 辞書（ベースAコードに定義） |
| 重要度 | CRITICAL — これがないと推論サーバがロードできない |

### 4. expert unfuse

| 項目 | 値 |
|------|-----|
| 処理 | MoE fused expert weights → individual expert tensors |
| 入力 | `SVD_OUTPUT_PATH/` |
| 出力 | `SVD_OUTPUT_PATH/` (experts.{i}.weight 形式) |
| num_experts | 8 (ベースA固定値) |
| 重要度 | CRITICAL — MoE推論に必須 |

### 5. gate_proj + x_proj → in_proj 統合

| 項目 | 値 |
|------|-----|
| 処理 | 2つの projection テンソルを dim=0 で concat → in_proj |
| 入力 | `gate_proj.weight`, `x_proj.weight` |
| 出力 | `in_proj.weight` |
| 重要度 | CRITICAL — 推論サーバの期待するテンソル形式 |

### 6. submission.zip 生成

| 項目 | 値 |
|------|-----|
| 処理 | `SVD_OUTPUT_PATH/` 全体を zip パッケージ化 |
| 入力 | `SVD_OUTPUT_PATH/` |
| 出力 | `/kaggle/working/submission.zip` |
| 圧縮 | ZIP_DEFLATED |
| 重要度 | CRITICAL — Kaggle提出の最終成果物 |

---

## 補助資産（主線対象外・変更しない）

| 資産 | 説明 | 除外理由 |
|------|------|----------|
| router ロジック | MoE top-k routing, load balancing | 精度改善主線と無関係 |
| 多数決 (majority voting) | 複数推論の集約 | 1実験1変数原則に反する |
| SymPy系 | Symbolic solver | モデル本体改善に非直結 |

---

## 実行環境（ベースA固定値）

| 環境変数 | 値 |
|---------|-----|
| 実行環境 | Kaggle Notebook |
| GPU | T4 x2 または A100 |
| Python | 3.10+ |
| PyTorch | 2.x |
| dtype | bfloat16 |
| seed | 42 |

---

## 比較基準スコア

| 指標 | 値 |
|------|-----|
| Kaggle public score | 0.86 |
| 提出日 | (記録予定) |
| 提出 Notebook | (記録予定) |
| submission.zip MD5 | (記録予定) |

> **Day2以降の判定:** 上記スコアを下回る変更は Reject する。

---

## 変更禁止事項

Day2以降、以下を**変更してはならない**。

1. `rank_ratio` の値 (0.5) — 変更する場合は別実験として記録する
2. `num_experts` の値 (8) — モデル固有のアーキテクチャ値
3. `NEMOTRON_KEY_MAP` の内容 — 変更すると推論サーバがロードできない
4. zip 生成のファイル構造 — Kaggle提出形式の要件
5. asymmetric SVD の有効/無効 — これがスコア0.86の核心

---

## Day2 実験計画（参考）

| 実験 | 変更点 | 判定基準 |
|------|--------|----------|
| Exp-B1 | training-serving misalignment 修正 | A比スコア >= 0.86 |
| Exp-B2 | min logprob 分析・閾値設定 | A比スコア >= 0.86 |
| Exp-B3 | low-minlogprob 再重点学習 | A比スコア >= 0.86 |
| Exp-B4 | deterministic CoT 再設計 | A比スコア >= 0.86 |
| Exp-B5 | bit manipulation 強化 | A比スコア >= 0.86 |
