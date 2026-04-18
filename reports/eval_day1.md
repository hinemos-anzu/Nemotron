# Evaluation Report — Day1
# NVIDIA Nemotron Model Reasoning Challenge

**Date:** 2026-04-18
**Evaluator:** claude/improve-nemotron-score-ZGuqa
**Audit scope:** Day1 成果物が「Aの提出資産を比較基準として固定する」目的を満たしているか

---

## Verdict

```
SOFT FAIL
```

**理由:** Day1 の文書体系は完成しており比較基準の骨格は固定されたが、
**Kaggle上での実際の実行確認（submission.zip の生成・提出）が未完了**であるため
PASS ではなく SOFT FAIL とする。
文書完全性は十分であり、Day2 は条件付きで開始できる。

---

## 1. What in A Must Be Preserved（Aで守るべきもの）

以下の6つは **変更禁止の中核資産** である。

| # | 資産 | 変更禁止理由 |
|---|------|-------------|
| 1 | adapter変換 | SVD surgeryの入力となる。消えると後続全滅 |
| 2 | Offline Asymmetric SVD Surgery | スコア0.86の主要因。rank_ratio=0.5, asymmetric=Trueを固定 |
| 3 | key rename | 推論サーバがロードできなくなる |
| 4 | expert unfuse | MoE推論に必須。num_experts=8固定 |
| 5 | gate_proj + x_proj → in_proj 統合 | 推論サーバが期待するテンソル形式 |
| 6 | submission.zip 生成 | Kaggle提出の最終成果物 |

これらのいずれかが欠けた場合、または処理順序が変わった場合は **即時 BLOCKED** とする。

---

## 2. What in A Should NOT Be Mainline（主線から外すべきもの）

| 資産 | 主線除外の根拠 |
|------|--------------|
| router ロジック | MoE routing の調整は精度改善主線と無関係 (D-002) |
| 多数決 (majority voting) | 1実験1変数原則に反する後処理 (D-002) |
| SymPy系 | Symbolic solver はモデル本体改善に非直結 (D-002) |

これらは Aの中に存在するが、**改善実験の変数として扱わない**。
Day2以降の実験でこれらを触れる場合は別実験として記録し、主線の実験と混在させないこと。

---

## 3. Whether Day1 Fixed the Baseline（Day1で比較基準は固定されたか）

### 固定された項目 ✓

- [x] **中核資産が明文化された** — `artifacts/day1_baseline_manifest.md` に6資産を列挙
- [x] **補助資産が切り分けられた** — router/多数決/SymPyを主線外と明記
- [x] **Kaggle前提が明文化された** — PATH 5変数、Dataset定義、実行順6ステップ
- [x] **比較基準スコアが固定された** — 0.86 を manifest に記録
- [x] **BLOCKED条件が整理された** — runbook に6条件を記載
- [x] **Day1スコープが守られた** — 精度改善実験なし、文書のみ

### 未固定の項目 △

- [ ] **Kaggle実行確認が未完了** — submission.zip を実際に生成・提出していない
- [ ] **submission.zip の MD5 未記録** — manifest に記録予定と書かれているが空欄
- [ ] **Notebook URL 未記録** — ベースAの実際の Notebook リンクが未記載

### 評価

文書レベルでの比較基準固定は **完了**。
実環境（Kaggle）での確認は **未完了**（SOFT FAILの主因）。

---

## 4. Whether Day2 Can Start（Day2を開始できるか）

```
条件付き YES
```

### 条件

1. Kaggle Notebook でベースA の runbook を実行し、submission.zip が生成できること
2. submission.zip の MD5 を `artifacts/day1_baseline_manifest.md` に記録すること
3. 提出 Notebook URL を manifest に記録すること

上記3条件を満たしてから Day2 の Exp-B1（training-serving misalignment修正）を開始する。
条件を満たさずに Day2 に進んだ場合、A比評価が不可能となる。

---

## 5. Blockers（ブロッカー）

| # | ブロッカー | 深刻度 | 対処方法 |
|---|-----------|--------|----------|
| B1 | Kaggle実行確認が未完了 | MEDIUM | runbook を Kaggle Notebook で実行する |
| B2 | submission.zip MD5 未記録 | LOW | 実行後に manifest を更新する |
| B3 | Notebook URL 未記録 | LOW | 実行後に manifest を更新する |
| B4 | Dataset バージョン未確認 | MEDIUM | Kaggle で nemotron-base v1 の存在を確認する |
| B5 | PEFT バージョン未固定 | MEDIUM | Notebook requirements に peft バージョンを pin する |

**CRITICAL BLOCKER なし** — Day1文書は揃っており、即時実行可能な状態。

---

## 6. Recommendation（推奨事項）

### Day1 残作業（本日中）

1. **Kaggle Notebook を開き、day1_kaggle_runbook.md の Step1〜Step6 を実行する**
2. submission.zip が生成されたら MD5 を記録する
3. `artifacts/day1_baseline_manifest.md` の「比較基準スコア」欄を更新する
4. PEFT, transformers, PyTorch のバージョンを Notebook requirements に pin する

### Day2 開始条件の確認

Day2 開始前に以下を確認する。
- [ ] manifest の MD5 が記録済み
- [ ] Kaggle 提出が成立している（or 少なくとも submission.zip が生成済み）
- [ ] Day1 SOFT FAIL の残作業が完了している

---

## 監査観点別評価

| 観点 | 評価 | 根拠 |
|------|------|------|
| Aの中核資産が明確か | PASS | manifest に6資産を明記、各資産の入出力・固定値を記載 |
| Aの補助資産が切り分けられているか | PASS | router/多数決/SymPyをD-002で主線除外と決定、全文書に反映 |
| Kaggle前提が明文化されているか | PASS | PATH/dataset/model/tokenizer/実行順すべて runbook に記載 |
| Day2以降の比較基準として使えるか | SOFT PASS | 文書は完成、Kaggle実行確認のみ未完了 |
| BLOCKED条件が明確か | PASS | runbook に6条件のBLOCKEDリスト、manifest に変更禁止5事項 |

---

## One-Line Verdict

```
SOFT FAIL — 文書基準は固定完了、Kaggle実行確認（submission.zip生成）のみ未完了。
```

## Why

Day1の目的は「Aの比較基準固定」であり、その文書体系（sprint_contract, decision_log,
gen_notes, runbook, manifest, inventory）は Day1 内で完成した。
中核資産6点・補助資産3点の切り分け・BLOCKED条件・PATH定義・実行順すべてが明文化されている。
唯一の未完了事項は **Kaggle Notebook上での実際の実行確認（submission.zip生成）** であり、
この確認なしには Day2 の A比評価が実行不能となるため PASS ではなく SOFT FAIL とする。

## What Day2 Should Target First

**Exp-B1: training-serving misalignment 修正**
— Kaggle実行確認が完了次第、即座に開始する。
推論時の temperature / sampling パラメータがトレーニング時設定と一致しているかを確認し、
misalignment を修正した状態で submission.zip を再生成して A比（0.86）と比較する。
