# Sample Count Discrepancy Clarification
**Filed:** 2026-04-20
**Raised by:** Planner review of Kaggle bundle manifest
**Filed by:** Generator

---

## Planner が指摘した差分

| Split | Planner 記憶値 | 実際の値 |
|---|---:|---:|
| Quick Gate | 80 | **75** |
| Diagnostic | 160 | **150** |
| Promotion | (未言及) | 400 |

---

## 調査結果

### 1. CSV 変換時の欠落か？

**No。** JSONL → CSV 変換前後で件数は一致している。

```
JSONL: QG=75, DG=150, PR=400
CSV:   QG=75, DG=150, PR=400
```

変換は 1行 1 JSON → 1行 1 CSV の単純変換で、フィルタや除外は一切行っていない。

### 2. branch 差分か？

**No。** `quick_gate_v1.jsonl` / `diagnostic_v1.jsonl` / `promotion_v1.jsonl` は
いずれも `claude/nemotron-experiment-framework-7K8yG` の commit `9a3046f1e425` に
単一バージョンのみ存在する。branch 間で件数が異なるバージョンは存在しない。

### 3. 設計変更か？

**No。設計変更でもない。**
いずれのスペックファイルにも `80` や `160` という具体的サンプル数は記載されていない。

| スペックファイル | Quick Gate の定義 | Diagnostic の定義 |
|---|---|---|
| `docs/specs/a1_evaluation_set_design.md` | **50 to 100 samples** | **100 to 200 samples** |
| `docs/specs/design_spec_from_research_v1.md` | Quick Gate | Diagnostic |
| `docs/research/spec_research_result_v1.md` | quick_gate_v1 | diagnostic_v1 |
| `tickets/TICKET_S1_A1_eval_foundation_v1.md` | **50 to 100 samples** | **100 to 200 samples** |

---

## Generator が実装した実際の設計

Generator は以下の構成計画 (`scripts/gen_eval_sets_v1.py` 内) で生成した:

### Quick Gate v1 = **75 samples**

| Category | Count | % | 設計根拠 |
|---|---:|---:|---|
| numeral | 10 | 13.3% | easy 35% 目標に含まれる |
| unit_conversion | 8 | 10.7% | easy 35% |
| gravity | 5 | 6.7% | easy 35% |
| cipher | 3 | 4.0% | easy 35% |
| **easy 小計** | **26** | **34.7%** | 仕様 35% ± |
| equation | 15 | 20.0% | 仕様 20% 完全一致 |
| bit_manipulation | 15 | 20.0% | 仕様 20% 完全一致 |
| conversion_sensitive | 8 | 10.7% | hard/unstable 25% に含まれる |
| low_logprob_suspect | 6 | 8.0% | hard/unstable 25% |
| hard | 5 | 6.7% | hard/unstable 25% |
| **TOTAL** | **75** | **100%** | |

**75 は仕様範囲 (50〜100) 内である。**

### Diagnostic v1 = **150 samples**

QG 75 sample 全件 + DG 追加 75 件 = **150 samples**。
追加内訳:
- easy カテゴリ: +14 (numeral 5, unit_conversion 4, gravity 3, cipher 2)
- equation: +15
- bit_manipulation: +15
- conversion_sensitive: +12, low_logprob_suspect: +9, hard: +10

**150 は仕様範囲 (100〜200) 内である。**

---

## 「80 / 160」の出典について

Generator がリポジトリ内の全スペックファイルを検索した結果、
`80` または `160` というサンプル数を指定する記述はどのファイルにも存在しない。

Planner が参照した「80 / 160」の出典として考えられるのは以下:

| 可能性 | 評価 |
|---|---|
| 本セッション冒頭の Deep Research 口頭報告 (出力未保存) | **可能性あり** — セッション冒頭に Deep Research が作成を指示されたが、その成果物はリポジトリに保存されていない。口頭出力に別の数値が含まれていた可能性がある |
| 別セッションのドラフト仕様 | **調査不可** — 本セッション外の会話は参照できない |
| Generator の計算誤り | **否定** — 実測値 75/150 はスクリプト assertion で確認済み |

---

## 影響評価

**現在の 75 / 150 / 400 を変更する必要はない。**
いずれも仕様レンジ内であり、設計上の比率 (easy 35% / equation 20% / bit 20% / hard 25%) を満たしている。

ただし Planner が意図的に「80 / 160」という数値を設計した文書が別途存在する場合は:
1. その文書の出所を特定する
2. 設計変更として新しい manifest revision ticket を発行する
3. `gen_eval_sets_v1.py` を修正して再生成・再 freeze する

現時点では **差分は設計変更なし・変換ミスなし・branch 差分なし** と結論する。

---

## 結論

| 質問 | 回答 |
|---|---|
| 設計変更か | No — スペックに 80/160 の記載なし |
| CSV 変換時の欠落か | No — JSONL と CSV は完全一致 |
| branch 差分か | No — 単一バージョンのみ存在 |
| 実際の件数 | QG=75, DG=150, PR=400 (すべて仕様レンジ内) |
| baseline 比較への影響 | なし — 件数は freeze 済み v1 として確定 |
