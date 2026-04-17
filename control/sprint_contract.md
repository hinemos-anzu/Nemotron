# Sprint Contract

## Sprint title
Day1 — min logprob 可視化基盤の整備

## Date / branch / owner
- Date: 2026-04-17
- Branch: `planner/day1-logprob`
- Owner: Planner
- Target executor: Generator

---

## Goal

**今回の目的は1つだけ:**
全訓練サンプルに対して per-sample min token logprob を計算・保存し、
「難しいサンプル（low-minlogprob）」の分布をカテゴリ別に可視化できる状態にする。

この基盤がなければ、次スプリント（reweighting）の設計根拠が存在しない。
可視化基盤の整備はDay1で完結させる。Day2以降には持ち越さない。

---

## In scope

1. `scripts/analyze_logprob.py` の設計・実装
   - ベースモデル（またはcheckpoint-0 SFTモデル）をロードしてforwardパスを実行
   - per-sample の min token logprob を計算（生成ではなくteacher-forcingで計算）
   - 出力: `outputs/logprob_scores.csv`（カラム: `sample_id, category, min_logprob, mean_logprob, seq_len`）
2. `scripts/visualize_logprob.py` の設計・実装
   - `logprob_scores.csv` を読んで以下を出力
     - カテゴリ別 min_logprob ヒストグラム（PNG）
     - 全体の percentile 分布（P10/P25/P50/P75/P90 の数値テーブル）
     - top-10% low-minlogprob サンプルのリスト（`outputs/hard_samples.csv`）

---

## Out of scope

以下は今スプリントでは**一切触らない**。Generatorは実装しないこと。

- fine-tuning / SFT の実行
- reweighting・損失重み付け
- CoT テンプレートの変更
- bit manipulation サンプルへの特別処理
- 推論スクリプト（inference.py 等）の変更
- SymPy / 外部ツール連携
- モデルのアーキテクチャ変更
- adapter の変換・マージ
- Kaggle への提出

---

## Inputs

### Planner 着手前確定条件（Generator に渡す前に Planner が自ら確認・記入すること）

以下3項目がすべて埋まるまで、Generator へ本スプリントを渡してはならない。
未確定のまま渡すことは BLOCKED 扱いとする。

| Item | 確定値 | 確定方法 | 状態 |
|------|--------|---------|------|
| 訓練データセットのパス | *(Planner記入)* | Kaggle dataset / local path を Planner が直接確認 | **PENDING** |
| ベースモデル or checkpoint のパス | *(Planner記入)* | HuggingFace model ID or local path を Planner が直接確認 | **PENDING** |
| カテゴリラベルの有無と列名 | *(Planner記入)* | データセットの schema を Planner が直接確認 | **PENDING** |

**PENDING が1つでも残っている場合、このスプリントは BLOCKED である。**
Generator に渡してよいのは、上表の「状態」欄が3つとも **CONFIRMED** になったときのみ。

カテゴリラベルが存在しない場合: BLOCKED を宣言し、ラベル付与方針を決定する mini-sprint を先行させる。

**Generatorへの指示:**
上表が CONFIRMED になった状態で渡される。パスは Planner が確定済みの値のみ使うこと。
`real` と `dummy` を混在させたまま実装してはならない。

---

## Required outputs

スプリント終了時に存在すること:

| ファイル | 必須条件 |
|---------|---------|
| `scripts/analyze_logprob.py` | 実行可能、引数ドキュメント付き |
| `scripts/visualize_logprob.py` | 実行可能、引数ドキュメント付き |
| `outputs/logprob_scores.csv` | 全訓練サンプル分（またはサブセット使用時はその旨をREADMEに記録） |
| `outputs/hard_samples.csv` | min_logprob が全体下位10%のサンプル一覧 |
| `outputs/logprob_hist_by_category.png` | カテゴリ別ヒストグラム |
| `outputs/logprob_percentiles.txt` | P10/P25/P50/P75/P90 の数値テーブル |
| `control/sprint_contract.md` | 本ファイル（更新があれば記録） |
| `control/decision_log.md` | DEC-001 の evidence_observed 欄を埋めること |

---

## Success criteria

以下を**すべて**満たすとき PASS とする。条件の1つでも欠ければ FAIL。

1. `logprob_scores.csv` の行数 ≥ 全訓練サンプル数の95%（欠損5%以内）
2. `logprob_percentiles.txt` に P10/P25/P50/P75/P90 の数値が存在する
3. `hard_samples.csv` の行数 = `logprob_scores.csv` 行数の10%（±1行許容）
4. カテゴリ別に min_logprob の中央値（P50）の差が **少なくとも1カテゴリペアで有意**（差 > 0.5 nats）
5. `analyze_logprob.py` が `--help` で引数一覧を出力できる
6. スクリプトが `dummy` / `placeholder` 変数を本番パスとして使っていない

---

## Failure branches

以下のいずれかが起きた場合の対応を Evaluator が判定する:

### FB-1: 計算速度がボトルネックになりDay1内に全サンプルをカバーできない
- 対応: 訓練セットから stratified sampling で **1,000サンプル** のサブセットを使う
- 条件: カテゴリ比率がオリジナルと±5%以内であること
- 記録: `outputs/logprob_scores.csv` ヘッダに `# subset=1000, seed=42` を明記
- 判定: Evaluatorがサブセット使用を確認したうえで PASS 可（ただし次スプリントで全量再計算を予定に入れる）

### FB-2: min_logprob とカテゴリの相関が全く観測されない（全カテゴリでP50差 < 0.1 nats）
- 対応: `mean_logprob` と `entropy（= -mean_logprob）` を追加指標として計算し再描画
- 条件: 追加指標で差異が観測できるか否かを Evaluator が報告
- 判定: 追加指標でも差異なし → DEC-001を REJECT し、施策2（reweighting）の根拠を再検討

### FB-3: カテゴリラベルが訓練データに存在しない
- 対応: Generator は**実装を止めて** Planner に即報告する（silent fallback 禁止）
- Planner対応: カテゴリラベルの付与方針を別途決定し、新たな mini-sprint を発行する
- 判定: FB-3発生時はDay1スプリントを BLOCKED として記録

---

## Stop conditions

以下が発生した場合、Generatorは実装を中断して Planner に報告すること:

- モデルのロードがメモリ不足でできない
- 訓練データへのアクセス権がない
- カテゴリラベルが存在しない（FB-3）
- logprob計算結果がすべて NaN または inf になる

---

## Notes for Generator

- **real と dummy を混在させない**: 本番パスが確定するまでスタブを書いてよいが、スタブには `raise NotImplementedError("real path required")` を入れること
- **引数はすべて CLI から渡す**: ハードコードしない（モデルパス・データパス・出力ディレクトリ）
- **logprob の計算方法**: teacher-forcing（正解トークン列に対するforwardパス）で行う。生成サンプリングは使わない
- **min logprob の定義**: 1サンプル内の全トークンの logprob のうち最小値（= 最も確信が低いトークン）
- **カテゴリラベルが欠損している場合**: `category="unknown"` として処理せず、FB-3として Planner に報告する
- **出力CSVのエンコーディング**: UTF-8、ヘッダ1行目、区切り文字はカンマ

---

## Notes for Evaluator

審査時に確認すること:

1. `logprob_scores.csv` の `sample_id` がユニークかつ訓練セットIDと一致するか
2. `hard_samples.csv` のカットオフがP10（下位10%）であることを数値で確認する
3. `logprob_hist_by_category.png` がカテゴリ別に描画されており、全カテゴリが含まれているか
4. スクリプト内に `dummy` / `placeholder` / `TODO` のまま残っているコードがないか
5. `logprob_percentiles.txt` の数値が `logprob_scores.csv` から再現できるか（抽出して確認）
6. Success criteria の6条件をすべてチェックリスト形式で報告すること

Evaluatorは「成功条件が全部満たされているか」を判定するのみ。
スクリプトの可読性・コードスタイルの評価は今スプリントの対象外。
