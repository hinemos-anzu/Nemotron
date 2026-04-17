# Decision Log

このファイルはスプリントごとの採否記録を蓄積する。
採否後に編集するのは Evaluator のみ。Planner が雛形を作成し、Evaluator が `evidence_observed` と `decision` 欄を埋める。

---

## DEC-001

| 項目 | 内容 |
|------|------|
| **Decision ID** | DEC-001 |
| **Sprint** | Day1 |
| **Date** | 2026-04-17 |
| **Hypothesis** | per-sample min token logprob を計算することで、訓練サンプルの「難しさ」がカテゴリ別に定量化でき、次スプリントの reweighting 設計根拠が得られる |
| **Why now** | reweighting を設計するにはどのサンプルが難しいかの観測が先行条件。基盤なしで reweighting の係数を決めることはできない |
| **Evidence expected** | - `logprob_scores.csv` が全訓練サンプルの95%以上をカバー<br>- カテゴリ別 P50(min_logprob) の差が少なくとも1ペアで > 0.5 nats<br>- `hard_samples.csv`（下位10%）のカテゴリ分布が偏っている（一様でない） |
| **Adoption rule** | Success criteria の6条件をすべて満たしたとき ADOPT。条件は sprint_contract.md に定義 |
| **Rejection rule** | 以下のいずれかで REJECT:<br>1. `logprob_scores.csv` の行数が全サンプルの95%未満（FB-1が解決されていない場合）<br>2. 全カテゴリで P50 差 < 0.1 nats（min_logprob がカテゴリ識別に無意味）<br>3. スクリプトに `dummy` / `placeholder` が本番パスとして混在している |
| **Follow-up if ADOPT** | Day2スプリント: `hard_samples.csv` を使った reweighting 設計に進む。weight係数の設計は Planner が別途定義 |
| **Follow-up if REJECT** | 理由によって分岐:<br>- min_logprob が無意味 → perplexity/entropy を代替指標として試すmini-sprint<br>- FB-3（カテゴリなし）→ カテゴリラベル付与方針を Planner が策定<br>- データアクセス不能 → インフラ確認を最優先タスクにする |
| **evidence_observed** | *(Evaluatorが記入)* |
| **decision** | *(Evaluatorが記入: ADOPT / REJECT / BLOCKED)* |
| **decided_by** | *(Evaluatorが記入)* |
| **decided_at** | *(Evaluatorが記入)* |
| **notes** | *(Evaluatorが記入)* |

---

## DEC-002（予定枠 — Day2）

| 項目 | 内容 |
|------|------|
| **Decision ID** | DEC-002 |
| **Sprint** | Day2（予定） |
| **Date** | 未定 |
| **Hypothesis** | low-minlogprob サンプルへの loss reweighting が easy サンプルの精度を維持しつつ hard サンプルの精度を改善する |
| **Why now** | DEC-001 ADOPT 後の次施策。基盤なしで設計できないため、Day1完了を前提とする |
| **Evidence expected** | *(Day2スプリント開始時に Planner が記入)* |
| **Adoption rule** | *(Day2スプリント開始時に Planner が記入)* |
| **Rejection rule** | *(Day2スプリント開始時に Planner が記入)* |
| **Follow-up if ADOPT** | *(Day2スプリント開始時に Planner が記入)* |
| **Follow-up if REJECT** | *(Day2スプリント開始時に Planner が記入)* |
| **evidence_observed** | *(Evaluatorが記入)* |
| **decision** | *(Evaluatorが記入)* |
| **decided_by** | *(Evaluatorが記入)* |
| **decided_at** | *(Evaluatorが記入)* |
| **notes** | DEC-001が REJECT された場合、このエントリは無効化されて mini-sprint が先行する |

---

## アーカイブポリシー

- 各 DEC エントリは一度 `decision` 欄が埋まったら編集しない（append-only）
- REJECT されたエントリも削除しない
- BLOCKED エントリは原因が解消されたあと再オープンせず、新しい DEC-XXX を発行する
