# Day2 B1 実装仕様書 v1.0

## 0. 目的
Day2 で実装するのは **B1: training-serving misalignment 修正のみ** である。  
対象コードは、Kaggle Notebook `original-nemotron-asymmetric-svd-26041602` を基準とする。

この仕様書の目的は、**何を B1 と呼ぶか** を固定し、Generator が **実装対象の取り違えを起こさない** ようにすることである。

## 1. B1 の定義
この Day1 コードにおける **training-serving misalignment** とは、以下を指す。

> **学習時に成立していた adapter の前提と、Kaggle 上の serving 時に vLLM + surgery 後 adapter を読む前提が一致していない可能性がある状態**

このコードでは、adapter は serving 前に以下の再構成を受けている。

- `adapter_config.json` の書き換え
- key rename
- expert unfuse
- `gate_proj` / `x_proj` の `in_proj` への merge
- rank compression
- surgery 後 safetensors 再保存

したがって B1 は、**surgery 後 adapter の実体と serving 時に参照される adapter metadata / module declaration / inference 設定を一致させる修正** に限定する。

## 2. B1 の仮説
B1 の仮説は次のとおり。

> **Day1 コードでは、adapter の重み本体は serving 用に再構成されているが、adapter_config / module 宣言 / serving 前提の一部が training 側の名残を持つ可能性がある。これを serving 実体に揃えることで、A固定基準に対して悪化を防ぎ、比較可能な改善候補にできる。**

これは精度向上そのものを保証する仮説ではない。  
Day2 Run1 では、**比較可能な 1変数修正として成立すること** が目的である。

## 3. この仕様での 1変数定義
今回の **1変数** は、次の 1 本だけである。

> **surgery 後 adapter 実体に合わせて serving metadata / serving load 前提を整合させる修正**

複数ファイルに変更が入っても、**意味上の変更変数は 1 つ** である。  
その 1 つとは、**serving alignment** である。

## 4. 変更対象
Generator が変更してよい対象は、以下の範囲に限定する。

### 4.1 許可する変更
1. `adapter_config.json` の **serving 整合化**
2. surgery 後 tensor 名と config 上の target module 宣言の **一致確認と修正**
3. serving 時に必要な metadata の **補正**
4. surgery 後 adapter を vLLM が読む前提に合わせた **軽微な load 前処理**
5. Day2 evidence 採取コードの追加
6. Day2 成果物 `reports/day2/*.md` の更新

### 4.2 実装対象ファイル
- 主対象: `original-nemotron-asymmetric-svd-26041602` の GitHub 反映版
- 補助対象:
  - `kaggle/original-nemotron-asymmetric-svd-26041602.py`
  - `reports/day2/change_manifest.md`
  - `reports/day2/experiment_result.md`
  - `reports/day2/submission_asset_check.md`
  - `reports/day2/adoption_decision.md`

## 5. 変更禁止
以下は **B1 ではない** ので禁止する。

1. router ロジック変更
2. 多数決ロジック変更
3. SymPy / deterministic solver 強化
4. sampling parameter の変更
   - `temperature`
   - `top_p`
   - `n`
   - `max_tokens`
   - `max_num_seqs`
   - `gpu_memory_utilization`
   - `max_model_len`
5. model path / adapter path / data path の変更
6. base model の変更
7. 再学習
8. min logprob 系施策
9. deterministic CoT 再設計
10. bit manipulation 強化
11. Day1 提出資産部の無関係な変更
12. submission.zip の中身を増減させる変更
13. B1 と無関係なリファクタ

## 6. Day1 コードに対する具体的な B1 実装方針
以下を **B1 の具体内容** として固定する。

### 6.1 adapter_config の serving 化
Day1 コードでは、surgery 前後で `adapter_config.json` が書き換えられている。  
ここで Generator が確認・必要なら修正してよい項目は以下である。

- `target_modules`
- `inference_mode`
- LoRA serving に必要な整合項目
- surgery 後 tensor 群と config 記述の一致

**最優先確認項目**
- `inference_mode=False` が serving 時に不整合なら、**serving 用に true 相当へ補正**
- `target_modules` が surgery 後の実 tensor 名と食い違う場合、**surgery 後実体に合わせて補正**

### 6.2 surgery 後 tensor 名と config 宣言の一致確認
このコードでは、以下の surgery が入っている。

- `trained_adapter_key_rename`
- expert 展開
- `gate_proj` / `x_proj` → `in_proj` merge

したがって Generator は、**最終的に保存される `adapter_model.safetensors` の key 群** と、  
**`adapter_config.json` の target_modules 宣言** が一致しているかを確認し、不一致なら修正する。

### 6.3 serving 前提の明文化
Generator は実装時に、B1 を次の文で `change_manifest.md` に明記すること。

> 今回の変更は、training 時の adapter 前提ではなく、surgery 後 adapter 実体に合わせて serving metadata / serving 前提を整合化した 1変数変更である。

## 7. B1 実装の最小差分
Generator は、なるべく以下の順で最小差分を選ぶこと。

1. `adapter_config.json` の整合化のみで直るならそれを採用
2. 不足する serving metadata 補正を追加
3. 必要最小限の pre-load 整形を追加
4. それでも足りない場合のみ、surgery 出力側の metadata 生成処理を補正

**禁止**
- 推論ロジック本体の改造
- ルータ分岐の追加
- 生成戦略変更

## 8. Generator が実装時に残すべき記録
### 8.1 `reports/day2/change_manifest.md`
必ず以下を書く。

- baseline SHA: `39f4bed90392567517b606d1301ae1c36a86a97c`
- modified SHA
- changed variable count = 1
- changed variable = `B1 training-serving misalignment 修正`
- **どの config / metadata / serving 前提を修正したか**
- **変更していない項目一覧**

### 8.2 `reports/day2/experiment_result.md`
必ず以下を書く。

- 実行環境 = Kaggle
- 比較対象 = A固定基準
- 実装対象 = `original-nemotron-asymmetric-svd-26041602`
- 観測結果
- A比で悪化したか否か
- 0.86超え方向の根拠有無

### 8.3 `reports/day2/submission_asset_check.md`
必ず以下を書く。

- `submission.zip` 生成可否
- path
- size
- file list
- PASS / FAIL / BLOCKED

### 8.4 `reports/day2/adoption_decision.md`
必ず以下を書く。

- `ADOPT / REJECT / HOLD`
- 判定理由
- A比で悪化していないか
- 次に進む条件

## 9. Day2 evidence 仕様
Generator は、Kaggle 実行後に以下を採取できる状態にすること。

- `/kaggle/working/day2_evidence.json`
- `/kaggle/working/day2_evidence.md`

最低限の必須項目は以下である。

- `timestamp_utc`
- `source_of_truth`
- `baseline_sha`
- `experiment_scope`
- `one_variable_rule`
- `submission_zip.path`
- `submission_zip.exists`
- `submission_zip.status`
- `submission_zip.size_bytes`
- `submission_zip.file_count`
- `submission_zip.file_list`
- `submission_assets_preserved`
- `comparable_against_baseline`
- `worse_than_baseline`
- `evidence_for_gt_086`
- `provisional_verdict`
- `notes`

## 10. 実装完了条件
Generator は、以下を満たした場合のみ Step 2 完了とする。

1. 実装対象コードに **B1 本体** が入っている
2. evidence 追加だけで終わっていない
3. 変更理由が **serving alignment** として説明できる
4. 変更禁止項目に触れていない
5. GitHub 正本 `planner/day1-logprob` に push 済み
6. remote HEAD SHA を報告できる
7. `READY FOR KAGGLE` または `NOT READY` を返せる

## 11. NOT READY 条件
以下のどれかなら `NOT READY` とする。

1. B1 の具体修正が入っていない
2. evidence 追加だけで終わっている
3. router / sampling / SymPy 等に手を入れている
4. target_modules / inference_mode / serving metadata の整合修正内容を説明できない
5. 1変数性を示せない

## 12. Generator への正式依頼文
そのまま渡せる形で固定する。

```text
Day2 Step 2 を再実行してください。

実装対象:
- original-nemotron-asymmetric-svd-26041602
- 作業ブランチ: planner/day1-logprob
- Baseline SHA: 39f4bed90392567517b606d1301ae1c36a86a97c

今回実装する B1 は、Day1 コードに対する training-serving misalignment 修正のみです。
意味上の1変数は「surgery 後 adapter 実体に合わせて serving metadata / serving 前提を整合化すること」です。

実装してよいこと:
1. adapter_config.json の serving 整合化
2. surgery 後 tensor key と target_modules 宣言の整合化
3. serving 時に必要な metadata / load 前提の最小補正
4. day2_evidence.json / day2_evidence.md の採取コード
5. reports/day2/*.md 更新

実装してはいけないこと:
- router 変更
- 多数決変更
- SymPy 変更
- sampling parameter 変更
- model / adapter / data path 変更
- 再学習
- min logprob 系
- deterministic CoT 再設計
- bit manipulation 強化
- submission.zip 中身変更
- B1 と無関係なリファクタ

必須:
- evidence 追加だけで完了扱いしないこと
- どの config / metadata / serving 前提を直したかを change_manifest.md に明記すること
- remote HEAD SHA を返すこと
- 最終的に READY FOR KAGGLE / NOT READY を返すこと
```

## 13. 再発防止のための指示文追記
今後の取り違え防止として、以下を Day2 Generator 指示文に追記する。

```text
実装対象コードが指定されていても、B1 の具体変更仕様が未定義なら着手してはならない。
evidence 追加だけで B1 実装完了と見なしてはならない。
Step 2 完了の必須条件は、B1 本体の実装が入っていることである。
```

## 14. PM 結論
今回の分担は以下で固定する。

- **PM**: B1 実装仕様の定義
- **Generator**: 上記仕様どおりのコード実装・push
- **ユーザー**: 必要時に対象コード提供と Kaggle 実行

次に行うべきことは、  
**この仕様書をそのまま Generator に渡して Step 2 をやり直させること** である。
