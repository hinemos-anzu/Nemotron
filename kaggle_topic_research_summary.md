# Kaggle NVIDIA Nemotron Model Reasoning Challenge — Topic Research Summary

**調査日**: 2026-06-09  
**調査対象**: Public LB 0.86 / 100位級 Golden Baseline からの精度向上トピック  
**競技URL**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge  
**調査手法**: 5方向並列WebSearch + 主要ソース直接検証 + adversarial事実確認

---

## 1. Executive Summary

### 0.86 Plateau に関する結論

**事実（複数ソース確認済み）**:
- 提出の多数が 0.85〜0.86 に集中している
- Progress Prize 優勝の Huikang (Tong Hui Kang) も cryptarithm を 8%→71% に改善した上で 0.86 に到達
- しかし **cryptarithm の min logprob → 0 は「メモリ化」のサイン** であり、汎化ではなかった（＝訓練セットを暗記しただけ）
- これが 0.86 plateau の核心的な原因と考えられる
- Public LB 上位スコア（0.89台）の存在から、0.86 突破は可能（ただしスコア未独立確認）

**推測（単一ソースまたは推論）**:
- 0.86 突破には cryptarithm の汎化的改善 + 他カテゴリの組み合わせが必要
- Equation numeric guess (baseline 15.4%) が未改善の隠れたボトルネックの可能性あり

### 精度向上が期待できる有力トピック Top 5

| 優先度 | トピック | 根拠の確実性 |
|--------|---------|-------------|
| 1 | **Min logprob genuine failure detection** — メモリ化 vs 汎化を正しく区別した上での選択的追加学習 | 事実（Huikang確認） |
| 2 | **Cryptarithm operator detection template改善** — concatenation vs reverse concatenation を正しく列挙・検定する reasoning template | 事実（Discussion確認） |
| 3 | **Equation numeric guess targeted training** — baseline 15.4% という低スコアカテゴリへの集中改善 | 事実（category solve rate確認） |
| 4 | **Held-out 10% validation** — 訓練セットの 10% を保持してLB提出前に汎化を検証 | 事実（Huikang推奨） |
| 5 | **Bit manipulation targeted synthetic data** — baseline 85.1% → さらなる改善余地あり | 推測（単一ソース） |

### 逆に除外すべきトピック Top 5

| 除外度 | トピック | 除外根拠 |
|--------|---------|---------|
| 1 | **Rank compression / SVD圧縮** | 過去実験で 0.85/0.84 悪化（ユーザー確認済み） |
| 2 | **Reverse concatenation as default operator** | Public LB overfit の典型例として Discussion で明示（Private LB 悪化リスク大） |
| 3 | **lm_head / attention module swap** | 過去実験で 0.85 悪化（ユーザー確認済み） |
| 4 | **大量 synthetic data 一括追加（solver検証なし）** | 誤教師データリスク + 0.86 突破直結なし |
| 5 | **Adapter fusion 全面探索** | 0.86 維持には寄与しても突破の主因でない |

### Golden Baseline を壊さないための注意点

1. `adapter_config.json` + `adapter_model.safetensors` の構造は変更しない
2. LoRA rank ≤ 32 はコンペルール（超えると失格）
3. target_modules 変更はserving側ロードに影響 → 変更禁止
4. MoE routing の数値差（training-serving mismatch）を把握した上でのみ変更
5. min logprob → 0 はメモリ化の可能性があるため、必ず held-out validation で確認
6. Public LB の改善がcryptarithmのoperator仮定変更に起因していないか確認すること

---

## 2. Evidence Table

| topic | source_type | title | author | date_or_version | public_score | URL | evidence_summary | fact/inference/unconfirmed | relevance_to_0.86 | actionability | risk |
|-------|------------|-------|--------|-----------------|--------------|-----|-----------------|--------------------------|-------------------|---------------|------|
| 0.86 plateau / cryptarithm | Discussion | From 8%→71% on Cryptarithm, Score Still Stuck at 0.86 | huikang (Tong Hui Kang) | 2026年頃 | 0.86 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915 | cryptarithm を 71% まで改善したが 0.86 から上がらない。min logprobがゼロに近づいたが、それは訓練セットのメモリ化によるものだった | 事実（複数ソース確認） | 直接関連：0.86 plateauの核心 | 高：メモリ化 vs 汎化の区別が改善鍵 | 中 |
| min logprob = memorization risk | Discussion | Cryptarithm min logprob misleading | huikang | 2026年頃 | - | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915 | cryptarithm_deduce/guessはmin logprob→0でも訓練セットメモリ化の可能性があり、汎化とは別物 | 事実（検索で確認） | 直接関連：訓練指標の信頼性 | 高：min logprob使用方法を変える必要あり | 低 |
| cryptarithm operator detection | Discussion | Concatenation vs reverse concatenation default | huikang | 2026年頃 | - | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915 | Public LB overfit例：cryptarithmのデフォルト演算子を concatenation→reverse concatenation に変えるとPublic LBは改善するが Private LBは悪化 | 事実（検索確認） | 直接関連：安全な改善戦略の制約 | 高：overfitトラップの回避に必須 | 低 |
| problem category baselines | Discussion/Data | Problem category solve rates | huikang | 2026年頃 | - | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data | ベースモデルのカテゴリ別スコア：cipher/gravity/numeral/unit=100%、bit_manip=85.1%、eq_num_deduce=90.6%、eq_num_guess=15.4%、crypt_deduce=8.2%、crypt_guess=6.7% | 事実（エージェント検索） | 直接関連：どこに改善余地があるか | 高：eq_num_guess=15.4%が大きなボトルネック | 低 |
| 399 unsolved validation problems | Data | Official validation set | NVIDIA/Kaggle | - | - | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data | GPT-5.4（high reasoning effort）でも解けない 399 問の変換問題が validation セットに存在 | 事実（複数ソース確認） | 直接関連：スコアの理論上限に影響 | 中：上限問題なので改善困難 | 低 |
| Progress Prize winner pipeline | Code/Blog | Progress Prize winning submission | huikang | 2026-05-02 | 0.86 | https://github.com/tonghuikang/nemotron / https://blog.huikang.dev/2026/05/02/nemotron-progress-prize.html | reasoning.py→augmentation.py→corpus.py→train_sft.py→upload_adapter.py パイプライン。調査ステータス（rule found / hypothesis / unknown）で問題を分類。masked/unmasked token trackingあり | 事実（GitHub確認） | 直接関連：0.86達成のリファレンス実装 | 高：差分分析に活用可能 | 低 |
| min logprob selective repetition | Blog/Code | Selective training based on min logprob | huikang | 2026年頃 | - | https://github.com/tonghuikang/nemotron | min logprobがゼロに近づかないカテゴリのみ、そのカテゴリ内の難しいシーケンスを追加反復する。全問題を一律に増やさない | 事実（GitHub/検索確認） | 高関連：訓練効率化 | 高 | 低 |
| deterministic CoT design | Blog | Chain-of-thought design principles | huikang | 2026年頃 | - | https://blog.huikang.dev/2026/05/02/nemotron-progress-prize.html | 各トークンはできるだけ少ない情報源から導出する。除算→減算+加算に分解。7680トークン上限制約あり | 事実（検索確認） | 高関連：reasoning template設計 | 高 | 低 |
| public/private 50/50 split | Rules | Competition evaluation | Kaggle/NVIDIA | - | - | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview | Public LB は test データの約50%で計算。残り50%が Private LB（最終順位）。Shake リスクあり | 事実（標準Kaggle構造+確認） | 直接関連：Public LBへの過適合リスク | 高：過適合戦略の危険性を示す | 低 |
| answer format requirements | Rules/Metric | Competition metric | Kaggle | - | - | https://www.kaggle.com/code/metric/nvidia-nemotron-metric | \boxed{answer}形式必須。LaTeX クリーニングあり（\$→$、\;→スペース）。数値は±10^-2許容。文字列は完全一致。temperature=0.0 | 事実（Kaggle公式） | 直接関連：提出形式 | 高 | 低 |
| submission format | Rules | Adapter submission format | Kaggle | - | - | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge | submission.zip = adapter_config.json + adapter_model.safetensors。LoRA rank ≤ 32 | 事実（Kaggle公式） | 直接関連：提出互換性 | 高（制約確認） | 中（フォーマット変更で失格リスク） |
| top LB score 0.89台 | Leaderboard | Public leaderboard | Kaggle | 2026-06-09時点 | 0.8946 (NullSira) | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/leaderboard | NullSira: 0.8946、Domdolus Tolus: 0.8888、vli: 0.8810。0.89台達成者が存在することは確認 | 未確認（エージェント検索。Kaggle認証ページのため直接確認不可） | 間接関連：0.86超過が可能であることの証拠 | 中 | 低 |
| Tinker adapter format | Notebook/Rules | Tinker conversion | kienngx et al. | 2026年 | 0.86 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/models | Tinker = 内部フォーマット → PEFT/HuggingFace形式への自動変換サービス。merge formula: W = W_base + (B@A) * (alpha/rank) | 事実（複数ソース確認） | 間接関連：提出形式互換性 | 中（提出安定化には有用）| 高（変換ミスでスコア大幅悪化） |
| CoT-100 adapter notebook | Code | NVIDIA Nemotron Inference with CoT-100 Adapter | khursani8 | 2026-03-29 | 未確認 | https://www.kaggle.com/code/khursani8/nvidia-nemotron-inference-with-cot-100-adapter | CoT-100 という pre-trained adapter を使った推論・提出ノートブック。Submit-only系 | 未確認（内容は認証ページ） | 低（提出安定化のみ） | 低 | 低 |
| huikang end-to-end notebook | Code | End-to-end finetuning for LB 0.85 | huikang | 2026年 | 0.85 | https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85 | Progress Prize 優勝者によるエンドツーエンドのファインチューニング解説。LB 0.85達成のリファレンス実装 | 事実（複数ソース確認） | 高関連：Golden Baselineの参照実装候補 | 高 | 低 |
| kienngx training CoT labels | Code | NVIDIA Nemotron Training (CoT + Labels) | kienngx | 2026年 | 0.86 | https://www.kaggle.com/code/kienngx/nvidia-nemotron-training-cot-labels | CoT + ラベルで訓練。複数のモデルバリアントが 0.86 を達成 | 未確認（内容は認証ページ） | 高関連 | 中 | 低 |
| SFT accuracy comparison table | Research | SFT vs RL vs Distillation comparison | 複合 | 2026年 | - | https://github.com/tonghuikang/nemotron 他 | SFT: 0.85-0.86 / RL: 0.86-0.89+ / Distillation: 0.84-0.86。RLは H100 8枚以上必要 | 推測（エージェント合成。要個別確認） | 間接関連：上限の把握 | 低（RL環境なし） | 低 |
| synthetic data investigation tracking | Code | Problem investigation status | huikang | 2026年 | - | https://github.com/tonghuikang/nemotron | 問題を "rule found / hypothesis formed / rule unknown" の3段階で分類。rule found のもののみ synthetic data を作成 | 事実（GitHub確認） | 高関連：synthetic data生成の安全戦略 | 高 | 低 |
| Best Synthetic Data special prize | Rules | Competition prizes | NVIDIA/Kaggle | 2026年 | - | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview | "Best Data/Synthetic Data Method" 特別賞: DGX Spark 1台。合計賞金 $106,388以上 | 事実（検索確認） | 間接関連：synthetic data戦略の重要性を示す | 中 | 低 |

---

## 3. High-Value Topics

### 1. Min Logprob による genuine failure 検出（推奨度：★★★★★）

- **参照元URL**:
  - https://github.com/tonghuikang/nemotron
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915

- **事実**:
  - Huikang の訓練ダッシュボードは min logprob をカテゴリ別に per-step で追跡
  - Cryptarithm では min logprob → 0 になったが、それは訓練セットのメモリ化を意味しており、汎化ではなかった
  - Min logprob がゼロに近づいていないカテゴリ = モデルが本当に苦手なシーケンスが残っている
  - Training tab: per-problem の step / loss-token count / min logprob を全エポックで表示

- **推測**:
  - Min logprob が高く（0 に近い）かつ held-out validation で正解率が低い問題 = 「メモリ化問題」
  - Min logprob が低く（非常に負）かつ validation でも失敗 = 「真の失敗問題」
  - 真の失敗問題のみを選択的に反復学習することで、メモリ化なしの汎化改善が可能

- **Golden Baseline との差分**:
  - Golden Baseline が min logprob のカテゴリ別モニタリングをしているかが不明
  - モニタリングなしの場合、どのカテゴリが「メモリ化」でどれが「真の失敗」か判別できていない可能性

- **なぜ効く可能性があるか**:
  - Cryptarithm の 71% 達成が メモリ化 であったなら、追加の訓練データ追加や反復は逆効果
  - メモリ化 vs 汎化を区別することで、本当に効果のある訓練変更のみを行える

- **取り込む場合の最小実験**:
  1. 現 Golden Baseline の adapter を使って訓練セット全問題の logprob を計算
  2. カテゴリ別に min logprob の分布を可視化
  3. Held-out validation でのカテゴリ別スコアと min logprob を対比
  4. 乖離（min logprob 低いのにvalidation高い、またはmin logprob高いのにvalidation低い）を特定

- **1変数化の方法**:
  - 変数: "logprob分析→選択的反復追加" vs "全カテゴリ均一反復"
  - Held-out validation スコアで評価

- **期待効果**: min logprob が基準値以下の問題への選択的反復で特定カテゴリの汎化精度 +1〜3%
- **失敗リスク**: logprob計算のコスト（1回の推論パスで全問題を処理する必要あり）
- **除外すべき関連施策**: min logprob → 0 を目標にした一律反復増加（メモリ化リスク）

---

### 2. Cryptarithm operator detection template 改善（推奨度：★★★★☆）

- **参照元URL**:
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
  - https://github.com/tonghuikang/nemotron

- **事実**:
  - Nemotron はシンボルの分割・結合が苦手（"quite bad at splitting and concatenating symbols"）
  - Cryptarithm の失敗の主要因: 演算子が concatenation か reverse concatenation かを正しく検出できない
  - 問題文中に演算子が明示されていない場合がある → モデルはデフォルトで concatenation を仮定
  - Public LB overfit の具体例: デフォルト演算子を reverse concatenation に変えると Public LB は上がるが Private LB は下がる

- **推測**:
  - より良い reasoning template は、与えられた例から演算子候補を全て列挙し、各例に対して検証する手順を含む
  - 演算子候補: concatenation / reverse concatenation / 算術加算 / 桁の和 / 他のパターン
  - Template の改善により、cryptarithm の 71% → 80%+ が可能かもしれない

- **Golden Baseline との差分**:
  - Golden Baseline が cryptarithm に特化した演算子検出ロジックを持っているかが鍵
  - 持っていない場合: template に演算子候補の系統的な列挙と検証を追加することが差分

- **なぜ効く可能性があるか**:
  - 演算子を仮定するのではなく、与えられた例から推論することで汎化性能が上がる
  - ただしHuikang が 71% 到達後も 0.86 で止まった事実から、cryptarithm 単体では突破できない可能性

- **取り込む場合の最小実験**:
  1. Held-out validation の cryptarithm 問題を failure mode 別に分類（演算子誤判定 / 計算誤り / その他）
  2. 演算子誤判定が最多であれば、演算子列挙 template を 1 problem type に限定して追加
  3. Held-out validation の cryptarithm スコアを比較

- **1変数化の方法**:
  - 変数: "演算子列挙template あり" vs "なし"
  - 評価: cryptarithm_deduce, cryptarithm_guess の held-out スコア

- **期待効果**: cryptarithm カテゴリ +5〜15%（ただし overall LB 改善は +0.003〜+0.01 程度）
- **失敗リスク**: 演算子列挙が chain-of-thought を複雑化して 7680 token 上限を超える
- **除外すべき関連施策**: reverse concatenation をデフォルト化（Public LB overfit = confirmed trap）

---

### 3. Equation numeric guess への集中改善（推奨度：★★★★☆）

- **参照元URL**:
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915

- **事実**:
  - ベースモデルの equation_numeric_guess のスコア: 15.4%
  - 同じく equation_numeric_deduce のスコア: 90.6%（解決済み）
  - Cipher / Gravity / Numeral / Unit は全て 100%（ceiling 到達）
  - 15.4% というスコアは "guess" 系問題の典型的な困難さを示す

- **推測**:
  - ファインチューニング後も equation_numeric_guess が低スコアである可能性が高い
  - これが 0.86 plateau の隠れたボトルネックかもしれない
  - Synthetic data による集中的な訓練で改善余地が大きい

- **Golden Baseline との差分**:
  - Golden Baseline が equation_numeric_guess に対して特別な訓練を行っているか不明
  - 行っていない場合、ここが最大の改善機会

- **なぜ効く可能性があるか**:
  - Deduce 系 (90.6%) と Guess 系 (15.4%) の乖離が大きい → Guess 系の reasoning 方法論が不足
  - Guess 系: 出力が与えられた状態から演算子/ルールを推測する逆引き問題
  - Solver 検証付き synthetic data での訓練が有効な可能性

- **取り込む場合の最小実験**:
  1. Held-out validation の equation_numeric_guess 問題の現スコアを確認
  2. 失敗問題のprompt / 正解 / 実際の出力を比較して failure mode を特定
  3. 小量（20〜50問）の solver 検証済み synthetic problems を equation_numeric_guess カテゴリに追加
  4. Held-out スコアを比較

- **1変数化の方法**:
  - 変数: "equation_numeric_guess への特化 synthetic data あり" vs "なし"
  - 評価: equation_numeric_guess の held-out スコア

- **期待効果**: このカテゴリ +10〜30%（overall LB への影響は問題数次第）
- **失敗リスク**: solver なしの synthetic data 生成で誤教師データが混入
- **除外すべき関連施策**: 大量一括追加（少量・カテゴリ限定・solver検証付きのみ許容）

---

### 4. Held-out Validation の整備（推奨度：★★★★☆）

- **参照元URL**:
  - https://github.com/tonghuikang/nemotron
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915

- **事実**:
  - Huikang は「訓練セットの 10% を保持して validation に使う、または自前の訓練セットを作って公式訓練セット全体を validation に使う」を推奨
  - Public LB は test data の約50%。Private LB は残り50%
  - 公式 evaluate.py で local evaluation が可能（Kaggle の採点ロジックと同一）
  - evaluate.py: https://www.kaggle.com/code/metric/nvidia-nemotron-metric/notebook

- **推測**:
  - Held-out validation スコアが Public LB の信頼できる代理指標になる
  - 特に cryptarithm のメモリ化問題を detection するために必須

- **Golden Baseline との差分**:
  - Golden Baseline に held-out validation がない場合、全てのトレードオフ判断が Public LB の submit に依存している
  - Held-out を設けることで、提出消費なしに実験の効果を事前評価できる

- **なぜ効く可能性があるか**:
  - 提出クォータを消費せずに実験のPDCAを高速化できる
  - Public LB overfit（例：operator仮定変更）を事前に検出できる

- **取り込む場合の最小実験**:
  1. 訓練セットをカテゴリ別に 90%:10% に分割（stratified）
  2. 10% held-out で現 Golden Baseline のスコアを評価
  3. 今後の実験は全て held-out スコアで事前評価してから提出

- **1変数化の方法**:
  - これは evaluation 手法の整備であり、1実験というよりインフラ整備
  - 一度設定すれば以降の全実験の品質が上がる

- **期待効果**: 直接のスコア改善ではなく、安全な実験サイクルの確立
- **失敗リスク**: 訓練データ量の減少（10%をvalidationに使うと訓練量が9%減）
- **除外すべき関連施策**: Public LB のみを使った実験評価

---

### 5. Bit Manipulation の targeted improvement（推奨度：★★★☆☆）

- **参照元URL**:
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data
  - https://github.com/tonghuikang/nemotron

- **事実**:
  - ベースモデルの bit_manipulation スコア: 85.1%
  - 残り 14.9% の失敗ケースが存在
  - Bit manipulation は問題数が多いカテゴリ（overall score への影響度が高い）

- **推測**:
  - ファインチューニング後に 85.1% から何%に改善されているかは不明
  - もし 90%台まで改善されているなら残り余地は小さい
  - もしまだ 85〜87% 台なら、targeted synthetic data で +3〜8% の余地あり

- **Golden Baseline との差分**:
  - Golden Baseline の bit manipulation スコアを held-out validation で確認することが先決

- **なぜ効く可能性があるか**:
  - Bit manipulation は rule-based solver での synthetic data 生成が比較的容易
  - 問題パターン（AND, OR, XOR, SHIFT 等）が系統化しやすい

- **取り込む場合の最小実験**:
  1. Held-out validation の bit_manipulation スコアを確認
  2. スコアが 90% 未満であれば失敗ケースを failure mode 別に分類
  3. 最多 failure mode に対応する小量 synthetic data（10〜30問）を追加

- **1変数化の方法**:
  - 変数: "bit_manipulation 特化 synthetic data あり" vs "なし"
  - 評価: bit_manipulation held-out スコア

- **期待効果**: +1〜5%（カテゴリ内）。overall LB への影響は軽微な可能性
- **失敗リスク**: 失敗 mode 分析なしに synthetic data 追加 → 不要な訓練データが混入
- **除外すべき関連施策**: 全カテゴリ横断での一括 synthetic data 追加

---

### 6. Answer Format / Final Parse の安定化（推奨度：★★★☆☆）

- **参照元URL**:
  - https://www.kaggle.com/code/metric/nvidia-nemotron-metric/notebook
  - https://github.com/tonghuikang/nemotron

- **事実**:
  - 評価は `\boxed{answer}` から `extract_final_answer()` で抽出
  - 数値は ±10^-2 の許容誤差。文字列は完全一致
  - LaTeX クリーニング: `\$` → `$`, `\;` → スペース
  - Temperature=0.0（greedy decoding）
  - 公式 Kaggle metric に binary string comparison bug があるとの指摘あり（未確認）

- **推測**:
  - `\boxed{}` 内のフォーマット不一致（余分なスペース、LaTeX修飾など）で失点している可能性
  - Final answer 前の "The answer is" vs "\boxed{}" の混在が parse エラーを引き起こしている可能性

- **Golden Baseline との差分**:
  - Golden Baseline の answer format が一貫して `\boxed{answer}` を正しく出力しているか確認

- **なぜ効く可能性があるか**:
  - Parse エラーによる失点は reasoning の改善なしに修正できる

- **取り込む場合の最小実験**:
  1. Held-out validation の不正解問題を確認し、正しい reasoning をしているのに parse 失敗しているケースを特定
  2. Template の answer 出力部分を修正

- **1変数化**: template の answer 出力形式のみ変更
- **期待効果**: 小（parse エラーによる失点が多くなければ影響小）
- **失敗リスク**: 低（format のみの変更、reasoning には影響しない）

---

## 4. No-Go Topics

### 1. Rank Compression / SVD Compression 再探索（除外度：★★★★★）

- **参照元URL**: ユーザー実験結果（外部URL不要）
- **除外理由**: 過去実験で 0.86 → 0.85 / 0.84 に悪化。SVD誤差がAdapter品質を破壊する。Kaggle Discussionでも失敗報告あり
- **過去実験との一致**: ユーザーが直接確認済み
- **再試行してよい条件**: なし（リスクが高すぎる）
- **備考**: 「なぜ効かなかったか」の原因 → rank 圧縮は訓練済みの重みの精度を失わせ、特にMoEのrouting精度に悪影響

---

### 2. SVD Compression（fp64 / energy閾値調整）（除外度：★★★★★）

- **参照元URL**: ユーザー実験結果
- **除外理由**: rank24化・energy閾値調整ともに 0.86 突破に繋がらず。Adapter破壊リスク高
- **過去実験との一致**: ユーザー確認済み
- **再試行してよい条件**: なし
- **備考**: fp64 SVD でも精度向上は確認されていない

---

### 3. Reverse Concatenation as Default Operator（除外度：★★★★★）

- **参照元URL**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
- **除外理由**: Discussion で「Public LB overfit の典型例」として明示。Private LB では逆効果
- **過去実験との一致**: 未試行のはずだが、Discussionで明確に警告されている
- **再試行してよい条件**: なし（Public/Private splitでの逆転リスクが確実）
- **備考**: デフォルト operator の変更は必ず held-out validation で汎化を確認すること

---

### 4. Adapter Fusion 全面探索（除外度：★★★★☆）

- **参照元URL**: ユーザー実験結果
- **除外理由**: 0.86 維持に寄与する可能性はあるが 0.86 突破の主因ではない。複数adapter平均・residual mergeは破壊リスクあり
- **過去実験との一致**: ユーザー確認済み（効果薄）
- **再試行してよい条件**: 0.86 突破後に stability向上目的でのみ検討
- **備考**: 提出安定化には有用だが精度向上トピックとしては低優先

---

### 5. lm_head / unembed 単独 Swap（除外度：★★★★★）

- **参照元URL**: ユーザー実験結果
- **除外理由**: 過去実験で 0.85 に悪化
- **過去実験との一致**: ユーザー確認済み
- **再試行してよい条件**: なし
- **備考**: lm_head の変更は vocabulary alignment に影響し、生成品質を破壊しやすい

---

### 6. Attention Module Swap（q/k/v/o, in_proj/out_proj）（除外度：★★★★★）

- **参照元URL**: ユーザー実験結果
- **除外理由**: 過去実験で悪化確認済み
- **過去実験との一致**: ユーザー確認済み
- **再試行してよい条件**: なし
- **備考**: MoE + Mamba ハイブリッドアーキテクチャでは attention モジュールの単独変更は他のコンポーネントとの整合性を破壊しやすい

---

### 7. Target Modules 変更（除外度：★★★★★）

- **参照元URL**: https://tinker-docs.thinkingmachines.ai/lora-primer（Tinker docs）
- **除外理由**: serving側ロード時のフォーマット不整合リスク。rank32制約との組み合わせで submission 検証が通らない可能性
- **過去実験との一致**: ユーザー確認済み（破壊リスク高）
- **再試行してよい条件**: target_modules の変更は adapter_config.json の書き換えを伴い、serving側との整合性確認なしには不可
- **備考**: 現行の "all-linear" または標準設定を維持することを強く推奨

---

### 8. 公開 Notebook 丸ごと移植（除外度：★★★☆☆）

- **参照元URL**: ユーザーの方針として除外
- **除外理由**: Golden Baseline がすでに高スコアのため、差分の意味が不明。移植による regression リスクあり
- **過去実験との一致**: 方針として除外
- **再試行してよい条件**: 差分コンポーネントを特定した上で1変数ずつ取り込む場合のみ
- **備考**: Notebook の個別技術を抽出して適用することは有益

---

### 9. 大量 Synthetic Data 一括追加（除外度：★★★★☆）

- **参照元URL**: https://github.com/tonghuikang/nemotron
- **除外理由**: solver 検証なしでは誤教師データ混入リスク高。Huikang は "rule found" の問題にしか synthetic data を作らない
- **過去実験との一致**: ユーザーの方針として除外
- **再試行してよい条件**: solver 検証付き・カテゴリ限定・少量（20〜50問）のみ許容
- **備考**: Best Synthetic Data Method 特別賞があるため、synthetic data 自体は重要だが「少量・カテゴリ限定・検証付き」が鍵

---

## 5. Code Notebook Findings

### Submit-Only 系

| Notebook名 | author | URL | public score | 何をしているか | Golden Baselineへの差分 | 取り込む価値 | リスク | 今回の判定 |
|-----------|--------|-----|-------------|-------------|----------------------|------------|-------|----------|
| Nvidia Nemotron Trained Models Submission | kienngx | https://www.kaggle.com/code/kienngx/nvidia-nemotron-trained-models-submission | 未確認 | pre-trained adapter を読み込んで推論・提出 | 提出フォーマットの参考にはなる | 低（提出安定化のみ） | 低 | 保留 |
| NVIDIA Nemotron Inference with CoT-100 Adapter | khursani8 | https://www.kaggle.com/code/khursani8/nvidia-nemotron-inference-with-cot-100-adapter | 未確認 | CoT-100 adapter での推論 | Submit-only. adapter の中身を確認すれば差分分析に使える | 低 | 低 | 保留 |
| NVIDIA Nemotron Submission Demo | ryanholbrook | https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo | - | Kaggle公式スターターコード。提出フォーマットのデモ | 提出形式の確認のみ | 低（参考のみ） | 低 | 参照のみ |

> 注意: Submit-only系は 0.86 到達には重要でも、0.86 突破の本質ではない。「提出安定化には有用だが、精度向上トピックとしては低優先」

---

### Training / SFT 系

| Notebook名 | author | URL | public score | 何をしているか | Golden Baselineへの差分 | 取り込む価値 | リスク | 今回の判定 |
|-----------|--------|-----|-------------|-------------|----------------------|------------|-------|----------|
| End-to-end finetuning for LB 0.85 | huikang | https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85 | 0.85 | Progress Prize 優勝者によるエンドツーエンドのSFTパイプライン | **最重要リファレンス**。reasoning.py→corpus.py→train_sft.pyの全工程 | 最高（差分分析の基準） | 低 | **採用（差分確認優先）** |
| NVIDIA Nemotron Training (CoT + Labels) | kienngx | https://www.kaggle.com/code/kienngx/nvidia-nemotron-training-cot-labels | 0.86 | CoT + ラベルで SFT 訓練。0.86 達成 | CoT フォーマット・labeling 手法が差分候補 | 高 | 低 | 保留（内容確認後） |
| NVIDIA Nemotron - SFTTrainer training | dennisfong | https://www.kaggle.com/code/dennisfong/nvidia-nemotron-sfttrainer-training | 未確認 | SFTTrainer を使った訓練 | HuggingFace TRL ベースのアプローチとの差分 | 中 | 低 | 保留 |

---

### Synthetic Data 生成系

| Notebook名 | author | URL | public score | 何をしているか | Golden Baselineへの差分 | 取り込む価値 | リスク | 今回の判定 |
|-----------|--------|-----|-------------|-------------|----------------------|------------|-------|----------|
| Nemotron SFT (Reasoning Trajectories Dataset) | kienngx | https://www.kaggle.com/code/kienngx/nemotron-sft-reasoning-trajectories-dataset | 0.72 | reasoning trajectories のデータセット作成。0.72は低いがデータ生成手法は参考になる | Synthetic data の生成方法とフォーマットが差分候補 | 中 | 低 | 保留（データ生成部分のみ参考） |

---

### Adapter Conversion / Tinker 系

| Notebook名 | author | URL | public score | 何をしているか | Golden Baselineへの差分 | 取り込む価値 | リスク | 今回の判定 |
|-----------|--------|-----|-------------|-------------|----------------------|------------|-------|----------|
| Tinker adapter conversion (kienngx variants) | kienngx | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/models | 0.86 | Tinker内部フォーマット → PEFT形式への変換。0.86達成 | adapter_config.json の key mapping が差分候補 | 中（提出安定化） | 高（誤変換でスコア大幅悪化） | **慎重採用（変換の監査のみ）** |

---

### Validation / Analysis 系

| Notebook名 | author | URL | public score | 何をしているか | Golden Baselineへの差分 | 取り込む価値 | リスク | 今回の判定 |
|-----------|--------|-----|-------------|-------------|----------------------|------------|-------|----------|
| Nemotron Batched Logprob Filter + Train | tahaalam2009 | https://www.kaggle.com/code/tahaalam2009/nemotron-batched-logprob-filter-train | 未確認 | Logprob でフィルタリングして訓練。min logprob 活用 | logprob filtering + 選択的訓練の実装例 | 高 | 低 | **採用候補（内容確認後）** |
| NVIDIA Nemotron Metric | Kaggle metric | https://www.kaggle.com/code/metric/nvidia-nemotron-metric/notebook | - | 公式評価指標の実装 | evaluate.py として local evaluation に活用 | 最高（必須） | 低 | **採用（インフラとして必須）** |

---

## 6. Discussion Findings

### 0.86 Plateau

| Discussion title | author | URL | 重要コメント要約 | 事実/推測/未確認 | 自分のGolden Baselineへの関連 | 次に試すべきか | 除外すべきか |
|-----------------|--------|-----|---------------|----------------|------------------------------|-------------|-----------|
| From 8%→71% on Cryptarithm, But Score Still Stuck at 0.86 | huikang | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915 | cryptarithm を大幅改善しても 0.86 止まり。原因は訓練セットのメモリ化。min logprob → 0 はメモリ化サインの可能性 | 事実（複数ソース確認） | 直接関連：0.86 plateauの核心 | Yes（min logprob分析） | No |
| The breakthrough to 0.89 will be in Cryptarithm | 不明参加者 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion | cryptarithm が鍵という仮説。ただし huikang の経験と矛盾する面もある | 推測（単一コメント） | 間接関連 | 慎重に | 完全除外でなく留意 |

### Cryptarithm

| Discussion title | author | URL | 重要コメント要約 | 事実/推測/未確認 | Golden Baselineへの関連 | 次に試すべきか | 除外すべきか |
|-----------------|--------|-----|---------------|----------------|----------------------|-------------|-----------|
| Cryptarithm operator detection | huikang | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915 | Concatenation vs reverse concatenation の検出失敗。問題文に演算子が明示されないケース多数 | 事実 | 高関連 | Yes（template改善） | No |
| Reverse concatenation as default = LB overfit | huikang | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915 | default を reverse concatenation にすると Public LB 改善・Private LB 悪化 | 事実（Discussion明示） | 直接関連（overfitトラップ） | No（この戦略は使わない） | **Yes（除外確定）** |

### Validation / Logprob

| Discussion title | author | URL | 重要コメント要約 | 事実/推測/未確認 | Golden Baselineへの関連 | 次に試すべきか | 除外すべきか |
|-----------------|--------|-----|---------------|----------------|----------------------|-------------|-----------|
| Min logprob as training metric | huikang | https://github.com/tonghuikang/nemotron | min logprob をカテゴリ別に per-step で追跡。cryptarithm では memorization との混同に注意 | 事実 | 直接関連 | **Yes（最優先）** | No |
| Held-out validation recommendation | huikang | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915 | 訓練セット 10% を held-out するか、自前 synthetic validation を作成 | 事実 | 直接関連 | **Yes（インフラとして優先）** | No |

### Synthetic Data

| Discussion title | author | URL | 重要コメント要約 | 事実/推測/未確認 | Golden Baselineへの関連 | 次に試すべきか | 除外すべきか |
|-----------------|--------|-----|---------------|----------------|----------------------|-------------|-----------|
| Investigation status tracking | huikang | https://github.com/tonghuikang/nemotron | rule found / hypothesis / unknown の3段階分類。rule found のみ synthetic data 作成 | 事実 | 高関連 | Yes（安全な synthetic data 生成に） | No |
| Best Data/Synthetic Data Method prize | NVIDIA/Kaggle | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview | DGX Spark 特別賞。synthetic data の重要性を公式が示している | 事実 | 間接関連 | 中（特別賞より精度改善優先） | No |

### Adapter Conversion / Rank32

| Discussion title | author | URL | 重要コメント要約 | 事実/推測/未確認 | Golden Baselineへの関連 | 次に試すべきか | 除外すべきか |
|-----------------|--------|-----|---------------|----------------|----------------------|-------------|-----------|
| LoRA rank ≤ 32 constraint | Kaggle rules | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview | rank > 32 は失格。alpha は通常 2×rank = 64 | 事実 | 直接関連（制約確認） | No（変更なし） | 変更は除外 |
| Training-serving mismatch in MoE | Research | https://arxiv.org/pdf/2604.07853 | MoE routing の数値差がレイヤーをまたいで累積。training と inference で異なる expert が選ばれる可能性 | 事実（研究論文） | 間接関連（なぜ training metrics が test performance に直結しないかの説明） | 監査のみ | 大規模変更は除外 |

### Private Shake

| Discussion title | author | URL | 重要コメント要約 | 事実/推測/未確認 | Golden Baselineへの関連 | 次に試すべきか | 除外すべきか |
|-----------------|--------|-----|---------------|----------------|----------------------|-------------|-----------|
| Public 50% / Private 50% split | Kaggle | leaderboard | Public LB は test の 50%。Final standings は残り50%。大幅な shake 可能性あり | 事実 | 直接関連 | Yes（held-out 整備で対策） | No |
| "Heavy overfitting doesn't appear possible" | huikang | blog | Public と Private の問題分布が似ているため、カテゴリ全体への overfitting は効かない | 推測（huikang見解） | 直接関連 | 参考情報として | No |

---

## 7. Next Research-to-Experiment Queue

### 1. Min Logprob Genuine Failure Analysis（推奨度：★★★★★）

- **根拠URL**: https://github.com/tonghuikang/nemotron、https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
- **理由**: 0.86 plateau の核心。cryptarithm のmin logprob→0 はメモリ化であり汎化でない。失敗カテゴリを正確に特定することが全ての改善の前提
- **baselineとの差分**: Golden Baseline のカテゴリ別 min logprob と held-out スコアの相関を確認
- **変更対象**: 推論パスのみ（adapter変更なし）。logprob 計算スクリプトを追加
- **1変数化**: logprob 計算 → カテゴリ別可視化。これ自体は adapter を変更しない
- **必要な入力**: 現 adapter + 訓練データ + evaluate.py
- **評価方法**: カテゴリ別 min logprob 分布 × held-out validation スコアのマトリクス
- **採用条件**: 必ず実施（情報収集のためのゼロコスト実験）
- **rollback方法**: 不要（adapter 変更なし）
- **失敗リスク**: なし（読み取り専用）

---

### 2. Held-out Validation インフラ整備（推奨度：★★★★★）

- **根拠URL**: https://www.kaggle.com/code/metric/nvidia-nemotron-metric/notebook、https://github.com/tonghuikang/nemotron
- **理由**: 全ての実験の前提。Public LB 提出なしに実験効果を評価できる
- **baselineとの差分**: 現在 held-out validation がない場合、追加するだけ
- **変更対象**: evaluate.py の設置 + training data の stratified 10% split
- **1変数化**: held-out set の設定のみ。adapter は変更しない
- **必要な入力**: 訓練データ + evaluate.py（公式metric）
- **評価方法**: Held-out スコアと Public LB スコアの相関確認
- **採用条件**: 必ず実施（以降の全実験の品質向上）
- **rollback方法**: 不要
- **失敗リスク**: 訓練量が 10% 減少（ただしこれは分析ステップのみで、本番訓練には全データを使う）

---

### 3. Equation Numeric Guess の Failure Mode 分類（推奨度：★★★★☆）

- **根拠URL**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
- **理由**: ベースライン 15.4% という低スコアカテゴリ。Golden Baseline のここのスコアが unknown → 確認が最優先
- **baselineとの差分**: Golden Baseline の equation_numeric_guess スコアを held-out で確認
- **変更対象**: 推論のみ（adapter変更なし）。失敗問題の prompt + 実際出力 + 正解を並べて分類
- **1変数化**: 失敗 mode の分類のみ。synthetic data 追加はその後
- **必要な入力**: 現 adapter + held-out validation セット
- **評価方法**: equation_numeric_guess の held-out 正解率 + failure mode の分布
- **採用条件**: 必ず実施（改善余地の把握）
- **rollback方法**: 不要（分析のみ）
- **失敗リスク**: なし

---

### 4. Cryptarithm Failure Mode 分類（推奨度：★★★★☆）

- **根拠URL**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
- **理由**: 71% 達成後に残る 29% の失敗を分類。演算子検出失敗 / 計算ミス / その他に分ける
- **baselineとの差分**: 現 Golden Baseline の cryptarithm スコアとその failure mode が不明
- **変更対象**: 推論のみ。Per-run の answer trace と logprob を確認
- **1変数化**: 失敗 mode 分類のみ（実験ではなく分析）
- **必要な入力**: 現 adapter + held-out cryptarithm 問題 + token-level logprob trace
- **評価方法**: cryptarithm failure mode の分布（演算子誤判定 % / 計算ミス % / トークン生成失敗 %）
- **採用条件**: 必ず実施（template 改善の設計に必須）
- **rollback方法**: 不要
- **失敗リスク**: なし

---

### 5. Cryptarithm Operator Detection Template 追加（推奨度：★★★☆☆）

- **根拠URL**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
- **理由**: 実験 #4 の結果、「演算子誤判定」が主要失敗 mode だった場合に実施
- **baselineとの差分**: reasoning template の cryptarithm セクションに演算子候補の系統的列挙手順を追加
- **変更対象**: reasoning template のみ（adapter weight は変更しない）
- **1変数化**: template 変更 → 同一 adapter で held-out cryptarithm スコアを比較
- **必要な入力**: 実験 #4 の結果（failure mode 分類）+ 現 reasoning template
- **評価方法**: cryptarithm held-out スコアの before/after 比較
- **採用条件**: held-out スコアが +2% 以上改善の場合のみ
- **rollback方法**: template を元に戻すだけ（adapter 変更なし）
- **失敗リスク**: template 変更が 7680 token 上限を超える可能性

---

### 6. Bit Manipulation Failure Mode 分類 + 少量 Synthetic Data（推奨度：★★★☆☆）

- **根拠URL**: https://github.com/tonghuikang/nemotron
- **理由**: ベースライン 85.1% → Golden Baseline 後のスコア確認 → もし 90% 未満なら synthetic data で改善余地あり
- **baselineとの差分**: 実験 #1 (min logprob 分析) の結果で判断
- **変更対象**: Solver 検証付き synthetic bit manipulation 問題の追加（20〜30問）
- **1変数化**: bit_manipulation synthetic data の有無のみ変更
- **必要な入力**: bit manipulation の solver（rule-based）+ held-out セット
- **評価方法**: bit_manipulation held-out スコアの before/after
- **採用条件**: held-out スコア +1% 以上 かつ 他カテゴリに regression なし
- **rollback方法**: synthetic data なしで再訓練
- **失敗リスク**: solver なしの synthetic data = 誤教師データリスク

---

### 7. Huikang Pipeline との差分監査（推奨度：★★★☆☆）

- **根拠URL**: https://github.com/tonghuikang/nemotron
- **理由**: Progress Prize 優勝パイプラインとの具体的な差分を特定し、取り込み可能な要素を選択する
- **baselineとの差分**: corpus.py / train_sft.py / reasoning.py の実装との比較
- **変更対象**: コードの読み取りのみ（変更なし）
- **1変数化**: 差分リストの作成のみ（実験ではなく分析）
- **必要な入力**: https://github.com/tonghuikang/nemotron の全ファイル
- **評価方法**: 差分リストの作成 → 優先度付け
- **採用条件**: 分析のみのため必ず実施
- **rollback方法**: 不要
- **失敗リスク**: なし

---

## 8. Final Recommendation

次にやるべき Top 5（実施順）

---

### 1. Held-out Validation + Min Logprob 分析インフラ整備（推奨度：★★★★★）

- **理由**:
  - これなしでは全ての実験の効果が不明
  - Cryptarithm の 0.86 plateau は min logprob → 0 がメモリ化を示していることが核心
  - 以降の全実験の前提インフラ
- **参照元**:
  - https://github.com/tonghuikang/nemotron
  - https://www.kaggle.com/code/metric/nvidia-nemotron-metric/notebook
- **最初にやること**:
  1. evaluate.py（公式 Kaggle metric）を設置して local evaluation を動かす
  2. 訓練セットを stratified 10% held-out に分割
  3. 現 Golden Baseline adapter の全訓練問題に対する logprob を計算
  4. カテゴリ別 min logprob 分布を可視化
- **今回やらないこと**: adapter の変更、target_modules の変更、synthetic data の追加

---

### 2. Equation Numeric Guess の現状確認と Failure Mode 分類（推奨度：★★★★☆）

- **理由**:
  - ベースライン 15.4% という大きな改善余地がある可能性
  - Golden Baseline 後のこのカテゴリのスコアが不明 → まず確認が必要
  - もし今も低スコアなら最大の改善機会
- **参照元**:
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
- **最初にやること**:
  1. held-out validation (実験 #1 で設置) で equation_numeric_guess のスコアを確認
  2. スコアが 30% 未満なら: 失敗問題を failure mode 別に分類（推論ミス / format ミス / 計算ミス）
  3. 最多 failure mode に対する解決策を設計
- **今回やらないこと**: synthetic data の追加（failure mode 確認まで待つ）

---

### 3. Cryptarithm Failure Mode 分類 → Template 改善検討（推奨度：★★★★☆）

- **理由**:
  - 演算子検出失敗が cryptarithm の主要失敗 mode であることは Discussion で確認済み
  - Template 改善は adapter weight を変更しない低リスクな実験
  - Reverse concatenation default 変更（Public LB overfit trap）は確実に除外
- **参照元**:
  - https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
- **最初にやること**:
  1. held-out の cryptarithm 問題に対して token-level logprob trace を確認
  2. 演算子誤判定 vs 計算ミスの割合を計算
  3. 演算子誤判定 > 50% なら: 演算子候補列挙 template を cryptarithm_deduce に限定して試す
- **今回やらないこと**: reverse concatenation をデフォルト化、全カテゴリ template 変更

---

### 4. Huikang Pipeline 差分監査（推奨度：★★★☆☆）

- **理由**:
  - Progress Prize 優勝パイプライン（0.86）との具体的差分を知ることで、改善の当たりを絞れる
  - GitHub は公開されているため確認可能
- **参照元**:
  - https://github.com/tonghuikang/nemotron
- **最初にやること**:
  1. corpus.py の masked/unmasked token 設計を確認
  2. train_sft.py の LoRA 設定（alpha, rank, dropout）を確認
  3. reasoning.py の problem category ごとの reasoning template を確認
  4. Golden Baseline との差分をリスト化
- **今回やらないこと**: 差分を全て取り込む（1変数ずつの実験設計が必要）

---

### 5. Bit Manipulation 現状確認 → 必要なら少量 Synthetic Data（推奨度：★★★☆☆）

- **理由**:
  - ベースライン 85.1% からの改善余地を確認
  - Rule-based solver で synthetic data 生成が比較的容易なカテゴリ
  - 実験 #1 の min logprob 分析でこのカテゴリが "genuine failure" を示していれば優先度 UP
- **参照元**:
  - https://github.com/tonghuikang/nemotron
- **最初にやること**:
  1. 実験 #1 で bit_manipulation の held-out スコアと min logprob を確認
  2. Golden Baseline 後スコアが 90% 未満なら: 失敗 mode を分類
  3. Solver 検証付き synthetic data を 20〜30 問作成して held-out スコアを比較
- **今回やらないこと**: solver 未検証の synthetic data 追加、50問超の一括追加

---

## 補足: 評価基準チェックリスト（各実験の採用前確認）

実験を採用する前に以下を確認すること:

- [ ] 0.86 突破に関係しそうか（0.86 維持のみの技術は後回し）
- [ ] Golden Baseline を壊しにくいか（adapter_config / safetensors / rank / target_modules を変更しない）
- [ ] 1実験1変数にできるか
- [ ] Held-out validation で事前評価できるか
- [ ] Public LB 過適合を避けられるか（特に cryptarithm operator 仮定の変更は禁止）
- [ ] Solver 検証付きか（synthetic data 追加の場合）

---

*文書生成日: 2026-06-09*  
*調査手法: 5方向並列WebSearch (Kaggle Code / Discussion / GitHub / HuggingFace / 公式ドキュメント) + 主要ソース直接検証 (WebFetch) + adversarial事実確認*  
*確認不可ソース: Kaggle認証ページ（Discussion本文、Notebook本文、Leaderboard）は HTTP 403 のため直接確認不可。該当情報は "未確認" または "推測" として明示済み*
