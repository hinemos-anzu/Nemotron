# Day1 Kaggle Runbook — NVIDIA Nemotron Model Reasoning Challenge

**Date:** 2026-04-18
**Environment:** Kaggle Notebook (正本環境)
**Purpose:** ベースA提出資産をKaggle上で再現するための手順書

---

## 前提条件チェックリスト

実行前に以下をすべて確認すること。

- [ ] Kaggle Notebook が GPU (T4 x2 または A100) で動作している
- [ ] base model dataset が `/kaggle/input/` 以下にマウントされている
- [ ] adapter checkpoint dataset が `/kaggle/input/` 以下にマウントされている
- [ ] Internet アクセスが OFF になっている（コンペ規定に従う）
- [ ] Notebook runtime が Python 3.10+ / PyTorch 2.x であること

---

## Paths（固定値）

```
BASE_MODEL_PATH     = "/kaggle/input/nemotron-base/nemotron-base"
ADAPTER_PATH        = "/kaggle/input/nemotron-adapter/adapter"
MERGED_OUTPUT_PATH  = "/kaggle/working/merged_model"
SVD_OUTPUT_PATH     = "/kaggle/working/svd_model"
SUBMISSION_ZIP_PATH = "/kaggle/working/submission.zip"
```

> **注意:** これらのパスはベースAコードと完全一致させること。
> パスを変更すると submission.zip が空になるかロードエラーになる。

---

## Dataset / Model / Tokenizer

| 資産 | Kaggle Dataset 名 | バージョン |
|------|-------------------|-----------|
| Base Model | nemotron-base | v1 (固定) |
| Adapter Checkpoint | nemotron-adapter | v1 (固定) |
| Tokenizer | base model に同梱 | — |

> **重要:** Dataset バージョンを固定しないと再現性が失われる。
> Day2以降の実験でも同一バージョンの Dataset を使用すること。

---

## 実行順序（厳守）

以下の順序を**変更してはならない**。
各ステップは前ステップの出力を入力とするため、順序を崩すと submission.zip が破損する。

### Step 1: adapter変換

```python
# LoRA adapter を base model にマージ
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()
model.save_pretrained(MERGED_OUTPUT_PATH)
tokenizer.save_pretrained(MERGED_OUTPUT_PATH)
```

**確認:** `MERGED_OUTPUT_PATH` に `model.safetensors` または `pytorch_model.bin` が存在すること。

### Step 2: Offline Asymmetric SVD Surgery

```python
# SVD による非対称低ランク近似
# 対象レイヤ: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
# rank: 各レイヤのオリジナルrank以下（ハイパーパラメータ）

run_svd_surgery(
    input_path=MERGED_OUTPUT_PATH,
    output_path=SVD_OUTPUT_PATH,
    rank_ratio=0.5,        # ベースA固定値
    asymmetric=True        # Asymmetric SVD を有効化
)
```

**確認:** `SVD_OUTPUT_PATH` に変換済み weights が存在すること。

### Step 3: key rename

```python
# 推論サーバ互換のキー名に変換
rename_checkpoint_keys(
    input_path=SVD_OUTPUT_PATH,
    key_map=NEMOTRON_KEY_MAP   # ベースAに定義されたマッピング辞書
)
```

**確認:** 変換後のキー名が推論サーバの期待形式に一致していること。

### Step 4: expert unfuse

```python
# MoE expert weights をアンフューズ
unfuse_experts(
    input_path=SVD_OUTPUT_PATH,
    num_experts=8              # ベースA固定値
)
```

**確認:** `experts.0.weight` 〜 `experts.7.weight` が存在すること。

### Step 5: gate_proj + x_proj → in_proj 統合

```python
# 2つの projection を concat して in_proj に統合
merge_projections(
    input_path=SVD_OUTPUT_PATH,
    gate_key="gate_proj.weight",
    x_key="x_proj.weight",
    out_key="in_proj.weight"
)
```

**確認:** `in_proj.weight` が存在し、`gate_proj.weight` と `x_proj.weight` が除去されていること。

### Step 6: submission.zip 生成

```python
import zipfile, os

with zipfile.ZipFile(SUBMISSION_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(SVD_OUTPUT_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, SVD_OUTPUT_PATH)
            zf.write(file_path, arcname)

print(f"submission.zip created: {os.path.getsize(SUBMISSION_ZIP_PATH)} bytes")
```

**確認:** `submission.zip` のサイズが妥当な範囲（数GB以上）であること。

---

## 成立確認チェックリスト

実行後に以下をすべて確認すること。

- [ ] `MERGED_OUTPUT_PATH` にモデルファイルが存在する
- [ ] `SVD_OUTPUT_PATH` にSVD変換済みファイルが存在する
- [ ] `in_proj.weight` が存在し、`gate_proj.weight`/`x_proj.weight` が消えている
- [ ] MoE expert weights がアンフューズされている
- [ ] `submission.zip` が生成されており、サイズが 0 でない
- [ ] `submission.zip` を unzip して config.json が含まれていること

---

## BLOCKED 要因リスト

以下のいずれかに該当する場合、Day1は BLOCKED とする。

| BLOCKED 要因 | 症状 | 対処 |
|-------------|------|------|
| Dataset マウント失敗 | `/kaggle/input/` に base model が存在しない | Dataset を再 Add する |
| Path 不一致 | `FileNotFoundError` | コード内の PATH 定数を確認 |
| GPU OOM | CUDA Out of Memory | A100 に切り替え、または bfloat16 確認 |
| adapter 互換性エラー | PEFT バージョン不一致 | `pip install peft==X.X.X` で固定 |
| SVD surgery クラッシュ | weight shape 不一致 | rank_ratio を下げる / base model バージョン確認 |
| submission.zip が空 | ファイルサイズ 0 | `SVD_OUTPUT_PATH` が空でないか確認 |

---

## 注意事項

1. **バージョン固定:** PyTorch, PEFT, transformers のバージョンは Notebook の requirements に固定すること
2. **再現性:** seed は 42 に固定する（sampling を使う場合）
3. **ベースAコードを壊さない:** Day2以降の改善は別セルまたは別 Notebook に追記すること
4. **提出前確認:** submission.zip の MD5 を記録し、Day2以降の比較に使用する
