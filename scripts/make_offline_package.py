"""
make_offline_package.py

オフライントレーニング用のzipパッケージを作成する。

作成されるzipの構成:
  nemotron_offline/
    scripts/
      train_sft.py
      prepare_corpus.py
      download_model.py
    reports/cryptarithm/
      corpus.jsonl
    requirements.txt
    run_training.sh
    run_training_kaggle.py    # Kaggle notebook用セル
    README_OFFLINE.txt

Usage:
    python scripts/make_offline_package.py
    # → nemotron_offline.zip が作成される
"""

from __future__ import annotations

import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = REPO_ROOT / "nemotron_offline.zip"

# zipに含めるファイル: (実際のパス, zip内パス)
FILES = [
    (REPO_ROOT / "scripts" / "train_sft.py",           "nemotron_offline/scripts/train_sft.py"),
    (REPO_ROOT / "scripts" / "prepare_corpus.py",      "nemotron_offline/scripts/prepare_corpus.py"),
    (REPO_ROOT / "scripts" / "download_model.py",      "nemotron_offline/scripts/download_model.py"),
    (REPO_ROOT / "reports" / "cryptarithm" / "corpus.jsonl",
                                                        "nemotron_offline/reports/cryptarithm/corpus.jsonl"),
]

REQUIREMENTS_TXT = """\
transformers>=4.40.0
trl>=1.0.0
peft>=0.10.0
accelerate>=0.28.0
bitsandbytes>=0.43.0
datasets>=2.18.0
torch>=2.2.0
"""

RUN_TRAINING_SH = """\
#!/bin/bash
# オフラインでのトレーニング実行スクリプト
# 事前に download_model.py でモデルをダウンロードしておくこと

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model_cache/nemotron-8b"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: モデルが見つかりません: $MODEL_DIR"
    echo "インターネットがある環境で先に実行してください:"
    echo "  python scripts/download_model.py --save_dir $MODEL_DIR"
    exit 1
fi

python "${SCRIPT_DIR}/scripts/train_sft.py" \\
    --corpus  "${SCRIPT_DIR}/reports/cryptarithm/corpus.jsonl" \\
    --output_dir "${SCRIPT_DIR}/adapter_output" \\
    --model_name "${MODEL_DIR}" \\
    --lora_r 16 \\
    --lora_alpha 32 \\
    --epochs 1 \\
    --batch_size 2 \\
    --grad_accum 4 \\
    --lr 2e-4 \\
    --max_seq_len 512
"""

RUN_TRAINING_KAGGLE_PY = '''\
# ===================================================================
# Kaggle オフライン実行用セル (インターネット無効時)
# 事前準備:
#   1. このzipをKaggleデータセットとしてアップロード
#      (例: dataset名 = "nemotron-offline-pkg")
#   2. モデルをKaggleデータセットとして追加
#      nvidia/nemotron → /kaggle/input/nvidia/nemotron/...
#      または事前ダウンロード済みzipをデータセットとして追加
# ===================================================================

import sys, os, zipfile, shutil

# --- コード+データの展開 ---
PKG_ZIP   = "/kaggle/input/nemotron-offline-pkg/nemotron_offline.zip"
WORK_DIR  = "/kaggle/working/Nemotron"

if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)

with zipfile.ZipFile(PKG_ZIP) as z:
    z.extractall("/kaggle/working/")

# zipの中身は nemotron_offline/ → WORK_DIR にリネーム
os.rename("/kaggle/working/nemotron_offline", WORK_DIR)

sys.path.insert(0, f"{WORK_DIR}/scripts")
os.chdir(WORK_DIR)
print("Setup done:", os.listdir(WORK_DIR))

# --- モデルパス (Kaggle model or ローカルキャッシュ) ---
# オプション1: KaggleにあるNVIDIA Nemotronモデルを使う場合
MODEL_PATH = "/kaggle/input/nvidia/nemotron/transformers/llama-3.1-nemotron-nano-8b-v1/1"
# オプション2: 自前でアップロードしたモデルキャッシュ
# MODEL_PATH = "/kaggle/input/nemotron-model-cache/nemotron-8b"

# --- トレーニング実行 ---
import subprocess
result = subprocess.run([
    "python", f"{WORK_DIR}/scripts/train_sft.py",
    "--corpus",     f"{WORK_DIR}/reports/cryptarithm/corpus.jsonl",
    "--output_dir", "/kaggle/working/adapter",
    "--model_name", MODEL_PATH,
    "--lora_r",     "16",
    "--lora_alpha", "32",
    "--epochs",     "1",
    "--batch_size", "2",
    "--grad_accum", "4",
    "--lr",         "2e-4",
    "--max_seq_len","512",
], check=True)
'''

README_OFFLINE_TXT = """\
=== Nemotron オフライントレーニングパッケージ ===

【手順】

1. インターネットがある環境でモデルをダウンロード:
   pip install transformers
   python scripts/download_model.py \\
       --model_name nvidia/Llama-3.1-Nemotron-Nano-8B-v1 \\
       --save_dir ./model_cache/nemotron-8b
   ※ model_cache/ フォルダはこのフォルダ内に作成されます

2. 依存パッケージをインストール:
   pip install -r requirements.txt

3. zipを展開してこのフォルダに入ってからスクリプトを実行:
   unzip nemotron_offline.zip
   cd nemotron_offline
   bash run_training.sh
   # アダプタは ./adapter_output/ に保存される

【重要】run_training.sh は nemotron_offline/ フォルダの中にあります。
  解凍した場所ではなく、必ず cd nemotron_offline してから実行してください。

【出力ファイル】
  nemotron_offline/adapter_output/adapter_model.safetensors
  nemotron_offline/adapter_output/adapter_config.json

【Kaggle オフラインの場合】
  run_training_kaggle.py の内容をノートブックセルに貼り付けてください。
  (事前にこのzipをKaggleデータセットとしてアップロード)

【推奨トレーニング設定 (RTX PRO6000 / A100)】
  --epochs 1 --batch_size 2 --grad_accum 4 --max_seq_len 512
  → 約100ステップ、2〜3時間で完了
"""


def main() -> None:
    print(f"Creating offline package: {OUTPUT_ZIP}", flush=True)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # ファイルを追加
        for src, dst in FILES:
            if src.exists():
                zf.write(src, dst)
                size_kb = src.stat().st_size / 1024
                print(f"  + {dst}  ({size_kb:.0f} KB)", flush=True)
            else:
                print(f"  SKIP (not found): {src}", flush=True)

        # 生成コンテンツを追加
        zf.writestr("nemotron_offline/requirements.txt",      REQUIREMENTS_TXT)
        zf.writestr("nemotron_offline/run_training.sh",       RUN_TRAINING_SH)
        zf.writestr("nemotron_offline/run_training_kaggle.py", RUN_TRAINING_KAGGLE_PY)
        zf.writestr("nemotron_offline/README_OFFLINE.txt",    README_OFFLINE_TXT)
        print("  + requirements.txt, run_training.sh, run_training_kaggle.py, README_OFFLINE.txt")

    size_mb = OUTPUT_ZIP.stat().st_size / 1e6
    print(f"\nDone: {OUTPUT_ZIP}  ({size_mb:.1f} MB)", flush=True)
    print("\n次のステップ:")
    print("  1. (インターネットあり) python scripts/download_model.py")
    print("  2. (インターネットなし) bash run_training.sh  または  run_training_kaggle.py をKaggleに貼り付け")


if __name__ == "__main__":
    main()
