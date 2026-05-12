"""
download_model.py

インターネットオフにする前に実行してモデルをローカルに保存するスクリプト。

Usage:
    python scripts/download_model.py \
        --model_name nvidia/Llama-3.1-Nemotron-Nano-8B-v1 \
        --save_dir ./model_cache/nemotron-8b

After running, use --model_name ./model_cache/nemotron-8b in train_sft.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def download(model_name: str, save_dir: str) -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(str(save_path))
    print(f"  Tokenizer saved → {save_path}", flush=True)

    print(f"Downloading model weights: {model_name}", flush=True)
    print("  (this may take 10-30 minutes depending on bandwidth)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.save_pretrained(str(save_path))
    print(f"  Model saved → {save_path}", flush=True)

    size_gb = sum(f.stat().st_size for f in save_path.rglob("*") if f.is_file()) / 1e9
    print(f"\nTotal size: {size_gb:.1f} GB", flush=True)
    print(f"\nNext step (offline): use --model_name {save_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-download model for offline training")
    p.add_argument("--model_name", default="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                   help="HuggingFace model ID")
    p.add_argument("--save_dir", default="./model_cache/nemotron-8b",
                   help="Local directory to save model weights")
    args = p.parse_args()
    download(args.model_name, args.save_dir)


if __name__ == "__main__":
    main()
