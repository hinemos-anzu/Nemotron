"""
train_sft.py

SFT (Supervised Fine-Tuning) with LoRA for the NVIDIA Nemotron Reasoning Challenge.

Reads a corpus JSONL file (produced by prepare_corpus.py), formats each record as an
OpenAI messages-format conversation, and fine-tunes the Nemotron-3-Nano-30B-A3B base
model with QLoRA (4-bit + LoRA) using TRL SFTTrainer.

Key competition requirements enforced here:
  - System prompt must be "detailed thinking off"
  - Answers must be enclosed in \\boxed{...}
  - LoRA rank must be ≤ 32
  - Submission needs adapter_model.safetensors + adapter_config.json

Output:
  {output_dir}/adapter_model.safetensors
  {output_dir}/adapter_config.json

Usage (Kaggle notebook):
    !pip install -q transformers trl peft accelerate bitsandbytes datasets

    !python scripts/train_sft.py \\
        --corpus reports/cryptarithm/corpus.jsonl \\
        --output_dir /kaggle/working/adapter \\
        --model_name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
        --lora_r 16 \\
        --lora_alpha 32 \\
        --epochs 3 \\
        --batch_size 1 \\
        --grad_accum 8 \\
        --lr 2e-4 \\
        --max_seq_len 2048
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(path: Path) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("verified", True):
                    records.append(r)
            except json.JSONDecodeError:
                pass
    return records


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

# Competition-required system prompt (disables chain-of-thought token generation
# in the base model's default mode so our explicit CoT controls the trace)
_SYSTEM_PROMPT = "detailed thinking off"


def record_to_messages(record: Dict) -> Dict:
    """
    Convert a CoT record to OpenAI messages format.

    Assistant turn format:
      <step-by-step reasoning>
      \\boxed{answer}

    This matches the competition's evaluation format (exact match after \\boxed{} extraction,
    or ±0.01 numeric tolerance for gravity/unit_conversion).
    """
    prompt  = record.get("prompt",  record.get("question", "")).strip()
    cot     = record.get("cot",     "").strip()
    answer  = record.get("answer",  "").strip()

    if cot:
        assistant_content = f"{cot}\n\\boxed{{{answer}}}"
    else:
        assistant_content = f"\\boxed{{{answer}}}"

    return {
        "messages": [
            {"role": "system",    "content": _SYSTEM_PROMPT},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    import torch
    from collections import Counter
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer

    corpus_path = Path(args.corpus)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enforce competition LoRA rank limit
    if args.lora_r > 32:
        print(f"WARNING: lora_r={args.lora_r} exceeds competition limit of 32. Clamping to 32.",
              flush=True)
        args.lora_r = 32

    # Detect float precision: T4/V100 only support fp16; A100/H100 support bf16
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"  GPU bf16 support: {use_bf16}  → using {'bf16' if use_bf16 else 'fp16'}", flush=True)

    print(f"Loading corpus: {corpus_path}", flush=True)
    records = load_corpus(corpus_path)
    print(f"  records: {len(records)}", flush=True)
    if not records:
        print("ERROR: corpus is empty", file=sys.stderr)
        sys.exit(1)

    cats = Counter(r.get("category", "unknown") for r in records)
    for cat, cnt in cats.most_common():
        print(f"  {cat:20s}: {cnt}", flush=True)

    # Convert to messages format and apply chat template via tokenizer
    msg_records = [record_to_messages(r) for r in records]
    dataset = Dataset.from_list(msg_records)
    print(f"  Dataset rows: {len(dataset)}", flush=True)

    # ---- Model loading (4-bit QLoRA) ----
    print(f"\nLoading base model: {args.model_name}", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA config ----
    # For Nemotron-3-Nano-30B (Mamba-Transformer MoE):
    #   - q_proj / k_proj / v_proj / o_proj: attention projections
    #   - gate_proj / up_proj / down_proj: MLP/expert projections
    #   - in_proj / out_proj: Mamba SSM state projections
    # Use --target_modules to override if the model variant differs.
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",")]
    else:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj", "out_proj",
        ]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # ---- SFT Trainer ----
    # SFTTrainer accepts a "messages" column and applies the tokenizer's chat template
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_len,
        packing=False,
    )

    print("\nStarting training...", flush=True)
    trainer.train()

    # ---- Save adapter only ----
    print(f"\nSaving adapter to {output_dir}", flush=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    required = ["adapter_model.safetensors", "adapter_config.json"]
    for fname in required:
        fpath = output_dir / fname
        if fpath.exists():
            print(f"  [OK] {fname}  ({fpath.stat().st_size / 1e6:.1f} MB)", flush=True)
        else:
            print(f"  [MISSING] {fname}", flush=True)

    print("\nTraining complete.", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA SFT for Nemotron Reasoning Challenge")
    p.add_argument("--corpus",      required=True,
                   help="Path to corpus JSONL (e.g. reports/cryptarithm/corpus.jsonl)")
    p.add_argument("--output_dir",  required=True,
                   help="Directory to save adapter_model.safetensors + adapter_config.json")
    p.add_argument("--model_name",
                   default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                   help="HuggingFace model ID for base model")
    p.add_argument("--lora_r",      type=int,   default=16,
                   help="LoRA rank (competition max: 32)")
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--target_modules", default="",
                   help="Comma-separated LoRA target module names (default: Nemotron MoE set)")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=1,
                   help="Per-device batch size (30B model fits at 1 on 2xT4)")
    p.add_argument("--grad_accum",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int,   default=2048)
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
