"""
train_sft.py

SFT (Supervised Fine-Tuning) with LoRA for the NVIDIA Nemotron Reasoning Challenge.

Reads a corpus JSONL file (produced by run_all_categories.py or prepare_corpus.py),
formats each record as a chat-style instruction, and fine-tunes the base Nemotron model
with QLoRA (4-bit + LoRA) using TRL SFTTrainer.

Output:
  {output_dir}/adapter_model.safetensors
  {output_dir}/adapter_config.json

These two files are the required contents of submission.zip.

Usage (Kaggle notebook):
    !python scripts/train_sft.py \\
        --corpus reports/cryptarithm/corpus.jsonl \\
        --output_dir /kaggle/working/adapter \\
        --model_name nvidia/Llama-3.1-Nemotron-Nano-8B-v1 \\
        --lora_r 16 \\
        --lora_alpha 32 \\
        --epochs 3 \\
        --batch_size 4 \\
        --grad_accum 4 \\
        --lr 2e-4 \\
        --max_seq_len 1024

Environment:
    pip install -q transformers trl peft accelerate bitsandbytes datasets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise reasoning assistant. "
    "Think step by step, show your work clearly, then give the final answer."
)

CATEGORY_HINTS: Dict[str, str] = {
    "gravity": "Use d = 0.5 * g * t^2. Fit g from examples via least-squares, then compute the answer.",
    "unit_conversion": "Identify the conversion factor from examples, then apply it to the test value.",
    "numeral": "Apply the Roman numeral system to convert the given number.",
    "cipher": "Extract the character substitution mapping from examples, then decode the test input.",
    "bit_manipulation": "Identify the bitwise operation pattern from 8-bit examples, then apply it.",
    "equation": "Identify the string-transformation rule from examples, then apply it to the test case.",
}


def format_record(record: Dict) -> str:
    """
    Convert a CoT record into a single training string using the Nemotron chat template.

    Format:
      <|system|>...<|end|>
      <|user|>...<|end|>
      <|assistant|>...<|end|>

    Falls back to a plain instruction format if the tokenizer template is not applied.
    """
    category = record.get("category", "")
    prompt = record.get("prompt", record.get("question", ""))
    cot = record.get("cot", "")
    answer = record.get("answer", "")

    hint = CATEGORY_HINTS.get(category, "")
    system = SYSTEM_PROMPT + (f"\n{hint}" if hint else "")

    user_msg = prompt.strip()

    # Combine CoT and final answer in the assistant turn
    if cot:
        assistant_msg = cot.strip() + f"\n\nFinal answer: {answer}"
    else:
        assistant_msg = f"Final answer: {answer}"

    # Nemotron / Llama-3 chat template tokens
    text = (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user_msg}<|end|>\n"
        f"<|assistant|>\n{assistant_msg}<|end|>"
    )
    return text


def load_corpus(path: Path) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("verified", True):  # include if verified or field absent
                    records.append(r)
            except json.JSONDecodeError:
                pass
    return records


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # Imports deferred so the script can be imported without GPU dependencies
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    corpus_path = Path(args.corpus)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus: {corpus_path}", flush=True)
    records = load_corpus(corpus_path)
    print(f"  records: {len(records)}", flush=True)
    if not records:
        print("ERROR: corpus is empty", file=sys.stderr)
        sys.exit(1)

    # Category distribution
    from collections import Counter
    cats = Counter(r.get("category", "unknown") for r in records)
    for cat, cnt in cats.most_common():
        print(f"  {cat:20s}: {cnt}", flush=True)

    # Build formatted text column
    texts = [format_record(r) for r in records]
    dataset = Dataset.from_dict({"text": texts})
    print(f"  Dataset rows: {len(dataset)}", flush=True)

    # ---- Model loading (4-bit QLoRA) ----
    print(f"\nLoading base model: {args.model_name}", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
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
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA config ----
    # Target all linear projection layers that exist in the model
    target_modules = args.target_modules.split(",") if args.target_modules else [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
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
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    # ---- SFT Trainer ----
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
    )

    print("\nStarting training...", flush=True)
    trainer.train()

    # ---- Save adapter only ----
    print(f"\nSaving adapter to {output_dir}", flush=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Verify required files exist
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
    p.add_argument("--model_name",  default="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                   help="HuggingFace model ID for base model")
    p.add_argument("--lora_r",      type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--target_modules", default="",
                   help="Comma-separated LoRA target modules (default: standard Llama projections)")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int,   default=1024)
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
