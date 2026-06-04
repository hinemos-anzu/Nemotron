#!/usr/bin/env python3
"""Phase 3 Step 3: Extract token-level logprob metrics from inference logs.

Two modes:
  1. POST-HOC MODE (default): Reads an existing golden_validation_predictions.jsonl
     and re-runs inference with logprob logging enabled (score_each_tokens=True in
     vLLM, or output_scores=True in transformers).
  2. INLINE MODE: If predictions.jsonl already contains per-token logprobs (e.g.,
     stored by phase3_run_golden_validation.py with --save-logprobs), parse them
     directly without re-running inference.

Output: min_logprob_summary.csv

SAFETY CONTRACT: Read-only. No adapter, model, or submission files are modified.

Usage (post-hoc re-inference on Kaggle):
    python phase3_extract_logprob.py \
        --predictions phase3_analysis/golden_validation_predictions.jsonl \
        --adapter /kaggle/input/... \
        --model /kaggle/input/... \
        --output phase3_analysis/min_logprob_summary.csv \
        --mode rerun

Usage (from inline logprob data in predictions.jsonl):
    python phase3_extract_logprob.py \
        --predictions phase3_analysis/golden_validation_predictions.jsonl \
        --output phase3_analysis/min_logprob_summary.csv \
        --mode inline
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOW_LOGPROB_THRESHOLD = -2.0  # tokens with logprob < this are "low confidence"
ANSWER_MARKER_RE = re.compile(r"\\boxed\{|the answer is|therefore", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Logprob utilities
# ---------------------------------------------------------------------------

def compute_min_logprob_stats(
    token_logprobs: List[float],
    token_texts: List[str],
    answer_start_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute min logprob, mean logprob, and low-confidence token counts."""
    if not token_logprobs:
        return {
            "min_logprob": None,
            "mean_logprob": None,
            "answer_min_logprob": None,
            "answer_mean_logprob": None,
            "low_conf_token_count": 0,
            "lowest_logprob_token": None,
            "lowest_logprob_context": None,
            "answer_low_conf_token_count": 0,
        }

    min_lp = min(token_logprobs)
    mean_lp = sum(token_logprobs) / len(token_logprobs)
    min_idx = token_logprobs.index(min_lp)
    low_count = sum(1 for lp in token_logprobs if lp < LOW_LOGPROB_THRESHOLD)

    # Context window around lowest logprob token
    ctx_start = max(0, min_idx - 5)
    ctx_end = min(len(token_texts), min_idx + 6)
    context = "".join(token_texts[ctx_start:ctx_end])

    # Answer segment (from answer marker to end)
    if answer_start_idx is not None and answer_start_idx < len(token_logprobs):
        ans_lps = token_logprobs[answer_start_idx:]
        ans_min = min(ans_lps) if ans_lps else None
        ans_mean = sum(ans_lps) / len(ans_lps) if ans_lps else None
        ans_low_count = sum(1 for lp in ans_lps if lp < LOW_LOGPROB_THRESHOLD)
    else:
        ans_min = None
        ans_mean = None
        ans_low_count = 0

    return {
        "min_logprob": round(min_lp, 4),
        "mean_logprob": round(mean_lp, 4),
        "answer_min_logprob": round(ans_min, 4) if ans_min is not None else None,
        "answer_mean_logprob": round(ans_mean, 4) if ans_mean is not None else None,
        "low_conf_token_count": low_count,
        "lowest_logprob_token": token_texts[min_idx] if token_texts else None,
        "lowest_logprob_context": context,
        "answer_low_conf_token_count": ans_low_count,
    }


def find_answer_start_idx(token_texts: List[str]) -> Optional[int]:
    """Find the token index where the answer section begins."""
    combined = ""
    for idx, tok in enumerate(token_texts):
        combined += tok
        if ANSWER_MARKER_RE.search(combined[-100:]):  # sliding window
            return idx
    return None


# ---------------------------------------------------------------------------
# Inline mode: parse logprob data already stored in predictions.jsonl
# ---------------------------------------------------------------------------

def extract_inline_logprobs(
    predictions_path: Path,
    output_path: Path,
) -> None:
    """Parse inline logprob data from predictions.jsonl."""
    rows: List[Dict[str, Any]] = []

    with predictions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # Inline logprobs stored as list of {"token": ..., "logprob": ...}
            token_data = rec.get("token_logprobs", [])
            if not token_data:
                # No inline data — produce None-filled row
                rows.append({
                    "problem_id": rec["problem_id"],
                    "category": rec.get("category", "other"),
                    "subcategory": rec.get("subcategory", "unknown"),
                    "is_correct": rec.get("is_correct", False),
                    "parse_success": rec.get("parse_success", False),
                    "min_logprob": None,
                    "mean_logprob": None,
                    "answer_min_logprob": None,
                    "answer_mean_logprob": None,
                    "low_conf_token_count": None,
                    "lowest_logprob_token": None,
                    "lowest_logprob_context": None,
                    "answer_low_conf_token_count": None,
                    "logprob_source": "missing",
                })
                continue

            token_lps = [float(t["logprob"]) for t in token_data]
            token_texts = [str(t["token"]) for t in token_data]
            ans_idx = find_answer_start_idx(token_texts)
            stats = compute_min_logprob_stats(token_lps, token_texts, ans_idx)

            rows.append({
                "problem_id": rec["problem_id"],
                "category": rec.get("category", "other"),
                "subcategory": rec.get("subcategory", "unknown"),
                "is_correct": rec.get("is_correct", False),
                "parse_success": rec.get("parse_success", False),
                "logprob_source": "inline",
                **stats,
            })

    _write_logprob_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


# ---------------------------------------------------------------------------
# Re-run mode: re-infer with logprobs enabled
# ---------------------------------------------------------------------------

def extract_logprobs_rerun(
    predictions_path: Path,
    adapter_path: str,
    model_path: str,
    output_path: Path,
    seed: int = 42,
) -> None:
    """Re-run inference with token-level logprob logging."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import PeftModel

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Load existing predictions to get prompts and metadata
    records: List[Dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    rows: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        question = rec.get("question", "")
        # Reconstruct minimal prompt (question only — same as Golden Baseline)
        prompt = f"Solve the following problem carefully.\n\nQuestion: {question}\n\nThink step by step. Put your final answer inside \\boxed{{}}."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        scores = out.scores  # tuple of (vocab_size,) tensors

        token_lps: List[float] = []
        token_texts: List[str] = []
        for step_idx, (tok_id, score_tensor) in enumerate(zip(generated_ids, scores)):
            log_probs = torch.log_softmax(score_tensor, dim=-1)
            lp = float(log_probs[0, tok_id].item())
            token_lps.append(lp)
            token_texts.append(tokenizer.decode([tok_id.item()]))

        ans_idx = find_answer_start_idx(token_texts)
        stats = compute_min_logprob_stats(token_lps, token_texts, ans_idx)

        rows.append({
            "problem_id": rec["problem_id"],
            "category": rec.get("category", "other"),
            "subcategory": rec.get("subcategory", "unknown"),
            "is_correct": rec.get("is_correct", False),
            "parse_success": rec.get("parse_success", False),
            "logprob_source": "rerun",
            **stats,
        })
        print(f"[{idx+1}/{len(records)}] {rec['problem_id']}: min_logprob={stats['min_logprob']}")

    _write_logprob_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_logprob_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "problem_id", "category", "subcategory",
        "is_correct", "parse_success",
        "min_logprob", "mean_logprob",
        "answer_min_logprob", "answer_mean_logprob",
        "low_conf_token_count", "lowest_logprob_token",
        "lowest_logprob_context", "answer_low_conf_token_count",
        "logprob_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="phase3_analysis/golden_validation_predictions.jsonl",
        help="Path to golden_validation_predictions.jsonl",
    )
    parser.add_argument(
        "--output",
        default="phase3_analysis/min_logprob_summary.csv",
        help="Output min_logprob_summary.csv path",
    )
    parser.add_argument(
        "--mode",
        choices=["inline", "rerun"],
        default="inline",
        help="'inline': parse existing logprob data; 'rerun': re-infer with logprobs",
    )
    parser.add_argument(
        "--adapter",
        default=os.environ.get("ADAPTER_PATH", ""),
        help="[rerun mode only] Adapter path",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", ""),
        help="[rerun mode only] Base model path",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"[ERROR] {pred_path} not found. Run phase3_run_golden_validation.py first.")
        return

    if args.mode == "inline":
        extract_inline_logprobs(pred_path, Path(args.output))
    else:
        if not args.adapter or not args.model:
            print("[ERROR] --adapter and --model are required in rerun mode.")
            return
        extract_logprobs_rerun(
            pred_path, args.adapter, args.model, Path(args.output), args.seed
        )


if __name__ == "__main__":
    main()
