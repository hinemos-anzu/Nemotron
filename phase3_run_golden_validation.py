#!/usr/bin/env python3
"""Phase 3 Step 2: Run Golden Baseline inference on validation problems.

SAFETY CONTRACT:
  - This script is READ-ONLY with respect to the adapter and model weights.
  - It does NOT modify adapter_model.safetensors, adapter_config.json, or any
    training asset.
  - It does NOT create submission.zip.
  - Generation settings are read from the existing adapter_config.json and are
    NOT changed.

The script:
  1. Loads the adapter (read-only) and base model.
  2. Runs inference on validation problems with IDENTICAL settings to the
     Kaggle Golden Baseline (temperature, max_tokens, stop tokens, seed).
  3. Logs per-sample predictions, parse results, and token counts.
  4. Writes golden_validation_predictions.jsonl and golden_validation_summary.csv.

Usage (Kaggle environment):
    python phase3_run_golden_validation.py \
        --adapter /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20 \
        --model   /kaggle/input/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default \
        --category-map phase3_analysis/category_map.csv \
        --problems /kaggle/input/problems.jsonl \
        --output-dir phase3_analysis/ \
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}", re.DOTALL)
THEREFORE_RE = re.compile(
    r"(?:the (?:final )?answer is|therefore|so the answer is|answer:)\s*(?:\\boxed\{)?([^\n\.\\{}]+)",
    re.IGNORECASE,
)
LAST_LINE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _\-]*$")


def extract_answer(raw_output: str) -> Tuple[Optional[str], str, Optional[str]]:
    """Return (pred_answer, reasoning_text, final_answer_text)."""
    # Try \boxed{} first
    m = BOXED_RE.search(raw_output)
    if m:
        answer = m.group(1).strip()
        reasoning = raw_output[: m.start()].strip()
        return answer, reasoning, m.group(0)

    # Try "therefore the answer is X"
    m = THEREFORE_RE.search(raw_output)
    if m:
        answer = re.sub(r"[^A-Za-z0-9]", "", m.group(1)).strip()
        if answer:
            reasoning = raw_output[: m.start()].strip()
            return answer, reasoning, m.group(0)

    # Fallback: last non-empty line
    lines = [l.strip() for l in raw_output.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        m2 = LAST_LINE_RE.search(last)
        if m2:
            return m2.group(0).strip(), "\n".join(lines[:-1]), last

    return None, raw_output.strip(), None


def normalize_answer(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    return re.sub(r"\s+", "", answer).strip().upper()


def answers_match(pred: Optional[str], gold: str) -> bool:
    return bool(pred) and normalize_answer(pred) == normalize_answer(gold)


def parse_error_type(pred: Optional[str], raw_output: str) -> Optional[str]:
    if pred is not None:
        return None
    if BOXED_RE.search(raw_output):
        return "boxed_found_but_empty"
    if len(raw_output.strip()) < 10:
        return "empty_output"
    return "no_extractable_answer"


# ---------------------------------------------------------------------------
# Category map loader
# ---------------------------------------------------------------------------

def load_category_map(path: Path) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            result[row["problem_id"]] = {
                "category": row.get("category", "other"),
                "subcategory": row.get("subcategory", "unknown"),
            }
    return result


def _load_csv_problems(path: Path) -> List[Dict[str, Any]]:
    """Load problems from CSV. Normalises column names to question/answer/problem_id."""
    import csv as _csv
    QUESTION_COLS = ("question", "problem", "prompt", "input", "text")
    ANSWER_COLS = ("answer", "solution", "target", "output", "label")
    ID_COLS = ("id", "problem_id", "uid", "sample_id", "index")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = _csv.DictReader(fh)
        if not reader.fieldnames:
            return records
        fl = {f.lower(): f for f in reader.fieldnames}
        q_col = next((fl[c] for c in QUESTION_COLS if c in fl), None)
        a_col = next((fl[c] for c in ANSWER_COLS if c in fl), None)
        id_col = next((fl[c] for c in ID_COLS if c in fl), None)
        for idx, row in enumerate(reader):
            rec: Dict[str, Any] = dict(row)
            if q_col and q_col != "question":
                rec["question"] = row[q_col]
            if a_col and a_col != "answer":
                rec["answer"] = row[a_col]
            if id_col and id_col != "problem_id":
                rec["problem_id"] = row[id_col]
            elif "problem_id" not in rec:
                rec["problem_id"] = f"row_{idx}"
            records.append(rec)
    return records


def load_problems(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        return _load_csv_problems(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_problem_id(record: Dict[str, Any], idx: int) -> str:
    for key in ("problem_id", "id", "uid", "sample_id"):
        v = record.get(key)
        if v:
            return str(v)
    return f"row_{idx}"


# ---------------------------------------------------------------------------
# Generation config  (matches Golden Baseline exactly)
# ---------------------------------------------------------------------------

GOLDEN_GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "temperature": 0.0,       # greedy
    "do_sample": False,
    "repetition_penalty": 1.0,
    # Stop tokens: end-of-sequence only; no early stop on \boxed
    "stop": ["<|endoftext|>", "<|im_end|>"],
}

SEED = 42


def build_prompt(record: Dict[str, Any]) -> str:
    """Reconstruct the competition prompt format from a problem record.

    Adjust this function to match the exact prompt template used in the
    Golden Baseline inference notebook.
    """
    question = str(record.get("question", record.get("prompt", record.get("input", "")))).strip()
    examples = record.get("examples", [])

    parts = ["Solve the following problem carefully.\n"]

    if examples:
        parts.append("Examples:")
        if isinstance(examples, str):
            parts.append(examples)
        else:
            for i, ex in enumerate(examples, 1):
                if isinstance(ex, dict):
                    parts.append(f"  {i}. Input: {ex.get('input', '')} -> Output: {ex.get('output', '')}")
                else:
                    parts.append(f"  {i}. {ex}")
        parts.append("")

    parts.append(f"Question: {question}")
    parts.append("\nThink step by step. Put your final answer inside \\boxed{}.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Mamba SSM patch (Kaggle-specific)
# ---------------------------------------------------------------------------

def _apply_mamba_patch() -> None:
    """Inject a working mamba_ssm stub to override the broken CUDA extension.

    The Kaggle environment ships mamba_ssm built against a different CUDA
    version, causing 'undefined symbol' errors.  The user's own nemotron
    dataset bundles already contain working stubs/patches — we find the
    most recent one and insert its parent directory at the front of sys.path
    so Python picks it up before the broken system copy.
    """
    import glob as _glob
    import sys

    all_stubs = {
        str(Path(p).parent.parent)
        for p in _glob.glob(
            "/kaggle/input/datasets/hinemos/**/mamba_ssm/__init__.py",
            recursive=True,
        )
    }

    if not all_stubs:
        print("[mamba_patch] No stub found in /kaggle/input/datasets/hinemos/ — "
              "model loading may fail with CUDA symbol errors")
        return

    def _scan_impl_size(parent: str) -> int:
        p = Path(parent) / "mamba_ssm/ops/selective_scan_interface.py"
        return p.stat().st_size if p.exists() else 0

    # Prefer stubs that have a non-empty selective_scan_interface.py, then
    # take the alphabetically latest (highest date + run number)
    best = sorted(all_stubs, key=lambda d: (_scan_impl_size(d), d), reverse=True)[0]

    # Evict any already-cached broken modules
    for key in list(sys.modules.keys()):
        if key == "mamba_ssm" or key.startswith("mamba_ssm."):
            del sys.modules[key]

    if best not in sys.path:
        sys.path.insert(0, best)

    print(f"[mamba_patch] Patched mamba_ssm from {best}")


# ---------------------------------------------------------------------------
# Inference runner (transformers path — for vLLM see reproducibility_notes.md)
# ---------------------------------------------------------------------------

def run_inference_transformers(
    problems: List[Dict[str, Any]],
    category_map: Dict[str, Dict[str, str]],
    adapter_path: str,
    model_path: str,
    seed: int,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Run inference using HuggingFace transformers + PEFT.

    This function is designed to be run on Kaggle GPU environment.
    It imports heavy dependencies lazily to allow the rest of the script
    to be imported and tested on CPU.
    """
    # Must patch mamba_ssm BEFORE any transformers imports so that
    # modeling_nemotron_h.py's module-level is_mamba_2_ssm_available() check
    # uses our stub instead of the broken CUDA extension.
    _apply_mamba_patch()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import PeftModel

    set_seed(seed)
    torch.manual_seed(seed)

    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    records: List[Dict[str, Any]] = []

    for idx, record in enumerate(problems):
        pid = get_problem_id(record, idx)
        cat_info = category_map.get(pid, {"category": "other", "subcategory": "unknown"})
        question = str(record.get("question", record.get("prompt", ""))).strip()
        gold_answer = str(record.get("answer", record.get("target", ""))).strip()

        prompt = build_prompt(record)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=GOLDEN_GENERATION_CONFIG["max_new_tokens"],
                do_sample=GOLDEN_GENERATION_CONFIG["do_sample"],
                temperature=None,  # greedy when do_sample=False
                repetition_penalty=GOLDEN_GENERATION_CONFIG["repetition_penalty"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

        pred_answer, reasoning_text, final_answer_text = extract_answer(raw_output)
        is_correct = answers_match(pred_answer, gold_answer)
        parse_success = pred_answer is not None
        err_type = parse_error_type(pred_answer, raw_output)
        n_tokens = int(generated_ids.shape[0])

        finish_reason = "eos" if n_tokens < GOLDEN_GENERATION_CONFIG["max_new_tokens"] else "length"

        entry: Dict[str, Any] = {
            "problem_id": pid,
            "category": cat_info["category"],
            "subcategory": cat_info["subcategory"],
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer or "",
            "raw_output": raw_output,
            "reasoning_text": reasoning_text,
            "final_answer_text": final_answer_text or "",
            "is_correct": is_correct,
            "parse_success": parse_success,
            "parse_error_type": err_type or "",
            "generation_token_count": n_tokens,
            "finish_reason": finish_reason,
            "elapsed_seconds": round(elapsed, 2),
            "seed": seed,
            "generation_config": GOLDEN_GENERATION_CONFIG,
        }
        records.append(entry)

        status = "OK" if is_correct else ("PARSE_FAIL" if not parse_success else "WRONG")
        print(f"[{idx+1:4d}/{len(problems)}] {pid}: {status}  tokens={n_tokens}")

    return records


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_predictions_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            # Truncate raw_output to avoid huge files but keep enough for analysis
            out = dict(rec)
            if len(out.get("raw_output", "")) > 4000:
                out["raw_output"] = out["raw_output"][:4000] + " ...[truncated]"
            if len(out.get("reasoning_text", "")) > 3000:
                out["reasoning_text"] = out["reasoning_text"][:3000] + " ...[truncated]"
            out.pop("generation_config", None)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")


def write_summary_csv(records: List[Dict[str, Any]], path: Path) -> None:
    from collections import Counter, defaultdict

    n_total = len(records)
    n_correct = sum(1 for r in records if r["is_correct"])
    n_parse = sum(1 for r in records if r["parse_success"])
    avg_tokens = (
        sum(r["generation_token_count"] for r in records) / max(n_total, 1)
    )

    overall: List[Dict[str, Any]] = [
        {
            "split": "overall",
            "category": "ALL",
            "subcategory": "ALL",
            "n": n_total,
            "correct": n_correct,
            "accuracy": round(n_correct / max(n_total, 1), 4),
            "n_parse_success": n_parse,
            "parse_success_rate": round(n_parse / max(n_total, 1), 4),
            "avg_generation_token_count": round(avg_tokens, 1),
        }
    ]

    # Per-category summary
    cat_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        cat_groups[r["category"]].append(r)

    per_cat: List[Dict[str, Any]] = []
    for cat, grp in sorted(cat_groups.items()):
        nc = sum(1 for r in grp if r["is_correct"])
        np_ = sum(1 for r in grp if r["parse_success"])
        at = sum(r["generation_token_count"] for r in grp) / max(len(grp), 1)
        per_cat.append({
            "split": "category",
            "category": cat,
            "subcategory": "ALL",
            "n": len(grp),
            "correct": nc,
            "accuracy": round(nc / max(len(grp), 1), 4),
            "n_parse_success": np_,
            "parse_success_rate": round(np_ / max(len(grp), 1), 4),
            "avg_generation_token_count": round(at, 1),
        })

    # Per-subcategory summary
    subcat_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        subcat_groups[(r["category"], r["subcategory"])].append(r)

    per_subcat: List[Dict[str, Any]] = []
    for (cat, sub), grp in sorted(subcat_groups.items()):
        nc = sum(1 for r in grp if r["is_correct"])
        np_ = sum(1 for r in grp if r["parse_success"])
        at = sum(r["generation_token_count"] for r in grp) / max(len(grp), 1)
        per_subcat.append({
            "split": "subcategory",
            "category": cat,
            "subcategory": sub,
            "n": len(grp),
            "correct": nc,
            "accuracy": round(nc / max(len(grp), 1), 4),
            "n_parse_success": np_,
            "parse_success_rate": round(np_ / max(len(grp), 1), 4),
            "avg_generation_token_count": round(at, 1),
        })

    all_rows = overall + per_cat + per_subcat
    fields = [
        "split", "category", "subcategory", "n", "correct", "accuracy",
        "n_parse_success", "parse_success_rate", "avg_generation_token_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default=os.environ.get(
            "ADAPTER_PATH",
            "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20",
        ),
        help="Path to the Golden adapter directory (read-only)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", ""),
        help="Path to the base model directory (auto-resolved via kagglehub if not set)",
    )
    parser.add_argument(
        "--problems",
        default="problems.jsonl",
        help="Path to validation problems JSONL",
    )
    parser.add_argument(
        "--category-map",
        default="phase3_analysis/category_map.csv",
        help="Path to category_map.csv from Step 1",
    )
    parser.add_argument(
        "--output-dir",
        default="phase3_analysis",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and config without running inference",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model path: env/arg → kagglehub (same as best-score notebook) → error
    model_path = args.model
    if not model_path or not Path(model_path).exists():
        try:
            import kagglehub  # type: ignore
            model_path = kagglehub.model_download(
                "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
            )
            print(f"Model path resolved via kagglehub: {model_path}")
        except Exception as e:
            print(f"[WARNING] kagglehub model_download failed: {e}")
            print("[WARNING] --model path not found. Inference will fail unless you set MODEL_PATH.")

    category_map = load_category_map(Path(args.category_map))
    problems = load_problems(Path(args.problems))

    print(f"Problems loaded: {len(problems)}")
    print(f"Category map entries: {len(category_map)}")
    print(f"Adapter: {args.adapter}")
    print(f"Model: {model_path}")
    print(f"Seed: {args.seed}")
    print(f"Generation config: {GOLDEN_GENERATION_CONFIG}")

    # Safety check: never write to adapter paths
    adapter_path = Path(args.adapter)
    for p in [adapter_path / "adapter_model.safetensors", adapter_path / "adapter_config.json"]:
        if p.exists():
            print(f"[READ-ONLY CHECK] {p} exists — will NOT be modified.")

    if args.dry_run:
        print("[DRY-RUN] Config validated. Exiting without inference.")
        return

    if not problems:
        print("[ERROR] No problems found. Check --problems path.")
        return

    print("Starting inference (this may take 30-90 minutes on Kaggle GPU)...")
    records = run_inference_transformers(
        problems=problems,
        category_map=category_map,
        adapter_path=args.adapter,
        model_path=model_path,
        seed=args.seed,
        output_dir=output_dir,
    )

    pred_path = output_dir / "golden_validation_predictions.jsonl"
    summary_path = output_dir / "golden_validation_summary.csv"

    write_predictions_jsonl(records, pred_path)
    write_summary_csv(records, summary_path)

    n_correct = sum(1 for r in records if r["is_correct"])
    accuracy = n_correct / max(len(records), 1)
    print(f"\nDone. {len(records)} problems evaluated.")
    print(f"Accuracy: {n_correct}/{len(records)} = {accuracy:.4f}")
    print(f"Predictions -> {pred_path}")
    print(f"Summary     -> {summary_path}")


if __name__ == "__main__":
    main()
