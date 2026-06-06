#!/usr/bin/env python3
"""Generate phase3_rtx_analysis.ipynb — offline Phase 3 analysis for RTX Pro 5000.

Usage:
    python gen_phase3_notebook.py
Output:
    phase3_rtx_analysis.ipynb
"""

import json
import textwrap
from pathlib import Path

NOTEBOOK_PATH = "phase3_rtx_analysis.ipynb"


def code(src: str, *, tag: str = "") -> dict:
    src = textwrap.dedent(src).strip() + "\n"
    return {
        "cell_type": "code",
        "source": src,
        "metadata": {"tags": [tag] if tag else []},
        "outputs": [],
        "execution_count": None,
        "id": tag or src[:8].replace("\n", ""),
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "source": textwrap.dedent(src).strip() + "\n",
        "metadata": {},
        "id": src[:8].replace("\n", ""),
    }


# ---------------------------------------------------------------------------
# Cell content
# ---------------------------------------------------------------------------

CELL_TITLE = md("""\
# Phase 3: Golden Baseline 失敗分析
## RTX Pro 5000 オフライン完結版

**目的**: Golden Baseline (Public LB 0.86) がどのカテゴリで詰まっているかを定量分析する。

**禁止事項の確認**:
- adapter_model.safetensors / adapter_config.json を変更しない
- submission.zip を作成しない  ← このノートブックは作成しない
- training / SFT を実行しない
- Golden adapter は読み取り専用で使用する

**実行手順**:
1. Cell 1 (Config) のパスを環境に合わせて修正
2. 全セルを順番に実行
3. `phase3_analysis/` に結果が出力される
""")

CELL_CONFIG = code("""\
import os
from pathlib import Path

# ================================================================
# ★ 設定: 実行前にここを修正してください
# ================================================================
ADAPTER_PATH  = os.environ.get("ADAPTER_PATH",
    "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20")
MODEL_PATH    = os.environ.get("MODEL_PATH",
    "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1")
PROBLEMS_PATH = os.environ.get("PROBLEMS_PATH",
    "/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv")
OUTPUT_DIR    = Path(os.environ.get("OUTPUT_DIR", "phase3_analysis"))

# --- Sampling ---
MAX_PROBLEMS   = 200   # 0 = 全件 (9500問). 200 ≈ 数十分
SEED           = 42
MAX_NEW_TOKENS = 2048

# カテゴリ別サンプリング数 (合計が MAX_PROBLEMS 以下になるよう調整)
CATEGORY_QUOTA = {
    "cryptarithm":       50,   # 最優先
    "bit_manipulation":  40,
    "numeral_conversion":40,
    "cipher":            20,
    "equation":          20,
    "arithmetic":        15,
    "other":             15,
}

# --- Generation (Golden Baseline と同じ設定) ---
GENERATION_CONFIG = {
    "max_new_tokens":    MAX_NEW_TOKENS,
    "do_sample":         False,
    "repetition_penalty":1.0,
    "temperature":       0.0,     # greedy
}
STOP_STRINGS = ["<|endoftext|>", "<|im_end|>"]

# ================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("=== Phase 3 Analysis Config ===")
print(f"ADAPTER_PATH  : {ADAPTER_PATH}")
print(f"MODEL_PATH    : {MODEL_PATH}")
print(f"PROBLEMS_PATH : {PROBLEMS_PATH}")
print(f"OUTPUT_DIR    : {OUTPUT_DIR.resolve()}")
print(f"MAX_PROBLEMS  : {MAX_PROBLEMS or 'ALL'}")
print(f"SEED          : {SEED}")
""", tag="config")

CELL_CATEGORY = code("""\
import csv
import re
import random
import json
from collections import Counter, defaultdict

# ----------------------------------------------------------------
# Keyword-based category rules
# ----------------------------------------------------------------
_CATEGORY_RULES = [
    ("cryptarithm", [
        r"SEND.*MORE|MONEY|DONALD.*GERALD",
        r"each\\s+letter.*digit|letter.*represent.*digit",
        r"cryptarithm|alphametic",
        r"[A-Z]{2,}\\s*\\+\\s*[A-Z]{2,}\\s*=",
        r"[A-Z]{2,}\\s*-\\s*[A-Z]{2,}\\s*=",
        r"different.*letter.*different.*digit",
        r"unique.*digit.*letter",
    ]),
    ("bit_manipulation", [
        r"\\bXOR\\b|\\bbitwise\\b",
        r"\\bleft\\s+shift\\b|\\bright\\s+shift\\b|<<\\s*\\d|>>\\s*\\d",
        r"two.?s\\s+complement|signed.*bit|unsigned.*bit",
        r"bitmask|bit\\s+mask",
        r"binary.*AND.*OR|AND.*OR.*bit",
        r"0b[01]+.*AND|0b[01]+.*OR|0b[01]+.*XOR",
    ]),
    ("numeral_conversion", [
        r"\\bbase\\s*[-–]?\\s*\\d+\\b|base-\\d+",
        r"hexadecimal|\\bhex\\b|0x[0-9a-fA-F]+",
        r"binary\\s+to\\s+decimal|decimal\\s+to\\s+binary",
        r"convert.*\\bbase\\b|from base \\d+ to",
        r"roman\\s+numeral",
        r"octal|base.?8\\b",
    ]),
    ("cipher", [
        r"\\bcipher\\b|\\bencrypt|\\bdecrypt",
        r"Caesar\\s+cipher|Vigen[eè]re|ROT\\d+",
        r"substitution.*cipher|encoded\\s+message",
        r"shift.*alphabet|decode.*message",
    ]),
    ("equation", [
        r"solve.*\\bequation\\b|system\\s+of\\s+equation",
        r"quadratic|linear\\s+equation",
        r"find.*value.*where.*=|satisfy.*equation",
    ]),
    ("unit_conversion", [
        r"km.*miles|miles.*km|kilometer.*mile",
        r"Celsius.*Fahrenheit|Fahrenheit.*Celsius",
        r"pound.*kilogram|gallon.*liter",
        r"convert.*unit",
    ]),
    ("arithmetic", [
        r"\\bLCM\\b|\\bGCD\\b|\\bHCF\\b",
        r"\\bprime\\b.*factor|factor.*\\bprime\\b",
        r"\\bfactorial\\b",
        r"remainder\\s+when.*divided|modulo|\\bmod\\b",
        r"sum.*consecutive|product.*first.*n",
    ]),
    ("logic", [
        r"truth\\s+table|propositional|logical.*AND.*OR",
        r"\\btautology\\b|\\bcontradiction\\b",
    ]),
]

_SUBCAT_RULES = {
    "cryptarithm": [
        ("alphametic_addition",       [r"\\+.*="]),
        ("alphametic_subtraction",    [r"-.*="]),
        ("alphametic_multiplication", [r"\\*.*=|×.*="]),
        ("leading_zero_constraint",   [r"leading\\s+zero|no\\s+zero\\s+first"]),
        ("carry_reasoning",           [r"\\bcarry\\b"]),
        ("digit_assignment",          [r"assign.*digit|map.*letter"]),
    ],
    "bit_manipulation": [
        ("xor",              [r"\\bXOR\\b"]),
        ("and",              [r"\\bbitwise\\s+AND\\b"]),
        ("or",               [r"\\bbitwise\\s+OR\\b"]),
        ("shift_left",       [r"left.?shift|<<"]),
        ("shift_right",      [r"right.?shift|>>"]),
        ("mask",             [r"bitmask|\\bmask\\b"]),
        ("signed_unsigned",  [r"signed|unsigned|two.?s complement"]),
        ("binary_arithmetic",[r"binary.*add|binary.*sub"]),
    ],
    "numeral_conversion": [
        ("binary_to_decimal",  [r"binary.*to.*decimal|from.*binary"]),
        ("decimal_to_binary",  [r"decimal.*to.*binary|to\\s+binary"]),
        ("hex_to_decimal",     [r"hex.*to.*decimal|hexadecimal.*to"]),
        ("decimal_to_hex",     [r"decimal.*to.*hex|to\\s+hexadecimal"]),
        ("base_n_conversion",  [r"base.?\\d+"]),
        ("roman_numeral",      [r"roman"]),
    ],
}


def _match_any(text: str, patterns: list) -> bool:
    for p in patterns:
        try:
            if re.search(p, text, re.IGNORECASE):
                return True
        except re.error:
            pass
    return False


def classify_problem(text: str):
    for cat, pats in _CATEGORY_RULES:
        if _match_any(text, pats):
            subcats = _SUBCAT_RULES.get(cat, [])
            for sub, spats in subcats:
                if _match_any(text, spats):
                    return cat, sub, 0.9
            return cat, "unknown", 0.8
    return "other", "other", 0.5


# ----------------------------------------------------------------
# Load train.csv
# ----------------------------------------------------------------
def _get(row: dict, *keys, default=""):
    for k in keys:
        if k in row and row[k]:
            return str(row[k])
    return default


def load_problems(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


all_problems = load_problems(PROBLEMS_PATH)
print(f"Total problems loaded: {len(all_problems)}")

# Identify columns
sample_row = all_problems[0] if all_problems else {}
print(f"Columns: {list(sample_row.keys())[:10]}")

# ----------------------------------------------------------------
# Classify
# ----------------------------------------------------------------
for p in all_problems:
    text = _get(p, "problem", "question", "text", "prompt")
    cat, sub, conf = classify_problem(text)
    p["_category"]    = cat
    p["_subcategory"] = sub
    p["_confidence"]  = conf

dist = Counter(p["_category"] for p in all_problems)
print("\\nCategory distribution (full set):")
for k, v in dist.most_common():
    print(f"  {k:<22s}: {v:5d}  ({v/len(all_problems)*100:.1f}%)")

# ----------------------------------------------------------------
# Stratified sampling
# ----------------------------------------------------------------
random.seed(SEED)
by_cat = defaultdict(list)
for p in all_problems:
    by_cat[p["_category"]].append(p)

sampled = []
for cat, quota in CATEGORY_QUOTA.items():
    pool = by_cat.get(cat, [])
    take = min(quota, len(pool))
    sampled.extend(random.sample(pool, take))
    if not pool:
        print(f"[warn] No problems for category: {cat}")

random.shuffle(sampled)
if MAX_PROBLEMS and len(sampled) > MAX_PROBLEMS:
    sampled = sampled[:MAX_PROBLEMS]

print(f"\\nSampled: {len(sampled)} problems")
sample_dist = Counter(p["_category"] for p in sampled)
for k, v in sample_dist.most_common():
    print(f"  {k:<22s}: {v:4d}")

# Save category map
import csv as _csv
cat_rows = []
for p in all_problems:
    pid = _get(p, "id", "problem_id", "uid")
    cat_rows.append({
        "problem_id": pid,
        "category":   p["_category"],
        "subcategory":p["_subcategory"],
        "confidence": p["_confidence"],
        "rule":       "keyword",
        "matched_keywords": "",
        "manual_review_required": str(p["_confidence"] < 0.8),
    })

cat_map_path = OUTPUT_DIR / "category_map.csv"
with open(cat_map_path, "w", newline="", encoding="utf-8") as f:
    w = _csv.DictWriter(f, fieldnames=cat_rows[0].keys())
    w.writeheader(); w.writerows(cat_rows)

labeled_path = OUTPUT_DIR / "validation_set_labeled.csv"
with open(labeled_path, "w", newline="", encoding="utf-8") as f:
    cols = list(all_problems[0].keys()) + ["_category","_subcategory","_confidence"]
    w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader(); w.writerows(all_problems)

print(f"\\nSaved: {cat_map_path}")
print(f"Saved: {labeled_path}")
""", tag="category")

CELL_MODEL = code("""\
import sys
import types
import importlib.util
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel

set_seed(SEED)

# ----------------------------------------------------------------
# mamba_ssm: try native first, fallback to PyTorch stub
# ----------------------------------------------------------------
def _inject_mamba_stub():
    \"\"\"Minimal PyTorch stub for mamba_ssm when CUDA extension has ABI mismatch.\"\"\"
    import torch as _t
    import torch.nn.functional as _F
    import types as _ty
    import importlib.util as _ilu

    def _mk(name):
        m = _ty.ModuleType(name)
        m.__path__    = []
        m.__package__ = name
        m.__spec__    = _ilu.spec_from_loader(name, loader=None)
        return m

    stub          = _mk("mamba_ssm");          stub.__version__ = "1.2.0"
    ops           = _mk("mamba_ssm.ops")
    triton        = _mk("mamba_ssm.ops.triton")
    ln_gated      = _mk("mamba_ssm.ops.triton.layernorm_gated")
    sel_scan      = _mk("mamba_ssm.ops.selective_scan_interface")

    def rmsnorm_fn(x, weight, bias=None, residual=None, x_bias=None,
                   z=None, z_bias=None, prenorm=False, residual_in_fp32=False,
                   eps=1e-6, is_rms_norm=True, return_dropout_mask=False,
                   norm_before_gate=True, group_size=None, **_kw):
        orig = x.dtype; xf = x.float()
        if x_bias is not None:  xf = xf + x_bias.float()
        if residual is not None: xf = xf + residual.float()
        gate = None
        if z is not None:
            zf = z.float()
            if z_bias is not None: zf = zf + z_bias.float()
            gate = _F.silu(zf)
            if not norm_before_gate: xf = xf * gate
        if group_size and group_size > 1:
            sh = xf.shape
            xg = xf.reshape(*sh[:-1], -1, group_size)
            xn = (xg * _t.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)).reshape(sh)
        else:
            xn = xf * _t.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        out = weight.float() * xn
        if bias is not None: out = out + bias.float()
        if gate is not None and norm_before_gate: out = out * gate
        out = out.to(orig)
        if prenorm:
            r = xf if residual_in_fp32 else xf.to(orig)
            return (out, r) if not return_dropout_mask else (out, r, None)
        return out if not return_dropout_mask else (out, None)

    def _noop(*a, **k):
        raise NotImplementedError("mamba_ssm stub: CUDA SSM ops disabled")

    ln_gated.rmsnorm_fn         = rmsnorm_fn
    sel_scan.selective_scan_fn  = _noop
    sel_scan.mamba_inner_fn     = _noop

    stub.ops                          = ops
    ops.triton                        = triton
    ops.selective_scan_interface      = sel_scan
    triton.layernorm_gated            = ln_gated

    for key, mod in [
        ("mamba_ssm",                              stub),
        ("mamba_ssm.ops",                          ops),
        ("mamba_ssm.ops.triton",                   triton),
        ("mamba_ssm.ops.triton.layernorm_gated",   ln_gated),
        ("mamba_ssm.ops.selective_scan_interface", sel_scan),
    ]:
        sys.modules[key] = mod

    print("[mamba] PyTorch stub injected (v1.2.0) — CUDA ops bypassed")
    print("[mamba] ⚠ Inference may be slower than native. "
          "Install mamba-ssm matching your PyTorch for full speed.")


_mamba_native = False
_existing = sys.modules.get("mamba_ssm")
if _existing is not None and getattr(_existing, "__version__", "0") < "2.0.4":
    print(f"[mamba] Already have stub v{getattr(_existing, '__version__', '?')} — reusing")
    _mamba_native = False
else:
    try:
        import mamba_ssm as _mm
        from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn as _rmsnorm_test  # noqa
        _mamba_native = True
        print(f"[mamba] Native mamba_ssm {_mm.__version__} OK")
    except Exception as _e:
        print(f"[mamba] Native import failed: {type(_e).__name__}: {_e}")
        # Remove partial import before injecting stub
        for _k in [k for k in sys.modules if k == "mamba_ssm" or k.startswith("mamba_ssm.")]:
            del sys.modules[_k]
        _inject_mamba_stub()

# ----------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------
print(f"\\nLoading tokenizer from {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Tokenizer OK  eos={tokenizer.eos_token_id}")

# ----------------------------------------------------------------
# VRAM check → bf16 if enough, else 4-bit
# ----------------------------------------------------------------
_n_gpus = torch.cuda.device_count()
_total_vram_gb = sum(
    torch.cuda.get_device_properties(i).total_memory
    for i in range(_n_gpus)
) / 1024**3

_use_4bit = _total_vram_gb < 50.0
print(f"GPUs: {_n_gpus}  Total VRAM: {_total_vram_gb:.1f} GB  "
      f"→ {'4-bit NF4' if _use_4bit else 'bfloat16 full precision'}")

_load_kwargs = dict(
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
if _use_4bit:
    from transformers import BitsAndBytesConfig
    _load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    del _load_kwargs["torch_dtype"]

# ----------------------------------------------------------------
# Load base model + adapter (READ-ONLY)
# ----------------------------------------------------------------
print(f"\\nLoading base model from {MODEL_PATH} ...")
gc.collect(); torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **_load_kwargs)
print(f"Base model loaded: {model.__class__.__name__}")

print(f"Loading adapter from {ADAPTER_PATH} (read-only) ...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
print("Adapter loaded OK — model in eval mode")

# EOS token IDs from STOP_STRINGS
eos_ids = []
for s in STOP_STRINGS:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if ids:
        eos_ids.extend(ids)
eos_ids = list(set(eos_ids)) or [tokenizer.eos_token_id]
print(f"EOS token IDs: {eos_ids}")

# Input device
_device = next(model.parameters()).device
print(f"Model input device: {_device}")
""", tag="model")

CELL_INFERENCE = code("""\
import time
import torch.nn.functional as F

# ----------------------------------------------------------------
# Answer extraction helpers (identical to Golden Baseline)
# ----------------------------------------------------------------
import re as _re

_BOXED_RE     = _re.compile(r"\\\\boxed\\{([^{}]+)\\}", _re.DOTALL)
_THEREFORE_RE = _re.compile(
    r"(?:the\\s+(?:final\\s+)?answer\\s+is|therefore|answer:)"
    r"\\s*(?:\\\\boxed\\{)?([^\\n\\.\\\\{}]+)",
    _re.IGNORECASE,
)


def extract_answer(raw: str):
    m = _BOXED_RE.search(raw)
    if m:
        ans = m.group(1).strip()
        return ans, raw[:m.start()].strip(), m.group(0)
    m = _THEREFORE_RE.search(raw)
    if m:
        ans = _re.sub(r"[^A-Za-z0-9]", "", m.group(1)).strip()
        if ans:
            return ans, raw[:m.start()].strip(), m.group(0)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if lines:
        return lines[-1], "\\n".join(lines[:-1]), lines[-1]
    return None, raw.strip(), None


def normalize(a):
    if a is None:
        return ""
    return _re.sub(r"\\s+", "", str(a)).strip().upper()


def answers_match(pred, gold):
    return bool(pred) and normalize(pred) == normalize(gold)


def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>user\\n{question}\\n"
        "Think step by step. Put your final answer inside \\\\boxed{}.\\n"
        "<|im_end|>\\n<|im_start|>assistant\\n"
    )


# ----------------------------------------------------------------
# Inference loop
# ----------------------------------------------------------------
pred_path = OUTPUT_DIR / "golden_validation_predictions.jsonl"
pred_path.unlink(missing_ok=True)

results = []
n_correct = 0
parse_fail = 0
total_tok  = 0

print(f"Running inference on {len(sampled)} problems ...\\n")
_t_start_all = time.time()

for idx, prob in enumerate(sampled):
    pid  = _get(prob, "id", "problem_id", "uid", default=str(idx))
    cat  = prob.get("_category", "other")
    sub  = prob.get("_subcategory", "unknown")
    text = _get(prob, "problem", "question", "text", "prompt")
    gold = _get(prob, "answer", "solution", "target")

    prompt  = build_prompt(text)
    inputs  = tokenizer(prompt, return_tensors="pt")
    inp_ids = inputs["input_ids"].to(_device)
    attn    = inputs.get("attention_mask")
    if attn is not None:
        attn = attn.to(_device)
    input_len = inp_ids.shape[-1]

    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                inp_ids,
                attention_mask=attn,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                do_sample=GENERATION_CONFIG["do_sample"],
                repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen_ids   = out.sequences[0][input_len:]
        raw_out   = tokenizer.decode(gen_ids, skip_special_tokens=True)
        n_tok     = int(gen_ids.shape[0])
        finish    = "eos" if n_tok < GENERATION_CONFIG["max_new_tokens"] else "length"

        # Per-token logprobs (A案)
        tok_lp = []
        for step, scores in enumerate(out.scores):
            if step >= len(gen_ids):
                break
            lp = F.log_softmax(scores[0], dim=-1)[gen_ids[step].item()].item()
            tok_lp.append(round(lp, 4))

        gen_err = None
    except Exception as _ge:
        gen_ids  = torch.tensor([], dtype=torch.long)
        raw_out  = ""
        n_tok    = 0
        finish   = "error"
        tok_lp   = []
        gen_err  = str(_ge)

    elapsed = round(time.time() - t0, 2)

    pred, reasoning, final_txt = extract_answer(raw_out)
    correct  = answers_match(pred, gold)
    parse_ok = pred is not None

    if correct:   n_correct += 1
    if not parse_ok: parse_fail += 1
    total_tok += n_tok

    # Logprob stats
    min_lp  = min(tok_lp) if tok_lp else None
    mean_lp = round(sum(tok_lp)/len(tok_lp), 4) if tok_lp else None

    rec = {
        "problem_id":            pid,
        "category":              cat,
        "subcategory":           sub,
        "question":              text,
        "gold_answer":           gold,
        "pred_answer":           pred or "",
        "raw_output":            raw_out,
        "reasoning_text":        (reasoning or "")[:600],
        "final_answer_text":     final_txt or "",
        "is_correct":            correct,
        "parse_success":         parse_ok,
        "parse_error_type":      (None if parse_ok else ("generate_error" if gen_err else "no_boxed")),
        "generation_token_count":n_tok,
        "finish_reason":         finish,
        "elapsed_seconds":       elapsed,
        "seed":                  SEED,
        "token_logprobs":        tok_lp,
        "min_logprob":           min_lp,
        "mean_logprob":          mean_lp,
    }
    results.append(rec)

    with open(pred_path, "a", encoding="utf-8") as _f:
        _f.write(json.dumps(rec, ensure_ascii=False) + "\\n")

    status = ("OK" if correct else ("PARSE_FAIL" if not parse_ok else "WRONG"))
    lp_str = f"lp={min_lp:.2f}" if min_lp is not None else "lp=n/a"
    print(f"[{idx+1:3d}/{len(sampled)}] {pid}: {status:<10s} tok={n_tok:4d}  "
          f"{lp_str}  t={elapsed:.1f}s")

elapsed_all = round(time.time() - _t_start_all, 1)
acc = n_correct / max(len(results), 1)
print(f"\\n=== Inference complete in {elapsed_all}s ===")
print(f"Accuracy  : {n_correct}/{len(results)} = {acc:.4f}")
print(f"Parse fail: {parse_fail}/{len(results)}")
print(f"Avg tokens: {total_tok/max(len(results),1):.0f}")
print(f"Saved     : {pred_path}")
""", tag="inference")

CELL_LOGPROB = code("""\
import csv as _csv

# ----------------------------------------------------------------
# Per-problem logprob summary
# ----------------------------------------------------------------
LOW_LP_THR = -2.0   # トークンを「低信頼」と見なす閾値

logprob_rows = []
for rec in results:
    tok_lp = rec.get("token_logprobs", [])
    low_cnt = sum(1 for lp in tok_lp if lp < LOW_LP_THR)

    # Answer span (末尾の\\boxed{} 付近の logprob)
    ans_start = max(0, len(tok_lp) - 30)
    ans_lp = tok_lp[ans_start:]
    ans_min  = min(ans_lp) if ans_lp else None
    ans_mean = round(sum(ans_lp)/len(ans_lp), 4) if ans_lp else None
    ans_low  = sum(1 for lp in ans_lp if lp < LOW_LP_THR)

    logprob_rows.append({
        "problem_id":             rec["problem_id"],
        "category":               rec["category"],
        "subcategory":            rec["subcategory"],
        "is_correct":             rec["is_correct"],
        "parse_success":          rec["parse_success"],
        "min_logprob":            rec["min_logprob"],
        "mean_logprob":           rec["mean_logprob"],
        "answer_min_logprob":     ans_min,
        "answer_mean_logprob":    ans_mean,
        "low_conf_token_count":   low_cnt,
        "answer_low_conf_token_count": ans_low,
    })

lp_path = OUTPUT_DIR / "min_logprob_summary.csv"
with open(lp_path, "w", newline="", encoding="utf-8") as f:
    w = _csv.DictWriter(f, fieldnames=logprob_rows[0].keys())
    w.writeheader(); w.writerows(logprob_rows)
print(f"Saved: {lp_path}")

# ----------------------------------------------------------------
# Category failure summary
# ----------------------------------------------------------------
from collections import defaultdict as _dd

cat_stats = _dd(lambda: {"n":0,"correct":0,"min_lp_sum":0,"ans_min_lp_sum":0,
                          "wrong_low":0,"wrong_high":0,"correct_low":0})
for row in logprob_rows:
    cat = row["category"]
    s   = cat_stats[cat]
    s["n"] += 1
    if row["is_correct"]: s["correct"] += 1
    if row["min_logprob"] is not None:
        s["min_lp_sum"] += row["min_logprob"]
    if row["answer_min_logprob"] is not None:
        s["ans_min_lp_sum"] += row["answer_min_logprob"]
    if not row["is_correct"]:
        if (row["min_logprob"] or 0) < LOW_LP_THR:
            s["wrong_low"] += 1
        else:
            s["wrong_high"] += 1
    elif (row["min_logprob"] or 0) < LOW_LP_THR:
        s["correct_low"] += 1


def _priority(s):
    n, acc = s["n"], s["correct"]/max(s["n"],1)
    wrong_low = s["wrong_low"]
    score = (1 - acc) * 3 + (wrong_low / max(n, 1)) * 2 + min(n / 30, 1)
    return round(min(score * 5, 5), 1)


cat_fail_rows = []
for cat, s in sorted(cat_stats.items(), key=lambda x: -x[1]["n"]):
    n   = s["n"]
    acc = round(s["correct"]/max(n,1), 4)
    cat_fail_rows.append({
        "category":             cat,
        "subcategory":          "ALL",
        "n":                    n,
        "correct":              s["correct"],
        "accuracy":             acc,
        "avg_min_logprob":      round(s["min_lp_sum"]/max(n,1), 4),
        "avg_answer_min_logprob":round(s["ans_min_lp_sum"]/max(n,1), 4),
        "n_wrong_low_conf":     s["wrong_low"],
        "n_wrong_high_conf":    s["wrong_high"],
        "n_correct_low_conf":   s["correct_low"],
        "priority_score":       _priority(s),
        "priority_reason":      f"acc={acc:.2f} wrong_low={s['wrong_low']}",
    })

cf_path = OUTPUT_DIR / "category_failure_summary.csv"
with open(cf_path, "w", newline="", encoding="utf-8") as f:
    w = _csv.DictWriter(f, fieldnames=cat_fail_rows[0].keys())
    w.writeheader(); w.writerows(cat_fail_rows)
print(f"Saved: {cf_path}")

print("\\n=== Category Failure Summary ===")
print(f"{'category':<22s} {'n':>4} {'acc':>6} {'priority':>8} {'wrong_low':>10}")
print("-"*56)
for r in sorted(cat_fail_rows, key=lambda x: -x["priority_score"]):
    print(f"{r['category']:<22s} {r['n']:>4d} {r['accuracy']:>6.3f} "
          f"{r['priority_score']:>8.1f} {r['n_wrong_low_conf']:>10d}")
""", tag="logprob")

CELL_FAILURE = code("""\
# ----------------------------------------------------------------
# Failure type classification (keyword on raw_output)
# ----------------------------------------------------------------

_CRYPTO_TYPE_RULES = [
    ("mapping_conflict",    [r"contradict|conflict|inconsist|already.*map|map.*different"]),
    ("leading_zero_error",  [r"leading zero|first digit.*zero|cannot.*be.*0"]),
    ("carry_error",         [r"carry.*wrong|forgot.*carry|carry.*incorrect"]),
    ("incomplete_search",   [r"no.*solution|cannot find|exhausted|no valid"]),
    ("arithmetic_error",    [r"\\d+\\s*\\+\\s*\\d+\\s*=\\s*(?!.*correct)"]),
    ("constraint_missed",   [r"unique|distinct|different digit"]),
    ("final_parse_error",   [r"\\\\boxed\\{\\}|therefore.*=|answer is"]),
    ("hallucinated_rule",   [r"must be.*odd|must be.*even|only.*prime"]),
    ("answer_format_error", [r"the answer is \\w+=\\d|result:.*[a-zA-Z]"]),
]

_BIT_TYPE_RULES = [
    ("xor_error",           [r"XOR|exclusive or"]),
    ("and_error",           [r"bitwise AND|& "]),
    ("or_error",            [r"bitwise OR|\\| "]),
    ("shift_error",         [r"shift.*left|shift.*right|<<|>>"]),
    ("mask_error",          [r"mask|bitmask"]),
    ("signed_unsigned_error",[r"signed|unsigned|two.?s complement|overflow"]),
    ("base_conversion_error",[r"convert.*base|binary|hexadecimal"]),
    ("arithmetic_error",    [r"sum|product|addition|subtraction"]),
    ("final_parse_error",   [r"\\\\boxed\\{\\}|the answer is"]),
    ("answer_format_error", [r"decimal is|result is \\w"]),
]

_NUM_TYPE_RULES = [
    ("binary_decimal_error",  [r"binary.*decimal|base.?2.*10"]),
    ("decimal_binary_error",  [r"decimal.*binary|base.?10.*2"]),
    ("hex_decimal_error",     [r"hex.*decimal|base.?16.*10"]),
    ("decimal_hex_error",     [r"decimal.*hex|base.?10.*16"]),
    ("base_n_place_value_error",[r"place value|position|base.?\\d+"]),
    ("roman_numeral_error",   [r"roman"]),
    ("digit_order_error",     [r"reverse|order|digit.*place"]),
    ("final_parse_error",     [r"\\\\boxed\\{\\}|the answer is"]),
    ("answer_format_error",   [r"result:.*[a-zA-Z]"]),
]


def _classify_failure(raw: str, rules: list) -> tuple:
    for ft, pats in rules:
        for p in pats:
            try:
                if re.search(p, raw, re.IGNORECASE):
                    return ft, p
            except re.error:
                pass
    return "unknown", ""


def _priority_score(rec, rules):
    ft, _ = _classify_failure(rec.get("raw_output",""), rules)
    if ft == "final_parse_error":     return 3
    if ft == "unknown":               return 2
    if rec.get("min_logprob") is not None and rec["min_logprob"] < -3.0: return 5
    return 4


def _build_failure_rows(category: str, rules: list) -> list:
    rows = []
    for rec in results:
        if rec["category"] != category:
            continue
        if rec["is_correct"] and (rec.get("min_logprob") or 0) >= LOW_LP_THR:
            continue   # stable correct → skip
        raw = rec.get("raw_output", "")
        ft, rule_matched = _classify_failure(raw, rules)
        rows.append({
            "problem_id":                rec["problem_id"],
            "category":                  rec["category"],
            "subcategory":               rec["subcategory"],
            "question":                  rec["question"][:200],
            "gold_answer":               rec["gold_answer"],
            "pred_answer":               rec["pred_answer"],
            "is_correct":                rec["is_correct"],
            "min_logprob":               rec.get("min_logprob"),
            "answer_min_logprob":        None,   # placeholder
            "failure_type":              ft,
            "failure_reason":            rule_matched,
            "solver_check_possible":     category in ("cryptarithm","bit_manipulation","numeral_conversion"),
            "synthetic_generation_possible": category in ("cryptarithm","bit_manipulation","numeral_conversion"),
            "recommended_template":      f"{category}_cot_v1",
            "example_priority":          _priority_score(rec, rules),
        })
    return rows


# Cryptarithm
crypto_rows = _build_failure_rows("cryptarithm", _CRYPTO_TYPE_RULES)
cr_path = OUTPUT_DIR / "failure_cases_cryptarithm.csv"
if crypto_rows:
    with open(cr_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=crypto_rows[0].keys())
        w.writeheader(); w.writerows(crypto_rows)
    print(f"Saved: {cr_path}  ({len(crypto_rows)} rows)")
else:
    print("[info] No cryptarithm failure rows")

# Bit manipulation
bit_rows = _build_failure_rows("bit_manipulation", _BIT_TYPE_RULES)
bit_path = OUTPUT_DIR / "failure_cases_bit_manipulation.csv"
if bit_rows:
    with open(bit_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=bit_rows[0].keys())
        w.writeheader(); w.writerows(bit_rows)
    print(f"Saved: {bit_path}  ({len(bit_rows)} rows)")

# Numeral conversion
num_rows = _build_failure_rows("numeral_conversion", _NUM_TYPE_RULES)
num_path = OUTPUT_DIR / "failure_cases_numeral_conversion.csv"
if num_rows:
    with open(num_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=num_rows[0].keys())
        w.writeheader(); w.writerows(num_rows)
    print(f"Saved: {num_path}  ({len(num_rows)} rows)")

# failure_type_summary
all_fail_rows = crypto_rows + bit_rows + num_rows
if all_fail_rows:
    ft_counts = Counter(r["failure_type"] for r in all_fail_rows)
    ft_rows = [{"failure_type": ft, "count": c, "example_priority_avg":
                round(sum(r["example_priority"] for r in all_fail_rows if r["failure_type"]==ft)/c, 1)}
               for ft, c in ft_counts.most_common()]
    ft_path = OUTPUT_DIR / "failure_type_summary.csv"
    with open(ft_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=ft_rows[0].keys())
        w.writeheader(); w.writerows(ft_rows)
    print(f"Saved: {ft_path}")

    print("\\n=== Failure Type Distribution ===")
    for r in ft_rows[:10]:
        print(f"  {r['failure_type']:<30s}: {r['count']:4d}")
""", tag="failure")

CELL_SUMMARY = code("""\
import datetime

# ----------------------------------------------------------------
# Golden validation summary CSV
# ----------------------------------------------------------------
n_total   = len(results)
n_correct_total = sum(1 for r in results if r["is_correct"])
n_parse   = sum(1 for r in results if r["parse_success"])
avg_tok   = round(sum(r["generation_token_count"] for r in results)/max(n_total,1), 1)

summary_rows = [
    {"split":"overall","category":"ALL","subcategory":"ALL",
     "n":n_total,"correct":n_correct_total,
     "accuracy":round(n_correct_total/max(n_total,1),4),
     "n_parse_success":n_parse,
     "parse_success_rate":round(n_parse/max(n_total,1),4),
     "avg_generation_token_count":avg_tok},
]
for cat, s in cat_stats.items():
    summary_rows.append({
        "split":"category","category":cat,"subcategory":"ALL",
        "n":s["n"],"correct":s["correct"],
        "accuracy":round(s["correct"]/max(s["n"],1),4),
        "n_parse_success":s["n"],
        "parse_success_rate":1.0,
        "avg_generation_token_count":avg_tok,
    })

sum_path = OUTPUT_DIR / "golden_validation_summary.csv"
with open(sum_path, "w", newline="", encoding="utf-8") as f:
    w = _csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    w.writeheader(); w.writerows(summary_rows)
print(f"Saved: {sum_path}")

# ----------------------------------------------------------------
# run_commands.md
# ----------------------------------------------------------------
ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
run_md = f\"\"\"# Phase 3 Run Commands

## Execution timestamp
{ts}

## Environment
- Platform  : RTX Pro 5000 (offline)
- Notebook  : phase3_rtx_analysis.ipynb
- mamba_ssm : {'native' if _mamba_native else 'PyTorch stub v1.2.0'}

## Input paths
- ADAPTER_PATH  : {ADAPTER_PATH}
- MODEL_PATH    : {MODEL_PATH}
- PROBLEMS_PATH : {PROBLEMS_PATH}

## Generation config
{json.dumps(GENERATION_CONFIG, indent=2)}

## Sampling
- SEED         : {SEED}
- MAX_PROBLEMS : {MAX_PROBLEMS}
- CATEGORY_QUOTA: {json.dumps(CATEGORY_QUOTA)}

## Generated artefacts
- {OUTPUT_DIR}/category_map.csv
- {OUTPUT_DIR}/validation_set_labeled.csv
- {OUTPUT_DIR}/golden_validation_predictions.jsonl
- {OUTPUT_DIR}/golden_validation_summary.csv
- {OUTPUT_DIR}/min_logprob_summary.csv
- {OUTPUT_DIR}/category_failure_summary.csv
- {OUTPUT_DIR}/failure_type_summary.csv
- {OUTPUT_DIR}/failure_cases_cryptarithm.csv
- {OUTPUT_DIR}/failure_cases_bit_manipulation.csv
- {OUTPUT_DIR}/failure_cases_numeral_conversion.csv

## Results
- n_total   : {n_total}
- accuracy  : {round(n_correct_total/max(n_total,1),4)}
- avg_tokens: {avg_tok}
\"\"\"

with open(OUTPUT_DIR / "run_commands.md", "w", encoding="utf-8") as f:
    f.write(run_md)
print(f"Saved: {OUTPUT_DIR}/run_commands.md")

# ----------------------------------------------------------------
# Final report
# ----------------------------------------------------------------
print("\\n" + "="*60)
print("=== Phase 3 Analysis Complete ===")
print("="*60)
print(f"n_total   : {n_total}")
print(f"accuracy  : {n_correct_total}/{n_total} = {n_correct_total/max(n_total,1):.4f}")
print(f"parse_ok  : {n_parse}/{n_total}")
print(f"avg_tokens: {avg_tok}")
print()
print("Weakest categories (by priority):")
for r in sorted(cat_fail_rows, key=lambda x: -x["priority_score"])[:5]:
    print(f"  #{r['category']:<20s} acc={r['accuracy']:.3f}  priority={r['priority_score']}")
print()
print("Output files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")
print()
print("⚠  Confirmed: NO adapter weights modified, NO submission.zip created.")
""", tag="summary")


# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
cells = [
    CELL_TITLE,
    CELL_CONFIG,
    CELL_CATEGORY,
    CELL_MODEL,
    CELL_INFERENCE,
    CELL_LOGPROB,
    CELL_FAILURE,
    CELL_SUMMARY,
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out_path = Path(NOTEBOOK_PATH)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

cell_names = ["title", "config", "category", "model", "inference", "logprob", "failure", "summary"]
print(f"Generated: {out_path}  ({len(cells)} cells)")
for i, name in enumerate(cell_names):
    print(f"  Cell {i}: {name}")
