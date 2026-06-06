#!/usr/bin/env python3
"""Generate phase3_rtx_analysis.ipynb — self-contained offline Phase 3 analysis notebook."""
import json

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(src):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    })

# ---------------------------------------------------------------------------
# Cell 0: Title
# ---------------------------------------------------------------------------
md("""# Phase 3: Golden Baseline 失敗分析 — 自己完結オフライン版

**安全保証:** adapter / training / submission には一切触れません。

## 実行環境
- Kaggle Notebook (Internet OFF) または RTX Pro 5000
- GPU: T4 x2 / P100 / RTX Pro 5000 いずれも対応

## 出力ファイル (OUTPUT_DIR 以下)
- `golden_validation_predictions.jsonl` — 推論ログ
- `golden_validation_summary.csv` — カテゴリ別精度
- `step2_diagnostics.txt` — 診断ログ
""")

# ---------------------------------------------------------------------------
# Cell 1: Configuration
# ---------------------------------------------------------------------------
code(r"""# ============================================================
# CONFIGURATION
# ============================================================
import os
from pathlib import Path

ADAPTER_PATH  = os.environ.get("ADAPTER_PATH",
    "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20")
MODEL_PATH    = os.environ.get("MODEL_PATH",
    "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1")
PROBLEMS_PATH = os.environ.get("PROBLEMS_PATH",
    "/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", "/kaggle/working/phase3_analysis")
SEED          = 42
MAX_PROBLEMS  = 200   # 0 = all; 200 for stratified sample

# Stratified sampling quota (used when MAX_PROBLEMS > 0)
CATEGORY_QUOTA = {
    "cryptarithm":       50,
    "bit_manipulation":  40,
    "numeral_conversion": 40,
    "cipher":            20,
    "equation":          20,
    "arithmetic":        15,
    "other":             15,
}

# Generation config -- identical to Golden Baseline (do not modify)
GOLDEN_GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "temperature":    0.0,
    "do_sample":      False,
    "repetition_penalty": 1.0,
    "stop": ["<|endoftext|>", "<|im_end|>"],
}

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"Output dir : {OUTPUT_DIR}")
print(f"Adapter    : {ADAPTER_PATH}")
print(f"Model      : {MODEL_PATH}")
print(f"Problems   : {PROBLEMS_PATH}")
print(f"Limit      : {MAX_PROBLEMS if MAX_PROBLEMS else 'all'}")
""")

# ---------------------------------------------------------------------------
# Cell 2: mamba_ssm Option-B stub + causal_conv1d stub
# NOTE: triple-double-quotes MUST NOT appear inside this code(r"...") block.
#       All docstrings use single-line comments.
# ---------------------------------------------------------------------------
code(r"""# ============================================================
# mamba_ssm Option-B stub
# ============================================================
# The Kaggle notebook environment ships mamba_ssm compiled against a different
# CUDA ABI (selective_scan_cuda.so -> "undefined symbol").  We replace it with
# pure-PyTorch implementations that satisfy the model import chain AND enable
# the mamba2 fast-path by reporting version "2.0.4":
#   is_mamba_2_ssm_available() returns True when version >= 2.0.4
#
# Fast-path functions provided (pure PyTorch, no Triton/CUDA needed):
#   mamba_ssm.ops.triton.selective_state_update.selective_state_update
#   mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined
#   mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn
#   causal_conv1d.causal_conv1d_fn / causal_conv1d_update
#
# Generation speed: selective_state_update is O(1) per step so generation
# no longer degrades to the slow O(N) recurrence of the mamba2-disabled path.
# ============================================================

import sys as _sys, types as _types, importlib.util as _iutil

for _k in [k for k in list(_sys.modules) if k == "mamba_ssm" or k.startswith("mamba_ssm.")]:
    del _sys.modules[_k]
for _k in [k for k in list(_sys.modules) if k == "causal_conv1d" or k.startswith("causal_conv1d.")]:
    del _sys.modules[_k]

def _mkmod(name):
    m = _types.ModuleType(name)
    m.__path__ = []
    m.__package__ = name.rsplit(".", 1)[0] if "." in name else name
    m.__spec__ = _iutil.spec_from_loader(name, loader=None)
    return m

# ------------------------------------------------------------------
# rmsnorm_fn  (unconditional import in modeling_nemotron_h.py)
# ------------------------------------------------------------------
def _rmsnorm_fn(x, weight, bias=None, residual=None, x_bias=None,
                z=None, z_bias=None, prenorm=False, residual_in_fp32=False,
                eps=1e-6, is_rms_norm=True, return_dropout_mask=False,
                norm_before_gate=True, group_size=None, **_kw):
    import torch as _t, torch.nn.functional as _F
    dtype = x.dtype
    xf = x.float()
    if x_bias is not None: xf = xf + x_bias.float()
    if residual is not None: xf = xf + residual.float()
    gate = None
    if z is not None:
        zf = z.float()
        if z_bias is not None: zf = zf + z_bias.float()
        gate = _F.silu(zf)
        if not norm_before_gate: xf = xf * gate
    if group_size is not None and group_size > 1:
        sh = xf.shape
        xg = xf.reshape(*sh[:-1], -1, group_size)
        xn = (xg * _t.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)).reshape(sh)
    else:
        xn = xf * _t.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = weight.float() * xn
    if bias is not None: out = out + bias.float()
    if gate is not None and norm_before_gate: out = out * gate
    out = out.to(dtype)
    if prenorm:
        res_out = xf if residual_in_fp32 else xf.to(dtype)
        return (out, res_out) if not return_dropout_mask else (out, res_out, None)
    return out if not return_dropout_mask else (out, None)

# ------------------------------------------------------------------
# selective_state_update  (Mamba2 single-step SSM update)
# Pure-PyTorch, no Triton needed. O(1) per generation step.
# Mamba2: state(batch,nheads,headdim,dstate) updated in-place
#   x(batch,nheads,headdim)  dt(batch,nheads)  A(nheads,)
#   B/C(batch,ngroups,dstate)  D(nheads,)|(nheads,headdim)|None
#   z(batch,nheads,headdim)|None -> returns y(batch,nheads,headdim)
# ------------------------------------------------------------------
def _selective_state_update(state, x, dt, A, B, C, D=None, z=None,
                              dt_bias=None, dt_softplus=False, ngroups=1):
    import torch as _t, torch.nn.functional as _F
    dtype = state.dtype
    if dt_bias is not None: dt = dt + dt_bias.to(dt.dtype)
    if dt_softplus: dt = _F.softplus(dt)
    if state.dim() == 3:
        # Mamba1: state(batch, d_model, d_state)
        dA = _t.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(1)
        ns = (dA * state + dB * x.unsqueeze(-1)).to(dtype)
        state.copy_(ns)
        y = (ns * C.unsqueeze(1)).sum(-1)
        if D is not None: y = y + D * x
        if z is not None: y = y * _F.silu(z)
        return y
    # Mamba2: state(batch, nheads, headdim, dstate)
    dA = _t.exp(dt * A)                                # (batch, nheads)
    hpg = max(state.shape[1] // max(ngroups, 1), 1)
    B_e = B.repeat_interleave(hpg, dim=1) if B.shape[1] < state.shape[1] else B
    C_e = C.repeat_interleave(hpg, dim=1) if C.shape[1] < state.shape[1] else C
    dB  = dt.unsqueeze(-1) * B_e                       # (batch, nheads, dstate)
    ns  = (dA[:, :, None, None] * state
           + dB[:, :, None, :] * x[:, :, :, None]).to(dtype)
    state.copy_(ns)
    y = (ns * C_e[:, :, None, :]).sum(-1)              # (batch, nheads, headdim)
    if D is not None:
        y = y + (D[:, None] * x if D.dim() == 1 else D * x)
    if z is not None: y = y * _F.silu(z)
    return y

# ------------------------------------------------------------------
# mamba_chunk_scan_combined  (sequential PyTorch prefill scan)
# x(batch,seqlen,nheads,headdim)  dt(batch,seqlen,nheads)
# A(nheads,)  B/C(batch,seqlen,ngroups,dstate)
# ------------------------------------------------------------------
def _mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size,
                                D=None, z=None, dt_bias=None,
                                initial_states=None, seq_idx=None,
                                cu_seqlens=None, dt_softplus=False,
                                dt_limit=(0.0, float("inf")),
                                return_final_states=False,
                                final_states_out=None,
                                out_dtype=None, states_in_fp32=True):
    import torch as _t, torch.nn.functional as _F
    batch, seqlen, nheads, headdim = x.shape
    ngroups = B.shape[-2]
    dt_eff  = dt.clone()
    if dt_bias is not None: dt_eff = dt_eff + dt_bias
    if dt_softplus: dt_eff = _F.softplus(dt_eff)
    dt_eff = dt_eff.clamp(dt_limit[0], dt_limit[1])
    sdtype = _t.float32 if states_in_fp32 else x.dtype
    state  = (initial_states.clone().to(sdtype) if initial_states is not None
              else _t.zeros(batch, nheads, headdim, B.shape[-1],
                            device=x.device, dtype=sdtype))
    ys = []
    for t in range(seqlen):
        yt = _selective_state_update(
            state, x[:, t], dt_eff[:, t], A, B[:, t], C[:, t],
            D=D, z=z[:, t] if z is not None else None,
            dt_softplus=False, ngroups=ngroups,
        )
        ys.append(yt)
    y = _t.stack(ys, dim=1)
    if out_dtype is not None: y = y.to(out_dtype)
    if return_final_states:
        return y, state.to(x.dtype if not states_in_fp32 else _t.float32)
    return y

def _mamba_split_conv1d_scan_combined(*args, **kwargs):
    raise NotImplementedError(
        "mamba_split_conv1d_scan_combined not implemented in PyTorch stub. "
        "Unexpected call -- check model config."
    )

# ------------------------------------------------------------------
# causal_conv1d_fn / causal_conv1d_update
# ------------------------------------------------------------------
def _causal_conv1d_fn(x, weight, bias=None, initial_states=None,
                       final_states_out=None, activation=None):
    # Causal depthwise conv1d. x(batch,dim,seqlen) weight(dim,1,kernel_size)
    import torch.nn.functional as _F
    k = weight.shape[-1]
    out = _F.conv1d(_F.pad(x, (k - 1, 0)), weight, bias=bias, groups=x.shape[1])
    if activation in ("silu", "swish"): out = _F.silu(out)
    return out

def _causal_conv1d_update(x, conv_state, weight, bias=None,
                           activation=None, cache_seqlens=None):
    # Single-step conv. x(batch,dim) conv_state(batch,dim,kernel_size) in-place
    import torch as _t, torch.nn.functional as _F
    ns = _t.roll(conv_state, -1, dims=-1)
    ns[:, :, -1] = x
    conv_state.copy_(ns)
    out = (ns * weight[:, 0, :]).sum(-1)
    if bias is not None: out = out + bias
    if activation in ("silu", "swish"): out = _F.silu(out)
    return out

def _stub_fn(*a, **k):
    raise NotImplementedError("mamba_ssm stub: CUDA SSM kernels disabled (ABI mismatch)")

# ------------------------------------------------------------------
# Register module hierarchy
# ------------------------------------------------------------------
_mamba    = _mkmod("mamba_ssm")
_mamba.__version__ = "2.0.4"   # >= 2.0.4 -> is_mamba_2_ssm_available() = True

_ops      = _mkmod("mamba_ssm.ops")
_triton   = _mkmod("mamba_ssm.ops.triton")

_ln_gated = _mkmod("mamba_ssm.ops.triton.layernorm_gated")
_ln_gated.rmsnorm_fn = _rmsnorm_fn

_ssu      = _mkmod("mamba_ssm.ops.triton.selective_state_update")
_ssu.selective_state_update     = _selective_state_update
_ssu.selective_state_update_ref = _selective_state_update

_ssd      = _mkmod("mamba_ssm.ops.triton.ssd_combined")
_ssd.mamba_chunk_scan_combined        = _mamba_chunk_scan_combined
_ssd.mamba_split_conv1d_scan_combined = _mamba_split_conv1d_scan_combined

_sel_scan = _mkmod("mamba_ssm.ops.selective_scan_interface")
_sel_scan.selective_scan_fn = _stub_fn
_sel_scan.mamba_inner_fn    = _stub_fn

_mamba.ops = _ops
_ops.triton = _triton
_ops.selective_scan_interface = _sel_scan
_triton.layernorm_gated = _ln_gated
_triton.ssd_combined = _ssd

for _name, _mod in [
    ("mamba_ssm",                                   _mamba),
    ("mamba_ssm.ops",                               _ops),
    ("mamba_ssm.ops.triton",                        _triton),
    ("mamba_ssm.ops.triton.layernorm_gated",        _ln_gated),
    ("mamba_ssm.ops.triton.selective_state_update", _ssu),
    ("mamba_ssm.ops.triton.ssd_combined",           _ssd),
    ("mamba_ssm.ops.selective_scan_interface",      _sel_scan),
]:
    _sys.modules[_name] = _mod

_cc1d = _mkmod("causal_conv1d")
_cc1d.causal_conv1d_fn     = _causal_conv1d_fn
_cc1d.causal_conv1d_update = _causal_conv1d_update
_sys.modules["causal_conv1d"] = _cc1d

print("[mamba_stub] Injected mamba_ssm v2.0.4 (Option-B: PyTorch SSM fast-path)")
print("[mamba_stub] is_mamba_2_ssm_available() will return True")

try:
    import mamba_ssm as _ms
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update as _f1
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as _f2
    from causal_conv1d import causal_conv1d_fn as _f3
    print(f"[mamba_stub] Import check OK  version={_ms.__version__}")
except Exception as _e:
    print(f"[mamba_stub] Import check FAILED: {_e}")
""")

# ---------------------------------------------------------------------------
# Cell 3: Utilities
# ---------------------------------------------------------------------------
code(r"""import csv, gc, json, os, re, time, datetime, types as _types
from collections import defaultdict
from contextlib import nullcontext as _nullctx
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
for _pkg in ["transformers", "peft", "bitsandbytes"]:
    try:
        import importlib as _il
        _m = _il.import_module(_pkg)
        print(f"  {_pkg}: OK  v{getattr(_m, '__version__', '?')}")
    except ImportError:
        print(f"  {_pkg}: NOT FOUND")

BOXED_RE     = re.compile(r"\\boxed\{([^{}]+)\}", re.DOTALL)
THEREFORE_RE = re.compile(
    r"(?:the (?:final )?answer is|therefore|so the answer is|answer:)"
    r"\s*(?:\\boxed\{)?([^\n\.\\{}]+)",
    re.IGNORECASE,
)
LAST_LINE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _\-]*$")

def extract_answer(raw):
    m = BOXED_RE.search(raw)
    if m:
        return m.group(1).strip(), raw[:m.start()].strip(), m.group(0)
    m = THEREFORE_RE.search(raw)
    if m:
        a = re.sub(r"[^A-Za-z0-9]", "", m.group(1)).strip()
        if a:
            return a, raw[:m.start()].strip(), m.group(0)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if lines:
        m2 = LAST_LINE_RE.search(lines[-1])
        if m2:
            return m2.group(0).strip(), "\n".join(lines[:-1]), lines[-1]
    return None, raw.strip(), None

def normalize_answer(a):
    return re.sub(r"\s+", "", str(a or "")).strip().upper()

def answers_match(pred, gold):
    return bool(pred) and normalize_answer(pred) == normalize_answer(gold)

def parse_error_type(pred, raw):
    if pred is not None: return None
    if BOXED_RE.search(raw): return "boxed_found_but_empty"
    if len(raw.strip()) < 10: return "empty_output"
    return "no_extractable_answer"

def _load_csv_problems(path):
    QCOLS = ("question", "problem", "prompt", "input", "text")
    ACOLS = ("answer", "solution", "target", "output", "label")
    ICOLS = ("id", "problem_id", "uid", "sample_id", "index")
    records = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames: return records
        fl = {f.lower(): f for f in reader.fieldnames}
        qc = next((fl[c] for c in QCOLS if c in fl), None)
        ac = next((fl[c] for c in ACOLS if c in fl), None)
        ic = next((fl[c] for c in ICOLS if c in fl), None)
        for idx, row in enumerate(reader):
            rec = dict(row)
            if qc and qc != "question": rec["question"] = row[qc]
            if ac and ac != "answer":   rec["answer"]   = row[ac]
            if ic and ic != "problem_id": rec["problem_id"] = row[ic]
            elif "problem_id" not in rec: rec["problem_id"] = f"row_{idx}"
            records.append(rec)
    return records

def load_problems(path):
    p = Path(path)
    if not p.exists(): return []
    if p.suffix.lower() == ".csv": return _load_csv_problems(p)
    records = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line: records.append(json.loads(line))
    return records

def get_problem_id(record, idx):
    for key in ("problem_id", "id", "uid", "sample_id"):
        v = record.get(key)
        if v: return str(v)
    return f"row_{idx}"

def build_prompt(record):
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
                    parts.append(
                        f"  {i}. Input: {ex.get('input','')} -> Output: {ex.get('output','')}")
                else:
                    parts.append(f"  {i}. {ex}")
        parts.append("")
    parts.append(f"Question: {question}")
    parts.append("\nThink step by step. Put your final answer inside \\boxed{}.")
    return "\n".join(parts)

print("Utilities loaded")
""")

# ---------------------------------------------------------------------------
# Cell 4: Category classifier + stratified sampling
# ---------------------------------------------------------------------------
code(r"""_CATEGORY_PATTERNS = [
    ("cryptarithm",        re.compile(
        r"[A-Z]{2,}\s*\+\s*[A-Z]{2,}|each letter.*digit|letter.*unique|"
        r"cryptarithm|cryptoarithm|send\s*\+\s*more|verbal arithmetic",
        re.I)),
    ("bit_manipulation",   re.compile(
        r"\bXOR\b|\bAND\b|\bOR\b|\bNOT\b|bitwise|bit.*shift|binary.*operat",
        re.I)),
    ("numeral_conversion", re.compile(
        r"convert.*base|base\s*\d+|hexadecim|octal|binary.*number|"
        r"number.*system|roman numeral|\bbase-\d",
        re.I)),
    ("cipher",             re.compile(
        r"caesar|vigenere|substitut.*cipher|encrypt|decrypt|cipher|"
        r"encoded.*message|shift.*letter",
        re.I)),
    ("equation",           re.compile(
        r"solve.*equation|find.*value.*[xy]|system of equation|"
        r"linear equation|quadratic",
        re.I)),
    ("arithmetic",         re.compile(
        r"sum of|product of|average|mean of|\bdivide\b|\bmultiply\b|"
        r"remainder|modulo|\bmod\b|\bfactorial\b",
        re.I)),
]

def classify_problem(record):
    text = " ".join([
        str(record.get("question", "")),
        str(record.get("problem", "")),
        str(record.get("input", "")),
    ])
    for cat, pat in _CATEGORY_PATTERNS:
        if pat.search(text): return cat
    return "other"

def stratified_sample(problems, quota, seed=42):
    # Sample up to sum(quota.values()) problems with per-category limits.
    import random
    rng = random.Random(seed)
    by_cat: Dict[str, List] = defaultdict(list)
    for rec in problems:
        cat = classify_problem(rec)
        rec["_category"] = cat
        by_cat[cat].append(rec)
    selected = []
    for cat, limit in quota.items():
        pool = by_cat.get(cat, [])
        rng.shuffle(pool)
        selected.extend(pool[:limit])
    total = sum(quota.values())
    if len(selected) < total:
        used_ids = {get_problem_id(r, i) for i, r in enumerate(selected)}
        for rec in problems:
            if get_problem_id(rec, 0) not in used_ids:
                selected.append(rec)
                if len(selected) >= total: break
    rng.shuffle(selected)
    return selected

print("Category classifier loaded")
""")

# ---------------------------------------------------------------------------
# Cell 5: Model loading
# ---------------------------------------------------------------------------
code(r"""import os, gc, torch, datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel

_ea = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _ea:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        f"{_ea},expandable_segments:True" if _ea else "expandable_segments:True")

set_seed(SEED); torch.manual_seed(SEED)

_diag_path = Path(OUTPUT_DIR) / "step2_diagnostics.txt"
_sep = f"\n{'='*60}\nRun {datetime.datetime.utcnow().isoformat()}Z\n{'='*60}\n"
with _diag_path.open("a") as _fh: _fh.write(_sep)

def _diag(msg):
    print(msg, flush=True)
    with _diag_path.open("a") as _fh: _fh.write(msg + "\n")

_diag(f"PyTorch {torch.__version__}  CUDA {torch.version.cuda}")
for _gi in range(torch.cuda.device_count()):
    _pr = torch.cuda.get_device_properties(_gi)
    _fr, _tot = torch.cuda.mem_get_info(_gi)
    _diag(f"GPU {_gi}: {_pr.name}  SM{_pr.major}.{_pr.minor}  "
          f"{_tot/1024**3:.1f}GB total  {_fr/1024**3:.1f}GB free")

_cap0 = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
has_bf16 = _cap0[0] >= 8
compute_dtype = torch.bfloat16 if has_bf16 else torch.float16
_diag(f"SM cap: {_cap0}  bf16: {has_bf16}  compute_dtype={compute_dtype}")

try:
    from transformers.utils import is_mamba_2_ssm_available, is_causal_conv1d_available
    _diag(f"is_mamba_2_ssm_available()  = {is_mamba_2_ssm_available()}  (expected True)")
    _diag(f"is_causal_conv1d_available()= {is_causal_conv1d_available()}  (expected True)")
except Exception as _e:
    _diag(f"[warn] availability check: {_e}")

_diag(f"Loading tokenizer from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
_diag(f"Tokenizer OK  eos_token_id={tokenizer.eos_token_id}")

gc.collect(); torch.cuda.empty_cache()

_n_gpus = torch.cuda.device_count()
_total_vram = sum(
    torch.cuda.get_device_properties(i).total_memory for i in range(_n_gpus)
) / 1024**3
_diag(f"Total VRAM: {_total_vram:.1f}GB across {_n_gpus} GPU(s)")
_use_4bit = _total_vram < 50.0
_gpu_budget = {
    i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.85 / 1024**3)}GiB"
    for i in range(_n_gpus)
}

_bnb_config = None
if _use_4bit:
    from transformers import BitsAndBytesConfig as _BnBCfg
    _bnb_config = _BnBCfg(
        load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    _diag(f"Quantization: 4-bit NF4  [VRAM {_total_vram:.1f}GB < 50GB]")
else:
    _diag(f"Quantization: none  [VRAM {_total_vram:.1f}GB >= 50GB]")

_diag("Loading base model ...")
_load_kw = dict(torch_dtype=compute_dtype, device_map="auto",
                max_memory=_gpu_budget, low_cpu_mem_usage=True, trust_remote_code=True)
if _bnb_config is not None: _load_kw["quantization_config"] = _bnb_config
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **_load_kw)
_diag("Base model OK")

try:
    from peft import peft_model as _pm
    _orig = _pm.PeftModel._update_offload
    def _safe_upd(self, oi, aw):
        if not oi: return oi
        try: return _orig(self, oi, aw)
        except KeyError as _ke:
            print(f"[peft_patch] _update_offload KeyError skipped: {_ke}"); return oi
    _pm.PeftModel._update_offload = _safe_upd
    print("[peft_patch] OK")
except Exception as _pe:
    print(f"[peft_patch] {_pe}")

model.eval()
_diag(f"Loading adapter from {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
_diag("Adapter loaded OK")

if not has_bf16:
    _nc = 0
    for _n, _m in model.named_modules():
        if isinstance(_m, torch.nn.Conv1d):
            _m.weight.data = _m.weight.data.to(torch.float16)
            if _m.bias is not None: _m.bias.data = _m.bias.data.to(torch.float16)
            def _ph(mod, inp):
                return tuple(t.to(torch.float16)
                             if isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16
                             else t for t in inp)
            def _poh(mod, inp, out):
                return out.to(torch.bfloat16) if isinstance(out, torch.Tensor) else out
            _m.register_forward_pre_hook(_ph); _m.register_forward_hook(_poh)
            _nc += 1
    if _nc: _diag(f"[fp16-cast] Patched {_nc} Conv1d (SM {_cap0} < 8.0)")

gc.collect(); torch.cuda.empty_cache()
for _gi in range(_n_gpus):
    _fr, _tot = torch.cuda.mem_get_info(_gi)
    _diag(f"GPU {_gi} after PEFT+GC: {_fr/1024**3:.1f}GB free / {_tot/1024**3:.1f}GB")
_diag("Model ready"); print("Model ready")
""")

# ---------------------------------------------------------------------------
# Cell 6: Warmup + cache discovery + EOS setup
# ---------------------------------------------------------------------------
code(r"""import inspect as _inspect, sys as _sys

def _infer_input_device(m):
    for p in m.parameters():
        if p.device.type == "cuda": return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_device = _infer_input_device(model)
_diag(f"Input device: {input_device}")

_diag("[warmup] Running 1-token warmup ...")
try:
    _wu_ids = tokenizer(
        tokenizer.decode([tokenizer.eos_token_id or 1] * 4),
        return_tensors="pt"
    ).input_ids.to(input_device)
    with torch.no_grad():
        _ = model.generate(_wu_ids, max_new_tokens=1, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
    del _, _wu_ids; gc.collect(); torch.cuda.empty_cache()
    _diag("[warmup] Done")
except Exception as _wu_e:
    _diag(f"[warmup] Error (non-fatal): {_wu_e}")

_nemh_cache_cls = None
_inner_model = getattr(getattr(model, "base_model", model), "model", model)
for _mn, _mod in list(_sys.modules.items()):
    if _mod is None: continue
    for _cn in ("HybridMambaAttentionDynamicCache", "NemotronHHybridDynamicCache"):
        try:
            _cls = getattr(_mod, _cn, None)
        except Exception:
            continue
        if isinstance(_cls, type):
            _nemh_cache_cls = _cls
            _diag(f"[cache] Found {_cn} in {_mn}")
            break
    if _nemh_cache_cls is not None: break
if _nemh_cache_cls is None:
    _diag("[cache] Cache class not found -- OOM risk on long generation")

_model_cfg_for_cache = (
    getattr(model, "config", None)
    or getattr(getattr(model, "base_model", None), "config", None)
)
try:
    if _model_cfg_for_cache is None:
        _model_cfg_for_cache = model.base_model.model.config
except Exception:
    pass

# -----------------------------------------------------------------------
# EOS token list: use added_tokens_encoder to avoid the tok=1 bug.
#
# tokenizer.encode("<|im_end|>", add_special_tokens=False) can return
# MULTIPLE char-level tokens when the tokenizer treats angle-brackets as
# regular BPE characters (e.g. [28766, 321, 28730, 416, 28766]).
# Any of those common sub-tokens in eos_token_id causes generation to
# stop after the very first output token.
#
# added_tokens_encoder is a dict {str -> int} of specially registered
# tokens; it gives the single correct ID directly.
# -----------------------------------------------------------------------
_eos_ids: List[int] = []
if tokenizer.eos_token_id is not None:
    _eos_ids.append(int(tokenizer.eos_token_id))

for _st in GOLDEN_GENERATION_CONFIG.get("stop", []):
    if hasattr(tokenizer, "added_tokens_encoder") and _st in tokenizer.added_tokens_encoder:
        _tid = int(tokenizer.added_tokens_encoder[_st])
        if _tid not in _eos_ids: _eos_ids.append(_tid)
    else:
        try:
            _tids = tokenizer.encode(_st, add_special_tokens=False)
            if len(_tids) == 1 and _tids[0] not in _eos_ids:
                _eos_ids.append(_tids[0])
        except Exception:
            pass

_diag(f"[eos] effective eos_token_ids: {_eos_ids}")

_sdp_ctx_factory = None
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_k, SDPBackend as _SDPB
    _sdp_ctx_factory = lambda: _sdpa_k([_SDPB.MATH])
    _diag("[sdp] Using torch.nn.attention.sdpa_kernel([MATH])")
except (ImportError, AttributeError):
    pass
if _sdp_ctx_factory is None:
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        _sdp_ctx_factory = lambda: torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False)
        _diag("[sdp] Using torch.backends.cuda.sdp_kernel (legacy factory)")
    except Exception:
        _sdp_ctx_factory = _nullctx
        _diag("[sdp] sdp_kernel unavailable -- nullcontext")

_cfg_for_init = _model_cfg_for_cache
_init_opts: List[Dict[str, Any]] = []
if _nemh_cache_cls is not None:
    if _cfg_for_init is not None:
        _init_opts = [
            {"config": _cfg_for_init, "batch_size": 1, "dtype": compute_dtype, "device": input_device},
            {"config": _cfg_for_init, "batch_size": 1, "dtype": compute_dtype},
            {"config": _cfg_for_init, "max_batch_size": 1, "dtype": compute_dtype, "device": input_device},
            {"config": _cfg_for_init, "max_batch_size": 1, "dtype": compute_dtype},
            {"config": _cfg_for_init, "batch_size": 1},
            {},
        ]
    else:
        _init_opts = [{"device": input_device}, {}]
_best_init_kw = None

_diag("Inference setup complete"); print("Setup complete")
""")

# ---------------------------------------------------------------------------
# Cell 7: Load + sample problems + inference loop
# ---------------------------------------------------------------------------
code(r"""_all_problems = load_problems(PROBLEMS_PATH)
print(f"Total problems loaded: {len(_all_problems)}")

if MAX_PROBLEMS and MAX_PROBLEMS > 0:
    problems = stratified_sample(_all_problems, CATEGORY_QUOTA, seed=SEED)
    problems = problems[:MAX_PROBLEMS]
    print(f"Stratified sample: {len(problems)} problems")
    from collections import Counter as _Counter
    for cat, cnt in sorted(_Counter(classify_problem(r) for r in problems).items()):
        print(f"  {cat:<22}: {cnt}")
else:
    problems = _all_problems
    print(f"Using all {len(problems)} problems")

records: List[Dict[str, Any]] = []
_MAX_INPUT_TOKENS  = 768
_truncation_warned = False
_n_gen_errors      = 0

_jsonl_path = Path(OUTPUT_DIR) / "golden_validation_predictions.jsonl"
_done_ids: set = set()
if _jsonl_path.exists():
    with _jsonl_path.open("r", encoding="utf-8") as _rf:
        for _rline in _rf:
            try:
                _rd = json.loads(_rline.strip())
                if "problem_id" in _rd: _done_ids.add(_rd["problem_id"])
            except Exception:
                pass
    if _done_ids: _diag(f"[resume] Skipping {len(_done_ids)} already-completed problems")

with _jsonl_path.open("a", encoding="utf-8") as _jfh:
  for idx, record in enumerate(problems):
    pid = get_problem_id(record, idx)
    if pid in _done_ids: continue

    cat = record.get("_category") or classify_problem(record)
    question    = str(record.get("question", record.get("prompt", ""))).strip()
    gold_answer = str(record.get("answer", record.get("target", ""))).strip()

    prompt = build_prompt(record)
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)

    _was_truncated = False
    if inputs["input_ids"].shape[1] > _MAX_INPUT_TOKENS:
        inputs = {k: v[:, :_MAX_INPUT_TOKENS] for k, v in inputs.items()}
        _was_truncated = True
        if not _truncation_warned:
            _truncation_warned = True
            _diag(f"[warn] prompt truncated to {_MAX_INPUT_TOKENS} tokens at idx={idx}")

    _gen_inputs = {"input_ids": inputs["input_ids"]}
    if "attention_mask" in inputs: _gen_inputs["attention_mask"] = inputs["attention_mask"]
    _input_len = _gen_inputs["input_ids"].shape[1]

    _past_kv = None
    if _nemh_cache_cls is not None:
        _try_opts = [_best_init_kw] if _best_init_kw is not None else _init_opts
        for _kw in _try_opts:
            try:
                _past_kv = _nemh_cache_cls(**_kw)
                if _best_init_kw is None:
                    _best_init_kw = _kw; _diag(f"[cache] init OK: {list(_kw.keys())}")
                break
            except TypeError: continue
            except Exception as _cie:
                _diag(f"[cache] init error idx={idx}: {_cie}"); break
        if _past_kv is not None:
            try:
                _layers = (
                    getattr(_inner_model, "layers", None)
                    or getattr(getattr(_inner_model, "model", None), "layers", None)
                    or getattr(getattr(_inner_model, "backbone", None), "layers", None)
                )
                if _layers is not None:
                    for _ca in ("key_cache", "value_cache"):
                        for _li, _ts in enumerate(getattr(_past_kv, _ca, [])):
                            if not isinstance(_ts, torch.Tensor) or _li >= len(_layers): continue
                            try:
                                _ld = next(_layers[_li].parameters()).device
                                if _ts.device != _ld: getattr(_past_kv, _ca)[_li] = _ts.to(_ld)
                            except Exception: continue
                def _gsl(self, layer_idx=0):
                    for _t in getattr(self, "key_cache", []):
                        if isinstance(_t, torch.Tensor) and _t.dim() == 4: return int(_t.shape[-2])
                    return 0
                import types as _ty
                _past_kv.get_seq_length = _ty.MethodType(_gsl, _past_kv)
            except Exception as _dc_e:
                if idx == 0: _diag(f"[cache] device correction error: {_dc_e}")

    torch.cuda.empty_cache()

    try:
        t0 = time.time()
        with torch.no_grad():
            _gkw: Dict[str, Any] = dict(
                **_gen_inputs,
                max_new_tokens=GOLDEN_GENERATION_CONFIG["max_new_tokens"],
                do_sample=GOLDEN_GENERATION_CONFIG["do_sample"],
                repetition_penalty=GOLDEN_GENERATION_CONFIG["repetition_penalty"],
                eos_token_id=_eos_ids if _eos_ids else tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            if _past_kv is not None: _gkw["past_key_values"] = _past_kv
            with _sdp_ctx_factory(): output = model.generate(**_gkw)
        elapsed = time.time() - t0
    except Exception as _gen_exc:
        import traceback as _tb
        _diag(f"[{idx+1:4d}/{len(problems)}] {pid}: GENERATE_ERROR {_gen_exc}")
        if _n_gen_errors < 3: _diag(_tb.format_exc())
        _n_gen_errors += 1
        gc.collect(); torch.cuda.empty_cache()
        entry = {
            "problem_id": pid, "category": cat, "subcategory": "unknown",
            "question": question, "gold_answer": gold_answer,
            "pred_answer": "", "raw_output": f"[GENERATE_ERROR: {_gen_exc}]",
            "reasoning_text": "", "final_answer_text": "",
            "is_correct": False, "parse_success": False,
            "parse_error_type": "generate_error",
            "generation_token_count": 0, "finish_reason": "error",
            "was_prompt_truncated": _was_truncated, "elapsed_seconds": 0.0, "seed": SEED,
        }
        records.append(entry)
        _jfh.write(json.dumps(entry, ensure_ascii=False) + "\n"); _jfh.flush()
        continue

    gen_ids = output[0][_input_len:]
    raw_out = tokenizer.decode(gen_ids, skip_special_tokens=True)
    pred, reasoning, final_txt = extract_answer(raw_out)
    is_ok  = answers_match(pred, gold_answer)
    n_tok  = int(gen_ids.shape[0])
    _last_t = int(gen_ids[-1]) if n_tok > 0 else None
    _eos_set = set(_eos_ids) if _eos_ids else {tokenizer.eos_token_id}
    finish  = ("unknown" if _last_t is None
               else "eos" if _last_t in _eos_set else "length")

    entry = {
        "problem_id": pid, "category": cat, "subcategory": "unknown",
        "question": question, "gold_answer": gold_answer,
        "pred_answer": pred or "", "raw_output": raw_out,
        "reasoning_text": reasoning, "final_answer_text": final_txt or "",
        "is_correct": is_ok, "parse_success": pred is not None,
        "parse_error_type": parse_error_type(pred, raw_out) or "",
        "generation_token_count": n_tok, "finish_reason": finish,
        "was_prompt_truncated": _was_truncated,
        "elapsed_seconds": round(elapsed, 2), "seed": SEED,
    }
    records.append(entry)
    _jfh.write(json.dumps(entry, ensure_ascii=False) + "\n"); _jfh.flush()

    tok_s  = n_tok / max(elapsed, 0.01)
    status = "OK" if is_ok else ("PARSE_FAIL" if pred is None else "WRONG")
    print(f"[{idx+1:4d}/{len(problems)}] {pid}: {status}  "
          f"tokens={n_tok}  {tok_s:.1f} tok/s"
          + (" [TRUNC]" if _was_truncated else ""))

print(f"\nInference done: {len(records)} problems processed")
""")

# ---------------------------------------------------------------------------
# Cell 8: Summary
# ---------------------------------------------------------------------------
code(r"""from collections import Counter

n_total   = len(records)
n_correct = sum(1 for r in records if r["is_correct"])
n_parse   = sum(1 for r in records if r["parse_success"])
avg_tok   = sum(r["generation_token_count"] for r in records) / max(n_total, 1)
avg_s     = sum(r["elapsed_seconds"] for r in records) / max(n_total, 1)

_summary_path = Path(OUTPUT_DIR) / "golden_validation_summary.csv"

def _cat_row(split, cat, sub, grp):
    nc  = sum(1 for r in grp if r["is_correct"])
    np_ = sum(1 for r in grp if r["parse_success"])
    at  = sum(r["generation_token_count"] for r in grp) / max(len(grp), 1)
    return {"split": split, "category": cat, "subcategory": sub,
            "n": len(grp), "correct": nc,
            "accuracy": round(nc / max(len(grp), 1), 4),
            "n_parse_success": np_,
            "parse_success_rate": round(np_ / max(len(grp), 1), 4),
            "avg_generation_token_count": round(at, 1)}

_overall = [_cat_row("overall", "ALL", "ALL", records)]
_cg: Dict[str, List] = defaultdict(list)
for r in records: _cg[r["category"]].append(r)
_per_cat = [_cat_row("category", c, "ALL", g) for c, g in sorted(_cg.items())]

_fields = ["split","category","subcategory","n","correct","accuracy",
           "n_parse_success","parse_success_rate","avg_generation_token_count"]
with _summary_path.open("w", encoding="utf-8", newline="") as _fh:
    _w = csv.DictWriter(_fh, fieldnames=_fields)
    _w.writeheader(); _w.writerows(_overall + _per_cat)

print(f"\n{'='*55}")
print(f"Total     : {n_total}")
print(f"Accuracy  : {n_correct}/{n_total} = {n_correct/max(n_total,1):.4f}")
print(f"Parse OK  : {n_parse}/{n_total} = {n_parse/max(n_total,1):.4f}")
print(f"Avg tokens: {avg_tok:.1f}   Avg sec/problem: {avg_s:.1f}")
print(f"{'='*55}")
print("\nPer-category:")
for row in _per_cat:
    print(f"  {row['category']:<22}: {row['correct']:3d}/{row['n']:3d} = {row['accuracy']:.3f}")
print(f"\nOutputs:")
print(f"  {_jsonl_path}")
print(f"  {_summary_path}")
print(f"  {_diag_path}")
_fr = Counter(r["finish_reason"] for r in records)
print(f"\nFinish reasons: {dict(_fr)}")
_tok1 = sum(1 for r in records if r["generation_token_count"] <= 1)
if _tok1: print(f"WARNING: {_tok1} problems generated <= 1 token (EOS issue?)")
""")

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out_path = "phase3_rtx_analysis.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print(f"Generated: {out_path}  ({len(cells)} cells)")
