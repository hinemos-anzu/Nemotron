#!/usr/bin/env python3
"""Generate nemotron_rtx_inference.ipynb for RTX Pro 5000 (offline)."""
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
# Cell 1: Title
# ---------------------------------------------------------------------------
md("""# Nemotron-H 30B — Golden Baseline Inference (RTX Pro 5000)

オフライン実行版：インターネット接続不要、全コード自己完結。

## 手順
1. **Cell 2** のパスを設定してから実行
2. 全セルを上から順番に実行（`Run All`）
3. 結果は `OUTPUT_DIR` に保存される

## 出力ファイル
- `golden_validation_predictions.jsonl` — 各問題の予測
- `golden_validation_summary.csv` — カテゴリ別精度
- `step2_diagnostics.txt` — 診断ログ（OOM再起動後も追記）
""")

# ---------------------------------------------------------------------------
# Cell 2: Configuration
# ---------------------------------------------------------------------------
code(r"""# ============================================================
# CONFIGURATION — ここを編集してから実行
# ============================================================

# Goldenアダプターのパス（adapter_model.safetensors が含まれるディレクトリ）
# Kaggle: /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20
ADAPTER_PATH = "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20"

# Nemotron-H 30B ベースモデルのパス
# Kaggle: /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"

# 問題ファイル（JSONL または CSV）
# Kaggle: /kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv
PROBLEMS_PATH = "/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv"

# 結果出力先
OUTPUT_DIR = "/kaggle/working/phase3_analysis"

# カテゴリマップ CSV（なければ空文字 "" でOK、Step 1で自動生成される）
CATEGORY_MAP_PATH = ""

# 乱数シード（Golden Baseline と一致）
SEED = 42

# 問題数上限（0 = 全件、動作確認は 5 推奨）
MAX_PROBLEMS = 0

# Kaggle の壊れた mamba_ssm を差し替えるか（RTX Pro 5000 では False）
USE_MAMBA_PATCH = False

# ============================================================
# Generation config — Golden Baseline と完全一致（変更禁止）
# ============================================================
GOLDEN_GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "temperature": 0.0,
    "do_sample": False,
    "repetition_penalty": 1.0,
    "stop": ["<|endoftext|>", "<|im_end|>"],
}

# ============================================================
from pathlib import Path
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"Output dir : {OUTPUT_DIR}")
print(f"Adapter    : {ADAPTER_PATH}")
print(f"Model      : {MODEL_PATH}")
print(f"Problems   : {PROBLEMS_PATH}")
print(f"Limit      : {MAX_PROBLEMS or 'all'}")
""")

# ---------------------------------------------------------------------------
# Cell 3: Package check + all utility functions
# ---------------------------------------------------------------------------
code(r"""# --- Package check ---
import importlib, sys
for _pkg in ["torch", "transformers", "peft", "bitsandbytes"]:
    try:
        _m = importlib.import_module(_pkg)
        print(f"  {_pkg}: OK  v{getattr(_m, '__version__', '?')}")
    except ImportError:
        print(f"  {_pkg}: NOT FOUND — install and restart kernel")

# --- Imports ---
import csv, gc, json, os, re, time, datetime, types as _types
from collections import defaultdict
from contextlib import nullcontext as _nullctx
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Answer extraction helpers ---
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}", re.DOTALL)
THEREFORE_RE = re.compile(
    r"(?:the (?:final )?answer is|therefore|so the answer is|answer:)"
    r"\s*(?:\\boxed\{)?([^\n\.\\{}]+)",
    re.IGNORECASE,
)
LAST_LINE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _\-]*$")

def extract_answer(raw_output):
    m = BOXED_RE.search(raw_output)
    if m:
        return m.group(1).strip(), raw_output[:m.start()].strip(), m.group(0)
    m = THEREFORE_RE.search(raw_output)
    if m:
        answer = re.sub(r"[^A-Za-z0-9]", "", m.group(1)).strip()
        if answer:
            return answer, raw_output[:m.start()].strip(), m.group(0)
    lines = [l.strip() for l in raw_output.strip().splitlines() if l.strip()]
    if lines:
        m2 = LAST_LINE_RE.search(lines[-1])
        if m2:
            return m2.group(0).strip(), "\n".join(lines[:-1]), lines[-1]
    return None, raw_output.strip(), None

def normalize_answer(answer):
    if answer is None:
        return ""
    return re.sub(r"\s+", "", answer).strip().upper()

def answers_match(pred, gold):
    return bool(pred) and normalize_answer(pred) == normalize_answer(gold)

def parse_error_type(pred, raw_output):
    if pred is not None:
        return None
    if BOXED_RE.search(raw_output):
        return "boxed_found_but_empty"
    if len(raw_output.strip()) < 10:
        return "empty_output"
    return "no_extractable_answer"

# --- Problem / category loaders ---
def load_category_map(path):
    result = {}
    p = Path(path)
    if not p or not p.exists():
        return result
    with p.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            result[row["problem_id"]] = {
                "category": row.get("category", "other"),
                "subcategory": row.get("subcategory", "unknown"),
            }
    return result

def _load_csv_problems(path):
    QCOLS = ("question", "problem", "prompt", "input", "text")
    ACOLS = ("answer", "solution", "target", "output", "label")
    ICOLS = ("id", "problem_id", "uid", "sample_id", "index")
    records = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return records
        fl = {f.lower(): f for f in reader.fieldnames}
        qc = next((fl[c] for c in QCOLS if c in fl), None)
        ac = next((fl[c] for c in ACOLS if c in fl), None)
        ic = next((fl[c] for c in ICOLS if c in fl), None)
        for idx, row in enumerate(reader):
            rec = dict(row)
            if qc and qc != "question":
                rec["question"] = row[qc]
            if ac and ac != "answer":
                rec["answer"] = row[ac]
            if ic and ic != "problem_id":
                rec["problem_id"] = row[ic]
            elif "problem_id" not in rec:
                rec["problem_id"] = f"row_{idx}"
            records.append(rec)
    return records

def load_problems(path):
    p = Path(path)
    if not p.exists():
        return []
    if p.suffix.lower() == ".csv":
        return _load_csv_problems(p)
    records = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def get_problem_id(record, idx):
    for key in ("problem_id", "id", "uid", "sample_id"):
        v = record.get(key)
        if v:
            return str(v)
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
                    parts.append(f"  {i}. Input: {ex.get('input', '')} -> Output: {ex.get('output', '')}")
                else:
                    parts.append(f"  {i}. {ex}")
        parts.append("")
    parts.append(f"Question: {question}")
    parts.append("\nThink step by step. Put your final answer inside \\boxed{}.")
    return "\n".join(parts)

# --- Optional Kaggle mamba_ssm patch ---
def _apply_mamba_patch():
    import glob as _glob, importlib as _il, re as _re, sys
    stubs = {str(Path(p).parent.parent)
             for p in _glob.glob("/kaggle/input/datasets/hinemos/**/mamba_ssm/__init__.py", recursive=True)}
    if not stubs:
        print("[mamba_patch] No stub found — model loading may fail"); return
    def _sz(d):
        p = Path(d) / "mamba_ssm/ops/selective_scan_interface.py"
        return p.stat().st_size if p.exists() else 0
    best = sorted(stubs, key=lambda d: (_sz(d), d), reverse=True)[0]
    for k in list(sys.modules):
        if k == "mamba_ssm" or k.startswith("mamba_ssm."):
            del sys.modules[k]
    if best not in sys.path:
        sys.path.insert(0, best)
    try:
        stub = _il.import_module("mamba_ssm")
        raw_v = getattr(stub, "__version__", "0.0.0")
        stub.__version__ = "1.2.0"
        sys.modules["mamba_ssm"] = stub
        print(f"[mamba_patch] OK: {best}  (version {raw_v!r} -> '1.2.0')")
    except Exception as e:
        print(f"[mamba_patch] Warning: {e}")

print("Utilities loaded OK")
""")

# ---------------------------------------------------------------------------
# Cell 4: Model loading
# ---------------------------------------------------------------------------
code(r"""import os, gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
from peft import PeftModel

# CUDA allocator: must be set before any CUDA alloc
_ea = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _ea:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (f"{_ea},expandable_segments:True" if _ea else "expandable_segments:True")

set_seed(SEED)
torch.manual_seed(SEED)

# --- Diagnostics ---
_diag_path = Path(OUTPUT_DIR) / "step2_diagnostics.txt"
_sep = f"\n{'='*60}\nRun started {datetime.datetime.utcnow().isoformat()}Z\n{'='*60}\n"
with _diag_path.open("a") as fh:
    fh.write(_sep)

def _diag(msg):
    print(msg, flush=True)
    with _diag_path.open("a") as fh:
        fh.write(msg + "\n")

_diag("=== Nemotron-H 30B Inference ===")
_diag(f"PyTorch {torch.__version__}  CUDA {torch.version.cuda}")
for _gi in range(torch.cuda.device_count()):
    _pr = torch.cuda.get_device_properties(_gi)
    _fr, _tot = torch.cuda.mem_get_info(_gi)
    _diag(f"GPU {_gi}: {_pr.name}  SM{_pr.major}.{_pr.minor}  {_tot/1024**3:.1f}GB total  {_fr/1024**3:.1f}GB free")

# SM capability: cuDNN bf16 conv1d kernel requires SM >= 8.0 (Ampere+)
# RTX Pro 5000 = SM 8.9 (Ada Lovelace) or SM 10.x (Blackwell) -> has_bf16 = True
_cap0 = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
has_bf16 = _cap0[0] >= 8
compute_dtype = torch.bfloat16 if has_bf16 else torch.float16
_diag(f"GPU SM cap: {_cap0}  bf16 conv kernel: {has_bf16}  -> compute_dtype={compute_dtype}")

if USE_MAMBA_PATCH:
    _apply_mamba_patch()

# --- Tokenizer ---
_diag(f"Loading tokenizer from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
_diag(f"Tokenizer OK  eos_token_id={tokenizer.eos_token_id}")

gc.collect()
torch.cuda.empty_cache()

# GPU budget (80% per GPU)
_n_gpus = torch.cuda.device_count()
_gpu_budget = {
    i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.80 / 1024**3)}GiB"
    for i in range(_n_gpus)
}
_diag(f"GPU budget (80%): {_gpu_budget}")

# --- Load base model in 4-bit NF4 ---
_diag("Loading base model (4-bit NF4) ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=compute_dtype,
    device_map="auto",
    max_memory=_gpu_budget,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
for _gi in range(_n_gpus):
    _fr, _tot = torch.cuda.mem_get_info(_gi)
    _diag(f"GPU {_gi} after base model: {_fr/1024**3:.1f}GB free / {_tot/1024**3:.1f}GB")
_diag("Base model OK")

# Patch PEFT._update_offload to skip Mamba MoE keys absent from named_modules()
try:
    from peft import peft_model as _peft_mod
    _orig_upd = _peft_mod.PeftModel._update_offload
    def _safe_update_offload(self, offload_index, adapters_weights):
        if not offload_index:
            return offload_index
        try:
            return _orig_upd(self, offload_index, adapters_weights)
        except KeyError as _ke:
            print(f"[peft_patch] _update_offload KeyError skipped: {_ke}")
            return offload_index
    _peft_mod.PeftModel._update_offload = _safe_update_offload
    print("[peft_patch] Patched _update_offload OK")
except Exception as _pe:
    print(f"[peft_patch] Warning: {_pe}")

model.eval()

# --- Load PEFT adapter ---
_diag(f"Loading adapter from {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
_diag("PEFT adapter loaded OK")

# SM < 8.0 only: patch nn.Conv1d to cast bf16->fp16 at runtime
# (cuDNN bf16 conv1d kernel absent on SM 7.x/Turing).
# Keeps compute_dtype=bfloat16 so PEFT loads without dtype mismatch.
# On RTX Pro 5000 (SM >= 8.0) this block is skipped entirely.
if not has_bf16:
    _diag(f"[conv1d-fp16] SM {_cap0} < 8.0: patching nn.Conv1d for bf16->fp16 workaround")
    def _make_conv1d_hooks():
        def pre_h(mod, inputs):
            return tuple(
                t.to(torch.float16) if isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16 else t
                for t in inputs
            )
        def post_h(mod, inp, output):
            if isinstance(output, torch.Tensor) and output.dtype == torch.float16:
                return output.to(torch.bfloat16)
            return output
        return pre_h, post_h
    _n_conv = 0
    for _n, _m in model.named_modules():
        if isinstance(_m, torch.nn.Conv1d):
            _m.weight.data = _m.weight.data.to(torch.float16)
            if _m.bias is not None:
                _m.bias.data = _m.bias.data.to(torch.float16)
            _ph, _poh = _make_conv1d_hooks()
            _m.register_forward_pre_hook(_ph)
            _m.register_forward_hook(_poh)
            _n_conv += 1
    _diag(f"[conv1d-fp16] Patched {_n_conv} Conv1d modules")

gc.collect()
torch.cuda.empty_cache()
for _gi in range(_n_gpus):
    _fr, _tot = torch.cuda.mem_get_info(_gi)
    _diag(f"GPU {_gi} after PEFT+GC: {_fr/1024**3:.1f}GB free / {_tot/1024**3:.1f}GB")

_diag("Model + adapter ready")
print("Model ready")
""")

# ---------------------------------------------------------------------------
# Cell 5: Warmup + cache discovery + SDP setup
# ---------------------------------------------------------------------------
code(r"""import inspect as _inspect

# Find input device (first CUDA parameter)
def _infer_input_device(m):
    for p in m.parameters():
        if p.device.type == "cuda":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_device = _infer_input_device(model)
_diag(f"Input device: {input_device}")

# Warmup: force lazy imports inside model's forward pass
# (HybridMambaAttentionDynamicCache is only imported the first time generate() runs)
_diag("[warmup] 1-token warmup to force lazy imports...")
try:
    _wu_text = tokenizer.decode([tokenizer.eos_token_id or 1] * 4)
    _wu_ids = tokenizer(_wu_text, return_tensors="pt").input_ids.to(input_device)
    with torch.no_grad():
        _ = model.generate(
            input_ids=_wu_ids,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    del _, _wu_ids
    gc.collect(); torch.cuda.empty_cache()
    _diag("[warmup] Done")
except Exception as _wu_e:
    _diag(f"[warmup] Error (non-fatal): {type(_wu_e).__name__}: {_wu_e}")

# Discover HybridMambaAttentionDynamicCache from sys.modules
import sys as _sys
_nemh_cache_cls = None
_model_cfg_for_cache = None
_inner_model = getattr(getattr(model, "base_model", model), "model", model)
_model_cls_name = type(_inner_model).__name__
_diag(f"[cache] model class: {_model_cls_name}")

_cache_class_names = [
    "HybridMambaAttentionDynamicCache",  # actual class in modeling_nemotron_h.py
    "NemotronHHybridDynamicCache",        # fallback for other variants
]
for _mn, _mod in list(_sys.modules.items()):
    if _mod is None: continue
    for _cn in _cache_class_names:
        try:
            _cls = getattr(_mod, _cn, None)
        except Exception:
            continue
        if _cls is not None and isinstance(_cls, type):
            _nemh_cache_cls = _cls
            _diag(f"[cache] Found {_cn} in module: {_mn}")
            break
    if _nemh_cache_cls is not None:
        break

if _nemh_cache_cls is None:
    _diag("[cache] Cache class not found — OOM risk on long generation")
else:
    try:
        _diag(f"[cache] {_nemh_cache_cls.__name__}.__init__ sig: {_inspect.signature(_nemh_cache_cls.__init__)}")
    except Exception:
        pass
    _model_cfg_for_cache = (
        getattr(model, "config", None)
        or getattr(getattr(model, "base_model", None), "config", None)
    )
    if _model_cfg_for_cache is None:
        try:
            _model_cfg_for_cache = model.base_model.model.config
        except Exception:
            pass
    _diag(f"[cache] config found: {_model_cfg_for_cache is not None}  -> SSM states will be cached")

# Build EOS token list from stop tokens
_eos_ids: List[int] = []
if tokenizer.eos_token_id is not None:
    _eos_ids.append(int(tokenizer.eos_token_id))
for _st in GOLDEN_GENERATION_CONFIG.get("stop", []):
    try:
        _tids = tokenizer.encode(_st, add_special_tokens=False)
        if len(_tids) == 1 and _tids[0] not in _eos_ids:
            _eos_ids.append(_tids[0])
    except Exception:
        pass
_diag(f"[eos] effective eos_token_ids: {_eos_ids}")

# SDP context factory — fresh CM per generate() call.
# torch.backends.cuda.sdp_kernel() is generator-based (single-use in Python 3.12).
# torch.nn.attention.sdpa_kernel is class-based and reusable.
_sdp_ctx_factory = None
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel_new, SDPBackend as _SDPBackend
    _sdp_ctx_factory = lambda: _sdpa_kernel_new([_SDPBackend.MATH])
    _diag("[sdp] Using torch.nn.attention.sdpa_kernel([MATH])")
except (ImportError, AttributeError):
    pass
if _sdp_ctx_factory is None:
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        _sdp_ctx_factory = lambda: torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False)
        _diag("[sdp] Using torch.backends.cuda.sdp_kernel (legacy, per-call factory)")
    except (AttributeError, TypeError, RuntimeError):
        _sdp_ctx_factory = _nullctx
        _diag("[sdp] sdp_kernel unavailable — using nullcontext")

# Cache init options (probed once; best kwargs saved after first success)
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

_diag("Inference setup complete")
print("Setup complete — ready to run inference")
""")

# ---------------------------------------------------------------------------
# Cell 6: Load problems + inference loop
# ---------------------------------------------------------------------------
code(r"""# --- Load problems ---
problems = load_problems(PROBLEMS_PATH)
if MAX_PROBLEMS and MAX_PROBLEMS > 0:
    problems = problems[:MAX_PROBLEMS]
    print(f"[max-problems] Capped at {MAX_PROBLEMS}")
category_map = load_category_map(CATEGORY_MAP_PATH) if CATEGORY_MAP_PATH else {}
print(f"Problems loaded   : {len(problems)}")
print(f"Category map size : {len(category_map)}")

# --- Inference loop ---
records: List[Dict[str, Any]] = []
_MAX_INPUT_TOKENS = 768   # cap prompt to keep SSM intermediate tensors small
_truncation_warned = False
_n_gen_errors = 0

_jsonl_path = Path(OUTPUT_DIR) / "golden_validation_predictions.jsonl"

# Resume support: skip already-completed problems (survives OOM restarts)
_done_ids: set = set()
if _jsonl_path.exists():
    with _jsonl_path.open("r", encoding="utf-8") as _rf:
        for _rline in _rf:
            try:
                _rd = json.loads(_rline.strip())
                if "problem_id" in _rd:
                    _done_ids.add(_rd["problem_id"])
            except Exception:
                pass
    if _done_ids:
        _diag(f"[resume] Skipping {len(_done_ids)} already-completed problems")

with _jsonl_path.open("a", encoding="utf-8") as _jsonl_fh:
  for idx, record in enumerate(problems):
    pid = get_problem_id(record, idx)
    if pid in _done_ids:
        continue

    cat_info = category_map.get(pid, {"category": "other", "subcategory": "unknown"})
    question = str(record.get("question", record.get("prompt", ""))).strip()
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

    _gen_inputs: Dict[str, Any] = {"input_ids": inputs["input_ids"]}
    if "attention_mask" in inputs:
        _gen_inputs["attention_mask"] = inputs["attention_mask"]
    _input_len = _gen_inputs["input_ids"].shape[1]

    # Initialize a fresh HybridMambaAttentionDynamicCache per problem
    _past_kv = None
    if _nemh_cache_cls is not None:
        _try_opts = [_best_init_kw] if _best_init_kw is not None else _init_opts
        for _ik, _kw in enumerate(_try_opts):
            try:
                _past_kv = _nemh_cache_cls(**_kw)
                if _best_init_kw is None:
                    _best_init_kw = _kw
                    _diag(f"[cache] init OK with kwargs: {list(_kw.keys())}")
                break
            except TypeError:
                continue
            except Exception as _cie:
                _diag(f"[cache] init error (idx={idx}): {type(_cie).__name__}: {_cie}")
                break
        if _past_kv is None and (idx == 0 or _best_init_kw is None):
            _diag(f"[cache] all init signatures failed at idx={idx}")

        # Per-layer device correction (handles multi-GPU device_map="auto" splits)
        if _past_kv is not None:
            try:
                # NemotronH stores layers under .backbone.layers
                _layers = (
                    getattr(_inner_model, "layers", None)
                    or getattr(getattr(_inner_model, "model", None), "layers", None)
                    or getattr(getattr(_inner_model, "backbone", None), "layers", None)
                )
                if idx == 0:
                    _diag(f"[cache] _layers found: {_layers is not None}"
                          f"  (.layers / .model.layers / .backbone.layers)")
                if _layers is not None:
                    for _cache_attr in ("key_cache", "value_cache"):
                        _tc = getattr(_past_kv, _cache_attr, [])
                        for _li, _ts in enumerate(_tc):
                            if not isinstance(_ts, torch.Tensor) or _li >= len(_layers):
                                continue
                            try:
                                _ld = next(_layers[_li].parameters()).device
                                if _ts.device != _ld:
                                    getattr(_past_kv, _cache_attr)[_li] = _ts.to(_ld)
                            except Exception:
                                continue
                    if idx == 0:
                        _devs = [str(_past_kv.key_cache[i].device)
                                 for i in range(len(_past_kv.key_cache))
                                 if isinstance(_past_kv.key_cache[i], torch.Tensor)]
                        _diag(f"[cache] SSM/KV devices after correction: {sorted(set(_devs))} ({len(_devs)} tensors)")

                # Monkey-patch get_seq_length: DynamicCache.get_seq_length(layer_idx=0)
                # returns key_cache[0].shape[-2]. For Mamba SSM states [batch, d_state, d_inner]
                # this equals d_state (~64-256) instead of 0, corrupting position_ids.
                # Return 0 for a fresh cache (no 4-D attention tensors yet).
                def _get_seq_length_safe(self, layer_idx=0):
                    for _t in getattr(self, "key_cache", []):
                        if isinstance(_t, torch.Tensor) and _t.dim() == 4:
                            return int(_t.shape[-2])
                    return 0
                _past_kv.get_seq_length = _types.MethodType(_get_seq_length_safe, _past_kv)
            except Exception as _dc_e:
                if idx == 0:
                    _diag(f"[cache] device correction error: {type(_dc_e).__name__}: {_dc_e}")

    torch.cuda.empty_cache()

    try:
        t0 = time.time()
        with torch.no_grad():
            _generate_kwargs: Dict[str, Any] = dict(
                **_gen_inputs,
                max_new_tokens=GOLDEN_GENERATION_CONFIG["max_new_tokens"],
                do_sample=GOLDEN_GENERATION_CONFIG["do_sample"],
                repetition_penalty=GOLDEN_GENERATION_CONFIG["repetition_penalty"],
                eos_token_id=_eos_ids if _eos_ids else tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            if _past_kv is not None:
                _generate_kwargs["past_key_values"] = _past_kv
            with _sdp_ctx_factory():
                output = model.generate(**_generate_kwargs)
        elapsed = time.time() - t0
    except Exception as _gen_exc:
        import traceback as _tb
        _diag(f"[{idx+1:4d}/{len(problems)}] {pid}: GENERATE_ERROR "
              f"{type(_gen_exc).__name__}: {_gen_exc}")
        if _n_gen_errors < 3:
            _diag(f"[traceback]\n{_tb.format_exc()}")
        _n_gen_errors += 1
        gc.collect(); torch.cuda.empty_cache()
        entry = {
            "problem_id": pid, "category": cat_info["category"],
            "subcategory": cat_info["subcategory"], "question": question,
            "gold_answer": gold_answer, "pred_answer": "",
            "raw_output": f"[GENERATE_ERROR: {type(_gen_exc).__name__}: {_gen_exc}]",
            "reasoning_text": "", "final_answer_text": "",
            "is_correct": False, "parse_success": False,
            "parse_error_type": "generate_error",
            "generation_token_count": 0, "finish_reason": "error",
            "was_prompt_truncated": _was_truncated,
            "elapsed_seconds": 0.0, "seed": SEED,
            "generation_config": dict(GOLDEN_GENERATION_CONFIG),
        }
        records.append(entry)
        _jsonl_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _jsonl_fh.flush()
        continue

    generated_ids = output[0][_input_len:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    pred_answer, reasoning_text, final_answer_text = extract_answer(raw_output)
    is_correct = answers_match(pred_answer, gold_answer)
    parse_success = pred_answer is not None
    err_type = parse_error_type(pred_answer, raw_output)
    n_tokens = int(generated_ids.shape[0])

    _last_tok = int(generated_ids[-1]) if n_tokens > 0 else None
    _eos_set = set(_eos_ids) if _eos_ids else {tokenizer.eos_token_id}
    finish_reason = (
        "unknown" if _last_tok is None
        else "eos" if _last_tok in _eos_set
        else "length"
    )

    entry = {
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
        "was_prompt_truncated": _was_truncated,
        "elapsed_seconds": round(elapsed, 2),
        "seed": SEED,
        "generation_config": dict(GOLDEN_GENERATION_CONFIG),
    }
    records.append(entry)
    _jsonl_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _jsonl_fh.flush()

    status = "OK" if is_correct else ("PARSE_FAIL" if not parse_success else "WRONG")
    print(f"[{idx+1:4d}/{len(problems)}] {pid}: {status}  tokens={n_tokens}"
          + (" [TRUNC]" if _was_truncated else ""))

print(f"\nInference done: {len(records)} problems processed")
""")

# ---------------------------------------------------------------------------
# Cell 7: Write results + summary
# ---------------------------------------------------------------------------
code(r"""# Write clean JSONL (raw_output truncated to 4000 chars)
_pred_path = Path(OUTPUT_DIR) / "golden_validation_predictions.jsonl"
_summary_path = Path(OUTPUT_DIR) / "golden_validation_summary.csv"

with _pred_path.open("w", encoding="utf-8") as fh:
    for rec in records:
        out = dict(rec)
        if len(out.get("raw_output", "")) > 4000:
            out["raw_output"] = out["raw_output"][:4000] + " ...[truncated]"
        if len(out.get("reasoning_text", "")) > 3000:
            out["reasoning_text"] = out["reasoning_text"][:3000] + " ...[truncated]"
        out.pop("generation_config", None)
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")

# Summary CSV (overall + per-category + per-subcategory)
n_total   = len(records)
n_correct = sum(1 for r in records if r["is_correct"])
n_parse   = sum(1 for r in records if r["parse_success"])
avg_tok   = sum(r["generation_token_count"] for r in records) / max(n_total, 1)

def _cat_row(split, cat, sub, grp):
    nc = sum(1 for r in grp if r["is_correct"])
    np_ = sum(1 for r in grp if r["parse_success"])
    at = sum(r["generation_token_count"] for r in grp) / max(len(grp), 1)
    return {"split": split, "category": cat, "subcategory": sub,
            "n": len(grp), "correct": nc,
            "accuracy": round(nc / max(len(grp), 1), 4),
            "n_parse_success": np_,
            "parse_success_rate": round(np_ / max(len(grp), 1), 4),
            "avg_generation_token_count": round(at, 1)}

overall = [_cat_row("overall", "ALL", "ALL", records)]
cat_groups = defaultdict(list)
for r in records:
    cat_groups[r["category"]].append(r)
per_cat = [_cat_row("category", cat, "ALL", grp)
           for cat, grp in sorted(cat_groups.items())]
subcat_groups = defaultdict(list)
for r in records:
    subcat_groups[(r["category"], r["subcategory"])].append(r)
per_sub = [_cat_row("subcategory", cat, sub, grp)
           for (cat, sub), grp in sorted(subcat_groups.items())]

fields = ["split","category","subcategory","n","correct","accuracy",
          "n_parse_success","parse_success_rate","avg_generation_token_count"]
with _summary_path.open("w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(overall + per_cat + per_sub)

# Print summary
print(f"\n{'='*55}")
print(f"Total problems : {n_total}")
print(f"Accuracy       : {n_correct}/{n_total} = {n_correct/max(n_total,1):.4f}")
print(f"Parse success  : {n_parse}/{n_total} = {n_parse/max(n_total,1):.4f}")
print(f"Avg tokens     : {avg_tok:.1f}")
print(f"{'='*55}")
print(f"\nPer-category accuracy:")
for row in per_cat:
    print(f"  {row['category']:<20s}: {row['correct']:3d}/{row['n']:3d} = {row['accuracy']:.3f}")
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print(f"  {_pred_path.name}")
print(f"  {_summary_path.name}")
print(f"  step2_diagnostics.txt")
""")

# ---------------------------------------------------------------------------
# Write the notebook
# ---------------------------------------------------------------------------
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

out_path = "nemotron_rtx_inference.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print(f"Generated: {out_path}  ({len(cells)} cells)")
