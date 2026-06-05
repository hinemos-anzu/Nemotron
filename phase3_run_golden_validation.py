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

    After injection we also normalise mamba_ssm.__version__ to a valid PEP 440
    string below 2.0.4 ('1.2.0') so that:
      - is_mamba_2_ssm_available() → False  (avoids mamba2 CUDA path)
      - is_mamba_ssm_available()   → True   (model uses mamba1 stub ops)
    """
    import glob as _glob
    import importlib as _importlib
    import re as _re
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

    # Pre-import the stub so we can normalise its __version__ BEFORE
    # is_mamba_2_ssm_available() reads it.  The stubs ship with versions like
    # '2.2.2.stub' which are not valid PEP 440 strings — packaging.version.parse()
    # raises InvalidVersion on them.  Force version to '1.2.0' so the check
    # returns False for mamba2 (< 2.0.4) while mamba1 remains "available".
    try:
        stub = _importlib.import_module("mamba_ssm")
        raw_v = getattr(stub, "__version__", "0.0.0")
        # Strip non-numeric/dot suffix: '2.2.2.stub' → '2.2.2' → force to 1.2.0
        clean_v = _re.sub(r"[^0-9.].*$", "", raw_v).rstrip(".")
        stub.__version__ = "1.2.0"  # < 2.0.4 → is_mamba_2_ssm_available() = False
        sys.modules["mamba_ssm"] = stub
        print(f"[mamba_patch] Patched mamba_ssm from {best} "
              f"(version {raw_v!r} → '1.2.0')")
    except Exception as exc:
        print(f"[mamba_patch] Warning: version normalisation failed: {exc}")


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
    # PYTORCH_CUDA_ALLOC_CONF must be set BEFORE any CUDA allocation — the allocator
    # reads its config only on first init.  _apply_mamba_patch() calls
    # import_module('mamba_ssm') whose __init__.py may import torch (and thus
    # initialize the allocator), so we must set the env var first.
    # Use merge logic instead of setdefault so we don't clobber an existing value
    # that the parent process / notebook cell may have already set.
    import os as _os
    _existing_alloc = _os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in _existing_alloc:
        _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"{_existing_alloc},expandable_segments:True"
            if _existing_alloc else "expandable_segments:True"
        )

    # Must patch mamba_ssm BEFORE any transformers imports so that
    # modeling_nemotron_h.py's module-level is_mamba_2_ssm_available() check
    # uses our stub instead of the broken CUDA extension.
    _apply_mamba_patch()

    import gc as _gc
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import PeftModel

    set_seed(seed)
    torch.manual_seed(seed)

    # ── Diagnostics appended to file so they survive a kernel OOM restart ───────
    # Use append mode (not write_text("")) so a prior run's diagnostics are
    # preserved when the script is re-invoked after a crash.
    import datetime as _dt
    _diag_path = output_dir / "step2_diagnostics.txt"
    _diag_sep = (
        f"\n{'='*60}\n"
        f"Run started {_dt.datetime.utcnow().isoformat()}Z\n"
        f"{'='*60}\n"
    )
    with _diag_path.open("a") as _fh:
        _fh.write(_diag_sep)

    def _diag(msg: str) -> None:
        print(msg, flush=True)
        with _diag_path.open("a") as _fh:
            _fh.write(msg + "\n")

    _diag("=== Step 2 diagnostics ===")
    _diag(f"PyTorch {torch.__version__}  CUDA {torch.version.cuda}")
    for _gi in range(torch.cuda.device_count()):
        _pr = torch.cuda.get_device_properties(_gi)
        _fr, _tot = torch.cuda.mem_get_info(_gi)
        _diag(f"GPU {_gi}: {_pr.name}  {_tot/1024**3:.1f}GB total  {_fr/1024**3:.1f}GB free")

    # T4 (Volta) doesn't support bfloat16 natively
    has_bf16 = (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    compute_dtype = torch.bfloat16 if has_bf16 else torch.float16
    _diag(f"bf16 native: {has_bf16}  →  compute_dtype={compute_dtype}")

    # bitsandbytes probe
    _bnb_ok = False
    _BitsAndBytesConfig = None
    try:
        import bitsandbytes as _bnb_mod
        from transformers import BitsAndBytesConfig as _BnBCfg
        _BitsAndBytesConfig = _BnBCfg
        _bnb_ok = True
        _diag(f"bitsandbytes: OK  v{getattr(_bnb_mod, '__version__', '?')}")
    except Exception as _bnb_exc:
        _diag(f"bitsandbytes: FAILED  {type(_bnb_exc).__name__}: {_bnb_exc}")

    if not _bnb_ok:
        raise RuntimeError(
            "bitsandbytes is required to load this 60GB model on Kaggle T4×2 (32GB GPU, 29GB RAM).\n"
            "Fix: in a notebook cell run:  !pip install -q bitsandbytes --upgrade\n"
            "then restart the kernel and re-run from Cell 1."
        )

    _diag(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ── Free memory before the heavy model load ────────────────────────────────
    _gc.collect()
    torch.cuda.empty_cache()
    for _gi in range(torch.cuda.device_count()):
        _fr, _ = torch.cuda.mem_get_info(_gi)
        _diag(f"GPU {_gi} free after GC: {_fr/1024**3:.1f}GB")

    # GPU-only max_memory: 4-bit NF4 quantized 30B ≈ 17GB → must fit entirely on GPU.
    # bitsandbytes 4-bit validator rejects any CPU/disk modules, so budget must be
    # large enough (≥17GB total) to keep all quantized layers on GPU.
    # 80% × 14.6GB × 2 = ~23.4GB total budget (17GB model + ~6GB headroom).
    # After PEFT validation pass, gc+empty_cache reclaims ~4GB for inference.
    _n_gpus = torch.cuda.device_count()
    _pct = 0.80
    _gpu_budget = {
        i: f"{int(torch.cuda.get_device_properties(i).total_memory * _pct / 1024**3)}GiB"
        for i in range(_n_gpus)
    }
    _diag(f"GPU budget ({int(_pct*100)}%): {_gpu_budget}")

    # Pre-flight: 4-bit NF4 Nemotron-30B requires ≥17GB total GPU VRAM.
    # bitsandbytes rejects any CPU/disk offloading for quantized modules.
    # P100 (1×16GB) is insufficient; T4×2 (2×14.6GB = 29.2GB) is required.
    _total_vram_gb = sum(
        torch.cuda.get_device_properties(i).total_memory for i in range(_n_gpus)
    ) / 1024**3
    _diag(f"Total GPU VRAM: {_total_vram_gb:.1f}GB across {_n_gpus} GPU(s)")
    if _total_vram_gb < 16.5:
        raise RuntimeError(
            f"Insufficient GPU VRAM: {_total_vram_gb:.1f}GB across {_n_gpus} GPU(s). "
            "4-bit NF4 Nemotron-30B requires ≥17GB total GPU VRAM. "
            "Please use Kaggle T4×2 (29.2GB). "
            "Select 'GPU T4 x2' in Notebook Settings → Accelerator."
        )

    # ── 4-bit NF4 only — no 8-bit fallback ────────────────────────────────────
    # 8-bit (30GB) plus failed-4-bit garbage easily exceeds 29GB RAM and OOMs.
    # 4-bit (15GB) fits comfortably on GPU with headroom for KV-cache.
    _diag("Loading base model (4-bit NF4) ...")
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=_BitsAndBytesConfig(
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
        for _gi in range(torch.cuda.device_count()):
            _fr, _ = torch.cuda.mem_get_info(_gi)
            _diag(f"GPU {_gi} free after model load: {_fr/1024**3:.1f}GB")
        _diag("4-bit NF4 load: OK")
    except Exception as _e4:
        _diag(f"4-bit NF4 FAILED: {type(_e4).__name__}: {_e4}")
        try:
            del model
        except Exception:
            pass
        _gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"FATAL: 4-bit NF4 quantization failed: {type(_e4).__name__}: {_e4}\n"
            f"Diagnostics saved to: {_diag_path}\n"
            "Common fixes:\n"
            "  !pip install -q bitsandbytes --upgrade  (restart kernel after)\n"
            "  The Mamba MoE architecture may need a newer transformers or bnb version."
        ) from _e4

    # Monkey-patch PEFT._update_offload to skip Mamba MoE keys that are absent
    # from named_modules() in the stub (experts.100.down_proj etc.).
    try:
        from peft import peft_model as _peft_mod
        _orig_upd = _peft_mod.PeftModel._update_offload

        def _safe_update_offload(self, offload_index, adapters_weights):
            if not offload_index:
                return offload_index
            try:
                return _orig_upd(self, offload_index, adapters_weights)
            except KeyError as _ke:
                print(f"[peft_patch] _update_offload KeyError skipped "
                      f"(Mamba MoE key absent from named_modules): {_ke}")
                return offload_index

        _peft_mod.PeftModel._update_offload = _safe_update_offload
        print("[peft_patch] Patched _update_offload")
    except Exception as _pe:
        print(f"[peft_patch] Warning: {_pe}")

    # Set eval mode on the base model BEFORE the PEFT validation forward pass
    # so dropout is disabled during any internal pass inside from_pretrained().
    model.eval()

    # With quantization the model fits on GPU; no disk offload needed for PEFT.
    _diag(f"Loading adapter from {adapter_path}")
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as _peft_exc:
        _diag(f"PEFT load FAILED: {type(_peft_exc).__name__}: {_peft_exc}")
        try:
            del model
        except Exception:
            pass
        _gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"FATAL: PeftModel.from_pretrained() failed: {_peft_exc}\n"
            f"Check adapter path: {adapter_path}"
        ) from _peft_exc
    model.eval()

    # PEFT's adapter loading runs a validation forward pass that caches large
    # temporary tensors in PyTorch's CUDA allocator.  Flush them now so they
    # don't count against inference headroom.
    _gc.collect()
    torch.cuda.empty_cache()
    for _gi in range(torch.cuda.device_count()):
        _fr, _tot = torch.cuda.mem_get_info(_gi)
        _diag(f"GPU {_gi} free after PEFT + empty_cache: {_fr/1024**3:.1f}GB / {_tot/1024**3:.1f}GB")

    # model.device is undefined when layers span multiple devices; find first CUDA param.
    def _infer_input_device(m: torch.nn.Module) -> torch.device:
        for p in m.parameters():
            if p.device.type == "cuda":
                return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_device = _infer_input_device(model)
    _diag(f"Input device: {input_device}")

    # Locate NemotronH's custom hybrid cache class so each generate() call can
    # store SSM states incrementally (O(1) per step) rather than reprocessing
    # the full growing sequence each step (O(N) → OOM on 2048-token generation).
    #
    # The class is referenced by modeling_nemotron_h.py but may NOT be a top-level
    # attribute of that module if it is: (a) lazy-imported inside a function, or
    # (b) defined in a sibling module.  Use three escalating strategies.
    import sys as _sys
    import inspect as _inspect
    _nemh_cache_cls = None
    _model_cfg_for_cache = None
    try:
        # Identify the inner model class and its source module for diagnostics.
        _inner_model = getattr(getattr(model, "base_model", model), "model", model)
        _model_cls_name = type(_inner_model).__name__
        _model_module_name = type(_inner_model).__module__
        _diag(f"[cache] model class: {_model_cls_name}  module: {_model_module_name}")

        # Diagnostic: log every nemotron/transformers_modules name in sys.modules.
        _nemotron_mods = [m for m in _sys.modules if "nemotron" in m.lower()]
        _custom_mods = [m for m in _sys.modules if "transformers_modules" in m]
        _diag(f"[cache] nemotron in sys.modules: {_nemotron_mods}")
        _diag(f"[cache] transformers_modules in sys.modules: {_custom_mods}")

        # Strategy 1: broad scan of ALL loaded modules.
        # Narrow filter ("modeling_nemotron_h" substring) previously missed the
        # class when it lives in a sibling module or is only a local variable.
        for _mn, _mod in list(_sys.modules.items()):
            if _mod is None:
                continue
            try:
                _cls = getattr(_mod, "NemotronHHybridDynamicCache", None)
            except Exception:
                continue
            if _cls is not None and isinstance(_cls, type):
                _nemh_cache_cls = _cls
                _diag(f"[cache] Strategy-1 found NemotronHHybridDynamicCache in: {_mn}")
                break

        # Strategy 2: inspect source file directory for sibling .py modules and
        # try importing each one (handles case where class is in a separate file).
        if _nemh_cache_cls is None:
            _mod = _sys.modules.get(_model_module_name)
            if _mod is not None:
                try:
                    import os as _os2, importlib as _il2, importlib.util as _ilu
                    _src_file = _inspect.getfile(_mod)
                    _src_dir = _os2.path.dirname(_src_file)
                    _diag(f"[cache] source dir: {_src_dir}")

                    # Log NemotronHHybridDynamicCache refs in the main source file
                    # and all top-level class definitions — useful for diagnosis.
                    with open(_src_file) as _sf:
                        _src_text = _sf.read()
                    _cache_refs = [l.strip() for l in _src_text.splitlines()
                                   if "NemotronHHybridDynamicCache" in l
                                   or ("class" in l and "Cache" in l)]
                    _diag(f"[cache] cache refs in source: {_cache_refs[:10]}")

                    # List sibling .py files and log cache-like attrs from each.
                    _pkg_name = _model_module_name.rsplit(".", 1)[0] if "." in _model_module_name else ""
                    for _pyf in sorted(_os2.listdir(_src_dir)):
                        if not _pyf.endswith(".py"):
                            continue
                        _pypath = _os2.path.join(_src_dir, _pyf)
                        # Try package-relative import first (handles relative imports).
                        _probe_mod = None
                        if _pkg_name:
                            _rel = _pyf[:-3]
                            try:
                                _probe_mod = _il2.import_module(f".{_rel}", package=_pkg_name)
                            except Exception:
                                pass
                        # Fall back to raw file load (fails on relative imports but still
                        # lets us inspect top-level class definitions).
                        if _probe_mod is None:
                            try:
                                _spec = _ilu.spec_from_file_location(f"_nemh_probe_{_pyf[:-3]}", _pypath)
                                _probe_mod = _ilu.module_from_spec(_spec)
                                _spec.loader.exec_module(_probe_mod)
                            except Exception as _pfe:
                                _diag(f"[cache]   probe {_pyf}: {type(_pfe).__name__}: {str(_pfe)[:80]}")
                                continue
                        _cls = getattr(_probe_mod, "NemotronHHybridDynamicCache", None)
                        if _cls is not None and isinstance(_cls, type):
                            _nemh_cache_cls = _cls
                            _diag(f"[cache] Strategy-2 found NemotronHHybridDynamicCache in: {_pyf}")
                            break
                        _c_attrs = [a for a in dir(_probe_mod) if "cache" in a.lower()]
                        _diag(f"[cache]   {_pyf} cache attrs: {_c_attrs}")
                except Exception as _s2e:
                    _diag(f"[cache] Strategy-2 error: {type(_s2e).__name__}: {_s2e}")

        if _nemh_cache_cls is not None:
            _model_cfg_for_cache = (
                getattr(model, "config", None)
                or getattr(getattr(model, "base_model", None), "config", None)
            )
            if _model_cfg_for_cache is None:
                try:
                    _model_cfg_for_cache = model.base_model.model.config
                except Exception:
                    pass
            _diag("[cache] NemotronHHybridDynamicCache found — SSM states will be cached")
        else:
            _diag("[cache] NemotronHHybridDynamicCache not found after all strategies — OOM risk")
    except Exception as _ce:
        _diag(f"[cache] lookup failed: {type(_ce).__name__}: {_ce}")


    records: List[Dict[str, Any]] = []
    _MAX_INPUT_TOKENS = 768  # cap prompt length to keep SSM intermediate tensors small
    _truncation_warned = False  # fire once on first truncation event (not just idx==0)

    # Write JSONL incrementally so partial results survive a kernel OOM restart.
    _jsonl_path = output_dir / "golden_validation_predictions.jsonl"

    with _jsonl_path.open("w", encoding="utf-8") as _jsonl_fh:
      for idx, record in enumerate(problems):
        pid = get_problem_id(record, idx)
        cat_info = category_map.get(pid, {"category": "other", "subcategory": "unknown"})
        question = str(record.get("question", record.get("prompt", ""))).strip()
        gold_answer = str(record.get("answer", record.get("target", ""))).strip()

        prompt = build_prompt(record)
        inputs = tokenizer(prompt, return_tensors="pt").to(input_device)

        # Truncate long prompts from the RIGHT to keep the question at the start.
        # Truncating from the left (v[:, -N:]) would discard the question and keep
        # only the format-footer boilerplate — wrong model input.
        _was_truncated = False
        if inputs["input_ids"].shape[1] > _MAX_INPUT_TOKENS:
            inputs = {k: v[:, :_MAX_INPUT_TOKENS] for k, v in inputs.items()}
            _was_truncated = True
            if not _truncation_warned:
                _truncation_warned = True
                _diag(f"[warn] prompt truncated to {_MAX_INPUT_TOKENS} tokens "
                      f"(first occurrence at idx={idx}, pid={pid})")

        # Pass only input_ids + attention_mask to generate() — extra tokenizer keys
        # (token_type_ids, position_ids, etc.) cause TypeError in some model variants.
        _gen_inputs: Dict[str, Any] = {"input_ids": inputs["input_ids"]}
        if "attention_mask" in inputs:
            _gen_inputs["attention_mask"] = inputs["attention_mask"]
        _input_len = _gen_inputs["input_ids"].shape[1]

        # Initialize a fresh NemotronHHybridDynamicCache per problem so generate()
        # can store SSM states incrementally (O(1) per step) rather than reprocessing
        # the full growing sequence at every step (O(N) → OOM on long outputs).
        # Without this, the warning "NemotronH requires an initialized
        # NemotronHHybridDynamicCache" fires and torch_forward runs over the full
        # sequence each step until GPU 1 exhausts memory.
        _past_kv = None
        if _nemh_cache_cls is not None and _model_cfg_for_cache is not None:
            try:
                _past_kv = _nemh_cache_cls(
                    config=_model_cfg_for_cache,
                    batch_size=1,
                    dtype=compute_dtype,
                )
            except Exception as _cache_init_err:
                _diag(f"[{idx+1:4d}/{len(problems)}] {pid}: cache init failed "
                      f"({type(_cache_init_err).__name__}: {_cache_init_err}) — "
                      "falling back to no cache (OOM risk)")
                _past_kv = None

        # Free fragmented CUDA allocations before each generation step.
        # With expandable_segments=True this reclaims reserved-unallocated blocks
        # so the 768MB+ KV-cache allocation on GPU 1 succeeds every iteration.
        torch.cuda.empty_cache()

        try:
            t0 = time.time()
            with torch.no_grad():
                _generate_kwargs: Dict[str, Any] = dict(
                    **_gen_inputs,
                    max_new_tokens=GOLDEN_GENERATION_CONFIG["max_new_tokens"],
                    do_sample=GOLDEN_GENERATION_CONFIG["do_sample"],
                    temperature=None,  # greedy when do_sample=False
                    repetition_penalty=GOLDEN_GENERATION_CONFIG["repetition_penalty"],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
                if _past_kv is not None:
                    _generate_kwargs["past_key_values"] = _past_kv
                output = model.generate(**_generate_kwargs)
            elapsed = time.time() - t0
        except Exception as _gen_exc:
            _diag(f"[{idx+1:4d}/{len(problems)}] {pid}: GENERATE_ERROR "
                  f"{type(_gen_exc).__name__}: {_gen_exc}")
            _gc.collect()
            torch.cuda.empty_cache()
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
                "elapsed_seconds": 0.0, "seed": seed,
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

        # Determine finish reason by checking the last generated token against EOS.
        # The `< max_new_tokens` check misclassifies EOS generated at exactly the limit.
        _last_tok = int(generated_ids[-1]) if n_tokens > 0 else None
        finish_reason = (
            "eos" if _last_tok == tokenizer.eos_token_id else "length"
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
            "seed": seed,
            "generation_config": dict(GOLDEN_GENERATION_CONFIG),
        }
        records.append(entry)
        _jsonl_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _jsonl_fh.flush()

        status = "OK" if is_correct else ("PARSE_FAIL" if not parse_success else "WRONG")
        _trunc_flag = " [TRUNC]" if _was_truncated else ""
        print(f"[{idx+1:4d}/{len(problems)}] {pid}: {status}  tokens={n_tokens}{_trunc_flag}")

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
        "--max-problems",
        type=int,
        default=0,
        help="Limit inference to first N problems (0 = all). Useful for quick smoke tests.",
    )
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

    if args.max_problems and args.max_problems > 0:
        problems = problems[: args.max_problems]
        print(f"[max-problems] Capped at {args.max_problems}")

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
