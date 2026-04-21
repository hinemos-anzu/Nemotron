"""
kaggle/original-nemotron-asymmetric-svd-26041602.py

Baseline execution script: original Nemotron asymmetric-SVD submission path.
This is the Kaggle source-of-truth execution path referenced by:
  - TICKET_S1_6A_kaggle_runtime_execution_enablement_for_baseline_path.md
  - docs/specs/design_spec_from_research_v1.md (protected baseline assets)

PROTECTED ASSET — do not change strategy logic, conversion flow, SVD surgery,
key rename / merge logic, expert unfuse logic, or submission.zip generation.

PLACEHOLDER STATUS:
  This file is a structural placeholder.  The Kaggle execution role must replace
  the body below with the actual notebook/script content before execution.
  The interface contract (REQUIRED_INPUTS, EXPECTED_ARTIFACTS, stage log calls)
  must be preserved when the real implementation is inserted.

REQUIRED ENVIRONMENT:
  - Python 3.10+
  - torch, transformers, peft, accelerate, safetensors
  - /kaggle/input/  (mounted dataset root)
  - /kaggle/working/ (output root)
  - Adapter model weights at Kaggle model path (see ADAPTER_INPUTS)
  - Base model via kagglehub.model_download
"""

import csv
import json
import math
import os
import re
import sys
import traceback
from decimal import Decimal, getcontext
from fractions import Fraction
from math import gcd, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── interface contract ───────────────────────────────────────────────────────

REQUIRED_INPUTS = [
    # Standard Kaggle environment roots — always present
    "/kaggle/input",
    "/kaggle/working",
]

# Model / adapter assets required when real body is inserted.
# Paths confirmed with Kaggle execution role during TICKET_S1_6A.
# Checked by the real implementation, NOT by this placeholder.
# Kaggle model mount path: /kaggle/input/models/{user}/{slug}/{framework}/{variation}/{version}/
ADAPTER_PATH = "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20"
BASE_MODEL_ID  = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
COMP_DATA_PATH = "/kaggle/input/nvidia-nemotron-3-reasoning-challenge"

ADAPTER_INPUTS = [
    ADAPTER_PATH,
]

EXPECTED_ARTIFACTS = [
    "/kaggle/working/submission.zip",
    "/kaggle/working/run_complete.flag",
    "/kaggle/working/predictions.jsonl",
]

# ─── answer key solvers ───────────────────────────────────────────────────────

def _num(x: float) -> str:
    """Format float: integer if whole, otherwise up to 4 significant decimals."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return str(round(x, 4)).rstrip("0").rstrip(".")


def _fraction_to_decimal_str(n: int, d: int) -> str:
    """Return terminating decimal or 'int.nonrep(rep)' repeating notation."""
    g = gcd(abs(n), abs(d))
    n, d = n // g, d // g
    if d < 0:
        n, d = -n, -d
    integer_part = n // d
    remainder = abs(n) % d
    if remainder == 0:
        return str(integer_part)
    test = d
    while test % 2 == 0:
        test //= 2
    while test % 5 == 0:
        test //= 5
    if test == 1:
        getcontext().prec = 30
        result = str(Decimal(n) / Decimal(d))
        if "." in result:
            result = result.rstrip("0").rstrip(".")
        return result
    # repeating decimal
    seen: Dict[int, int] = {}
    digits: List[int] = []
    rem = remainder
    pos = 0
    while rem != 0 and rem not in seen:
        seen[rem] = pos
        rem *= 10
        digits.append(rem // d)
        rem %= d
        pos += 1
    prefix = str(integer_part) + "." if integer_part != 0 else "0."
    if rem == 0:
        return prefix + "".join(map(str, digits))
    start = seen[rem]
    non_rep = "".join(map(str, digits[:start]))
    rep = "".join(map(str, digits[start:]))
    return f"{prefix}{non_rep}({rep})"


def _solve_numeral(problem: str) -> str:
    p = problem.strip()
    m = re.search(r"Convert (\d+) \(decimal\) to hexadecimal", p)
    if m:
        return hex(int(m.group(1)))[2:].upper()
    m = re.search(r"decimal value of the binary number ([01]+)", p)
    if m:
        return str(int(m.group(1), 2))
    m = re.search(r"Convert (\d+) \(base 10\) to binary", p)
    if m:
        return bin(int(m.group(1)))[2:]
    m = re.search(r"What is (\d+) mod (\d+)", p)
    if m:
        return str(int(m.group(1)) % int(m.group(2)))
    m = re.search(r"Express (\d+)/(\d+) as a decimal", p)
    if m:
        return _fraction_to_decimal_str(int(m.group(1)), int(m.group(2)))
    m = re.search(r"What is 2\^(\d+)", p)
    if m:
        return str(2 ** int(m.group(1)))
    m = re.search(r"GCD of (\d+) and (\d+)", p)
    if m:
        return str(gcd(int(m.group(1)), int(m.group(2))))
    m = re.search(r"Convert (\d+) \(decimal\) to octal", p)
    if m:
        return oct(int(m.group(1)))[2:]
    return "CANNOT_COMPUTE"


def _solve_unit_conversion(problem: str) -> str:
    p = problem.strip()
    m = re.search(r"Express (\d+) seconds as minutes and seconds", p)
    if m:
        mins, secs = divmod(int(m.group(1)), 60)
        return f"{mins} min {secs} sec"
    m = re.search(r"Convert ([\d.]+) feet to meters.*Round to (\d+)", p)
    if m:
        return str(round(float(m.group(1)) * 0.3048, int(m.group(2))))
    m = re.search(r"Convert ([\d.]+) miles to kilometers.*Round to (\d+)", p)
    if m:
        return str(round(float(m.group(1)) * 1.60934, int(m.group(2))))
    m = re.search(r"Convert ([\d.]+) km/h to m/s.*Round to (\d+)", p)
    if m:
        return str(round(float(m.group(1)) / 3.6, int(m.group(2))))
    m = re.search(r"Convert ([\d.]+) liters? to milliliters?", p)
    if m:
        v = float(m.group(1)) * 1000
        return str(int(v) if v == int(v) else v)
    m = re.search(r"Convert ([+-]?[\d.]+)\s*°C to Kelvin", p)
    if m:
        k = float(m.group(1)) + 273.15
        return _num(k)
    m = re.search(r"Convert ([\d.]+) kg to grams?", p)
    if m:
        g = float(m.group(1)) * 1000
        return str(int(g) if g == int(g) else g)
    return "CANNOT_COMPUTE"


def _solve_gravity(problem: str) -> str:
    p = problem.strip()
    G = 9.8
    m = re.search(r"height h = ([\d.]+) m.*speed just before impact", p, re.I)
    if m:
        return str(round(sqrt(2 * G * float(m.group(1))), 2))
    m = re.search(r"potential energy of a ([\d.]+) kg object at height ([\d.]+) m", p)
    if m:
        pe = float(m.group(1)) * G * float(m.group(2))
        return _num(pe)
    m = re.search(r"thrown upward at ([\d.]+) m/s.*How high", p, re.I)
    if m:
        v = float(m.group(1))
        return str(round(v**2 / (2 * G), 2))
    m = re.search(r"([\d.]+) kg object is dropped from ([\d.]+) m.*kinetic energy", p)
    if m:
        ke = float(m.group(1)) * G * float(m.group(2))
        return _num(ke)
    m = re.search(r"dropped from h = ([\d.]+) m.*How long", p, re.I)
    if m:
        return str(round(sqrt(2 * float(m.group(1)) / G), 2))
    return "CANNOT_COMPUTE"


def _solve_cipher(problem: str) -> str:
    p = problem.strip()

    def shift(text: str, n: int) -> str:
        return "".join(
            chr((ord(c) - ord("A") + n) % 26 + ord("A")) if c.isalpha() else c
            for c in text.upper()
        )

    # "The string 'JBEYQ' was Caesar-encoded with shift 13. Reverse..."
    m = re.search(r"'([A-Z]+)'\s+was Caesar-encoded with shift (\d+)", p, re.I)
    if m:
        return shift(m.group(1), -int(m.group(2)))
    m = re.search(r"ROT13 to '([A-Z]+)'", p, re.I)
    if m:
        return shift(m.group(1), 13)
    m = re.search(r"Decode '([A-Z]+)' using a Caesar cipher with shift (\d+)", p, re.I)
    if m:
        return shift(m.group(1), -int(m.group(2)))
    m = re.search(r"Encode '([A-Z]+)' using a Caesar cipher with shift (\d+)", p, re.I)
    if m:
        return shift(m.group(1), int(m.group(2)))
    return "CANNOT_COMPUTE"


# ─── optional: attach to active harness stage logger ─────────────────────────
# If run via run_baseline_with_debug.py, a StageLogger is available.

def _coeff(s: str) -> float:
    """Parse coefficient string like '1', '-1', '+1', '' → float."""
    s = s.strip()
    if s in ("", "+"):
        return 1.0
    if s == "-":
        return -1.0
    return float(s)


def _solve_equation_family(problem: str) -> str:
    """Covers equation, hard, low_logprob_suspect categories."""
    p = problem.strip()

    # log₂(N) = x
    m = re.search(r"log[₂2]\((\d+)\)\s*=\s*x", p)
    if m:
        v = math.log2(int(m.group(1)))
        return _num(v)

    # Expand (x - P)(x - Q)
    m = re.search(r"Expand \(x\s*-\s*([+-]?\d+)\)\(x\s*-\s*([+-]?\d+)\)", p)
    if m:
        r1, r2 = int(m.group(1)), int(m.group(2))
        s, pr = r1 + r2, r1 * r2
        res = "x²"
        res += f" - {s}x" if s > 0 else (f" + {-s}x" if s < 0 else "")
        res += f" + {pr}" if pr > 0 else (f" - {-pr}" if pr < 0 else "")
        return res

    # Factor completely: x² + Bx + C
    m = re.search(r"Factor completely:\s*x²\s*\+\s*([+-]?\d+)x\s*\+\s*([+-]?\d+)\.", p)
    if m:
        B, C = int(m.group(1)), int(m.group(2))
        lim = abs(C) + abs(B) + 2
        for r1 in range(-lim, lim + 1):
            r2 = -B - r1
            if r1 * r2 == C:
                def _f(r: int) -> str:
                    return f"(x - {r})" if r > 0 else (f"(x + {-r})" if r < 0 else "(x)")
                return _f(r1) + _f(r2)
        return "CANNOT_FACTOR"

    # Simplify: (Ax² + Bx) / x
    m = re.search(r"\(([+-]?\s*\d*\.?\d*)x²\s*\+\s*([+-]?\s*\d*\.?\d*)x\)\s*/\s*x", p)
    if m:
        a = _coeff(m.group(1).replace(" ", ""))
        b = _coeff(m.group(2).replace(" ", ""))
        ai = int(a) if a == int(a) else a
        bi = int(b) if b == int(b) else b
        return f"{ai}x + {bi}" if bi >= 0 else f"{ai}x - {-bi}"

    # Solve quadratic: Ax² + Bx + C = 0
    m = re.search(
        r"Solve[^:]*:\s*([+-]?\s*\d*\.?\d+)x²\s*\+\s*([+-]?\s*\d*\.?\d+)x\s*\+\s*([+-]?\s*\d*\.?\d+)\s*=\s*0",
        p,
    )
    if m:
        a = _coeff(m.group(1).replace(" ", ""))
        b = _coeff(m.group(2).replace(" ", ""))
        c = float(m.group(3).replace(" ", ""))
        disc = b**2 - 4 * a * c
        if disc < -1e-9:
            return "no real roots"
        if abs(disc) < 1e-9:
            return _num(-b / (2 * a))
        x1 = (-b + sqrt(disc)) / (2 * a)
        x2 = (-b - sqrt(disc)) / (2 * a)
        lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)
        return f"{_num(lo)}, {_num(hi)}"

    # Evaluate f(x) = Ax² + Bx + C at x = N
    m = re.search(
        r"f\(x\)\s*=\s*([+-]?\s*\d*\.?\d+)x²\s*\+\s*([+-]?\s*\d*\.?\d+)x\s*\+\s*([+-]?\s*\d*\.?\d+)[,\s]+evaluate f\(([+-]?\d+)\)",
        p,
    )
    if m:
        a = _coeff(m.group(1).replace(" ", ""))
        b = _coeff(m.group(2).replace(" ", ""))
        c = float(m.group(3).replace(" ", ""))
        n = float(m.group(4))
        return _num(a * n**2 + b * n + c)

    # Slope of line through (x1,y1) and (x2,y2)
    m = re.search(r"\(([+-]?\d+),\s*([+-]?\d+)\).*\(([+-]?\d+),\s*([+-]?\d+)\)", p)
    if m:
        x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            return "undefined"
        f = Fraction(dy, dx)
        return str(int(f)) if f.denominator == 1 else str(f)

    # Linear equation Ax + B = C
    m = re.search(
        r"([+-]?\s*\d*\.?\d+)x\s*\+\s*([+-]?\s*\d+\.?\d*)\s*=\s*([+-]?\s*\d+\.?\d*)",
        p,
    )
    if m:
        a = _coeff(m.group(1).replace(" ", ""))
        b = float(m.group(2).replace(" ", ""))
        c = float(m.group(3).replace(" ", ""))
        if abs(a) < 1e-12:
            return "CANNOT_COMPUTE"
        return _num((c - b) / a)

    return "CANNOT_COMPUTE"


def _apply_transform(src_s: str, dst_s: str, tgt_s: str) -> str:
    b = len(src_s)
    src, dst, tgt, mask = int(src_s, 2), int(dst_s, 2), int(tgt_s, 2), (1 << b) - 1

    def rev(n: int) -> int:
        return int(format(n, f"0{b}b")[::-1], 2)

    if dst == (~src) & mask:
        return format((~tgt) & mask, f"0{b}b")
    if dst == src:
        return tgt_s
    if dst == rev(src):
        return format(rev(tgt), f"0{b}b")
    for k in range(1, b):
        if dst == ((src >> k) | (src << (b - k))) & mask:
            return format(((tgt >> k) | (tgt << (b - k))) & mask, f"0{b}b")
        if dst == ((src << k) | (src >> (b - k))) & mask:
            return format(((tgt << k) | (tgt >> (b - k))) & mask, f"0{b}b")
    return "CANNOT_COMPUTE"


def _solve_bit_family(problem: str) -> str:
    """Covers bit_manipulation and conversion_sensitive categories."""
    p = problem.strip()

    # Identify transformation rule and apply
    m = re.search(
        r"Identify the transformation rule:\s*([01]+)\s*(?:→|->)\s*([01]+)\..*?Apply.*?to\s*([01]+)\.",
        p, re.I | re.S,
    )
    if m:
        return _apply_transform(m.group(1), m.group(2), m.group(3))

    # Circular rotation
    m = re.search(
        r"Apply (right|left) circular rotation by (\d+) bit.*?(\d+)-bit binary number ([01]+)",
        p, re.I,
    )
    if m:
        direction, k, bits, val_str = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        val = int(val_str, 2)
        k %= bits
        mask = (1 << bits) - 1
        if direction.lower() == "right":
            result = ((val >> k) | (val << (bits - k))) & mask
        else:
            result = ((val << k) | (val >> (bits - k))) & mask
        return format(result, f"0{bits}b")

    # XOR / AND / OR
    for op_name, op in (("XOR", lambda a, b: a ^ b), ("AND", lambda a, b: a & b), ("OR", lambda a, b: a | b)):
        m = re.search(rf"What is (\d+) {op_name} (\d+)", p, re.I)
        if m:
            return str(op(int(m.group(1)), int(m.group(2))))

    # Shift
    m = re.search(r"(\d+) left-shifted by (\d+) bits?", p, re.I)
    if m:
        return str(int(m.group(1)) << int(m.group(2)))
    m = re.search(r"(\d+) right-shifted by (\d+) bits?", p, re.I)
    if m:
        return str(int(m.group(1)) >> int(m.group(2)))

    # Is bit K set?
    m = re.search(r"Is bit (\d+) set in (\d+)", p, re.I)
    if m:
        return "YES" if (int(m.group(2)) >> int(m.group(1))) & 1 else "NO"

    # Clear bit
    m = re.search(r"Clear bit (\d+) of the (\d+)-bit binary number ([01]+)", p, re.I)
    if m:
        k, bits, val = int(m.group(1)), int(m.group(2)), int(m.group(3), 2)
        return format(val & ~(1 << k) & ((1 << bits) - 1), f"0{bits}b")

    # Set bit
    m = re.search(r"Set bit (\d+) of the (\d+)-bit binary number ([01]+)", p, re.I)
    if m:
        k, bits, val = int(m.group(1)), int(m.group(2)), int(m.group(3), 2)
        return format((val | (1 << k)) & ((1 << bits) - 1), f"0{bits}b")

    # Toggle bit
    m = re.search(r"Toggle bit (\d+) of the (\d+)-bit binary number ([01]+)", p, re.I)
    if m:
        k, bits, val = int(m.group(1)), int(m.group(2)), int(m.group(3), 2)
        return format((val ^ (1 << k)) & ((1 << bits) - 1), f"0{bits}b")

    # Popcount
    m = re.search(r"popcount.*binary representation of (\d+)", p, re.I)
    if m:
        return str(bin(int(m.group(1))).count("1"))

    # Bitwise NOT as N-bit unsigned integer
    m = re.search(r"bitwise NOT of (\d+) as an? (\d+)-bit unsigned", p, re.I)
    if m:
        val, bits = int(m.group(1)), int(m.group(2))
        return str((~val) & ((1 << bits) - 1))

    return "CANNOT_COMPUTE"


def compute_expected_answer(sample: Dict) -> str:
    cat = sample["category"]
    prob = sample["problem"]
    try:
        if cat == "numeral":
            return _solve_numeral(prob)
        if cat == "unit_conversion":
            return _solve_unit_conversion(prob)
        if cat == "gravity":
            return _solve_gravity(prob)
        if cat == "cipher":
            return _solve_cipher(prob)
        if cat in ("equation", "hard", "low_logprob_suspect"):
            return _solve_equation_family(prob)
        if cat in ("bit_manipulation", "conversion_sensitive"):
            return _solve_bit_family(prob)
    except Exception as e:
        return f"SOLVER_ERROR: {e}"
    return "CANNOT_COMPUTE"


def _get_stage_logger():
    """Return active harness stage logger or a no-op logger."""
    try:
        # Try to load the stage log path set by the harness
        stage_log_path = os.environ.get("KAGGLE_HARNESS_STAGE_LOG")
        if stage_log_path:
            sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
            from debug_harness import StageLogger, _now
            import json as _json

            class _DirectLogger:
                def __init__(self, path):
                    self._path = path

                def log(self, stage, message="", status="ok", extra=None):
                    event = {
                        "timestamp": _now(),
                        "stage": stage,
                        "status": status,
                        "message": message,
                        "extra": extra or {},
                    }
                    try:
                        with open(self._path, "a", encoding="utf-8") as f:
                            f.write(_json.dumps(event) + "\n")
                    except Exception:
                        pass

            return _DirectLogger(stage_log_path)
    except Exception:
        pass

    class _NoopLogger:
        def log(self, *a, **kw): pass
    return _NoopLogger()


# ─── model loading & inference ───────────────────────────────────────────────

def _try_load_model() -> Tuple[Optional[object], Optional[object]]:
    adapter_dir = Path(ADAPTER_PATH)
    if not adapter_dir.exists():
        print(f"[baseline] Adapter not found at {ADAPTER_PATH} — ANSWER_KEY_ONLY mode", flush=True)
        return None, None
    try:
        import kagglehub
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        print(f"[baseline] Downloading base model: {BASE_MODEL_ID}", flush=True)
        model_path = kagglehub.model_download(BASE_MODEL_ID)
        print(f"[baseline] Base model path: {model_path}", flush=True)

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        mdl = PeftModel.from_pretrained(mdl, str(adapter_dir))
        mdl.eval()
        print("[baseline] Model loaded OK", flush=True)
        return mdl, tok
    except Exception as exc:
        print(f"[baseline] Model load failed: {type(exc).__name__}: {exc}", flush=True)
        return None, None


def _run_inference(model, tokenizer, problem: str) -> str:
    """Run inference using Nemotron chat format."""
    import torch
    # Nemotron-3 conversation format
    prompt = (
        "<extra_id_0>System\n"
        "Answer the following question precisely. Output only the final answer.\n"
        "<extra_id_1>User\n"
        f"{problem}\n"
        "<extra_id_1>Assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def _extract_answer(raw: str, category: str) -> Tuple[str, bool, bool]:
    """Returns (extracted, format_failure, extraction_failure)."""
    if not raw or not raw.strip():
        return "", True, True
    text = raw.strip()
    # strip common boilerplate
    text = re.sub(r"(?i)(the answer is|therefore[,.]?|so[,.]?|=\s*|answer\s*:\s*)", "", text).strip(" .:")
    if not text:
        return raw.strip()[:60], False, True
    # take last non-empty line (model often restates at the end)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    answer = lines[-1] if lines else text
    format_fail = False
    extract_fail = len(answer) > 120
    return answer[:120], format_fail, extract_fail


def _answers_match(predicted: str, expected: str) -> bool:
    if not predicted or not expected or expected.startswith("CANNOT"):
        return False
    p, e = predicted.strip().lower(), expected.strip().lower()
    if p == e:
        return True
    # numeric
    try:
        return abs(float(p) - float(e)) < max(1e-3, 1e-4 * abs(float(e)))
    except ValueError:
        pass
    # comma-separated numbers (quadratic roots)
    if "," in e and "," in p:
        try:
            ps = sorted(float(x) for x in p.split(","))
            es = sorted(float(x) for x in e.split(","))
            if len(ps) == len(es):
                return all(abs(a - b) < 1e-3 for a, b in zip(ps, es))
        except ValueError:
            pass
    # normalize whitespace for expressions like "4x + 7"
    return re.sub(r"\s+", "", p) == re.sub(r"\s+", "", e)


# ─── eval data loading ───────────────────────────────────────────────────────

def _load_eval_csv(eval_set: str, repo_root: Path) -> List[Dict]:
    csv_path_env = os.environ.get("NEMOTRON_EVAL_CSV", "")
    csv_path = Path(csv_path_env) if csv_path_env else repo_root / "data" / "eval" / f"{eval_set}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Eval CSV not found: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[baseline] Loaded {len(rows)} samples from {csv_path.name}", flush=True)
    return rows


# ─── result writing ───────────────────────────────────────────────────────────

def _write_per_sample_csv(results: List[Dict], repo_root: Path, eval_set: str) -> Path:
    out = repo_root / "data" / "eval" / f"baseline_measured_results_{eval_set}_kaggle.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "split_name", "category", "difficulty_bucket",
        "computed_expected_answer", "baseline_prediction_status",
        "model_raw_output", "model_extracted_answer", "baseline_correctness",
        "format_failure_flag", "extraction_failure_flag", "runtime_status", "notes",
    ]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"[baseline] Wrote per-sample results: {out.name} ({len(results)} rows)", flush=True)
    return out


def _write_category_summary(results: List[Dict], repo_root: Path, eval_set: str) -> Path:
    from collections import defaultdict
    cats: Dict[str, Dict] = defaultdict(lambda: dict(
        sample_count=0, measured_pass_count=0, measured_fail_count=0,
        measured_error_count=0, format_failure_count=0, extraction_failure_count=0,
    ))
    split_name = results[0]["split_name"] if results else eval_set
    for r in results:
        c = cats[r["category"]]
        c["sample_count"] += 1
        correctness = r.get("baseline_correctness", "N/A")
        status = r.get("runtime_status", "")
        if status not in ("OK", "MODEL_NOT_AVAILABLE"):
            c["measured_error_count"] += 1
        elif correctness == "YES":
            c["measured_pass_count"] += 1
        elif correctness == "NO":
            c["measured_fail_count"] += 1
        else:
            c["measured_error_count"] += 1
        if r.get("format_failure_flag") == "YES":
            c["format_failure_count"] += 1
        if r.get("extraction_failure_flag") == "YES":
            c["extraction_failure_count"] += 1
    out = repo_root / "data" / "eval" / "baseline_measured_category_summary_v1_kaggle.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split_name", "category", "sample_count",
        "measured_pass_count", "measured_fail_count", "measured_error_count",
        "format_failure_count", "extraction_failure_count", "pass_rate_or_accuracy",
    ]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cat, c in sorted(cats.items()):
            total = c["measured_pass_count"] + c["measured_fail_count"]
            rate = round(c["measured_pass_count"] / total, 4) if total > 0 else "N/A"
            w.writerow({"split_name": split_name, "category": cat, **c, "pass_rate_or_accuracy": rate})
    print(f"[baseline] Wrote category summary: {out.name}", flush=True)
    return out


# ─── main execution ───────────────────────────────────────────────────────────

def main():
    sl = _get_stage_logger()
    working_dir = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
    repo_root = Path(os.environ.get("NEMOTRON_REPO_ROOT", Path(__file__).parent.parent))
    working_dir.mkdir(parents=True, exist_ok=True)
    eval_set = os.environ.get("NEMOTRON_EVAL_SET", "quick_gate_v1")

    # ── load eval data ────────────────────────────────────────────────
    sl.log("data_load_start", f"Loading eval set: {eval_set}")
    samples = _load_eval_csv(eval_set, repo_root)
    sl.log("data_load_end", f"Loaded {len(samples)} samples")

    # ── compute answer keys analytically ─────────────────────────────
    sl.log("answer_key_start", "Computing expected answers analytically")
    cannot_count = 0
    for s in samples:
        s["computed_expected_answer"] = compute_expected_answer(s)
        if s["computed_expected_answer"].startswith("CANNOT"):
            cannot_count += 1
    sl.log("answer_key_end", f"Done — {cannot_count}/{len(samples)} CANNOT_COMPUTE",
           extra={"cannot_count": cannot_count})

    # ── load model ────────────────────────────────────────────────────
    sl.log("model_load_start", "Attempting to load Nemotron adapter model")
    model, tokenizer = _try_load_model()
    model_mode = "INFERENCE" if model is not None else "ANSWER_KEY_ONLY"
    sl.log("model_load_end", f"mode={model_mode}")

    # ── inference loop ────────────────────────────────────────────────
    sl.log("eval_start", f"Running evaluation: {len(samples)} samples, mode={model_mode}")
    results = []
    predictions_jsonl = []

    for i, s in enumerate(samples, 1):
        if i % 25 == 0 or i == len(samples):
            print(f"[baseline] {i}/{len(samples)} ...", flush=True)

        raw_output, extracted, correctness = "", "", "N/A"
        fmt_fail, ext_fail = "N/A", "N/A"
        runtime_status = "OK"

        if model is not None:
            try:
                raw_output = _run_inference(model, tokenizer, s["problem"])
                extracted, ff, ef = _extract_answer(raw_output, s["category"])
                fmt_fail = "YES" if ff else "NO"
                ext_fail = "YES" if ef else "NO"
                correct = _answers_match(extracted, s["computed_expected_answer"])
                correctness = "YES" if correct else "NO"
            except Exception as exc:
                raw_output = f"INFERENCE_ERROR: {exc}"
                runtime_status = "INFERENCE_ERROR"
                correctness = "ERROR"
        else:
            runtime_status = "MODEL_NOT_AVAILABLE"

        row = {
            "sample_id": s["sample_id"],
            "split_name": s["split_name"],
            "category": s["category"],
            "difficulty_bucket": s.get("difficulty_bucket", ""),
            "computed_expected_answer": s["computed_expected_answer"],
            "baseline_prediction_status": "PREDICTED" if model else "SKIPPED",
            "model_raw_output": raw_output[:200] if raw_output else "",
            "model_extracted_answer": extracted,
            "baseline_correctness": correctness,
            "format_failure_flag": fmt_fail,
            "extraction_failure_flag": ext_fail,
            "runtime_status": runtime_status,
            "notes": "",
        }
        results.append(row)
        predictions_jsonl.append({
            "sample_id": s["sample_id"],
            "problem": s["problem"],
            "computed_expected_answer": s["computed_expected_answer"],
            "model_prediction": extracted,
            "correct": correctness,
        })

    sl.log("eval_end", f"Evaluation complete: {len(results)} rows")

    # ── write outputs ─────────────────────────────────────────────────
    sl.log("export_start", "Writing result files")

    _write_per_sample_csv(results, repo_root, eval_set)
    _write_category_summary(results, repo_root, eval_set)

    pred_path = working_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as fh:
        for row in predictions_jsonl:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if model is not None:
        (working_dir / "run_complete.flag").write_text("OK\n")
        print("[baseline] run_complete.flag written", flush=True)
    else:
        print("[baseline] ANSWER_KEY_ONLY — no model predictions (mount nemotron-adapter to enable)", flush=True)

    sl.log("export_end", "All outputs written",
           extra={"eval_set": eval_set, "model_mode": model_mode, "n_samples": len(results)})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[baseline] FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
