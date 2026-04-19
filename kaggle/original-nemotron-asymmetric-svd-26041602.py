import os
import re
import json
import glob
import shutil
import zipfile
import multiprocessing
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
from collections import Counter

import torch
import pandas as pd
from safetensors import safe_open
from safetensors.torch import save_file
import kagglehub

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# ==========================================
# 1. Configuration & Setup
# ==========================================
TEST_GENERATION = True
ADAPTER_PATH = "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20"
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
DATA_PATH = Path("/kaggle/input/nvidia-nemotron-3-reasoning-challenge")
WORKING_ADAPTER_DIR = "/kaggle/working/adapter"

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_PTXAS_PATH"] = "/tmp/triton/backends/nvidia/bin/ptxas"

if TEST_GENERATION:
    commands = [
        "uv pip uninstall torch torchvision torchaudio",
        "tar -cf - -C /kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script . | tar -xf - -C /tmp",
        "chmod +x /tmp/triton/backends/nvidia/bin/ptxas",
        "chmod +x /tmp/triton/backends/nvidia/bin/ptxas-blackwell",
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True)
sys.path.insert(0, "/tmp")


# ==========================================
# 2. Offline Asymmetric SVD Surgery
# ==========================================
print("\n--- Starting Offline Asymmetric SVD Surgery ---")
os.makedirs(WORKING_ADAPTER_DIR, exist_ok=True)
shutil.copytree(ADAPTER_PATH, WORKING_ADAPTER_DIR, dirs_exist_ok=True)

with open(os.path.join(WORKING_ADAPTER_DIR, "adapter_config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)
config["target_modules"] = ["k_proj", "o_proj", "in_proj", "q_proj", "up_proj", "v_proj", "down_proj", "out_proj", "lm_head"]
config["inference_mode"] = False
config["lora_alpha"] = 32
config["lora_dropout"] = 0
with open(os.path.join(WORKING_ADAPTER_DIR, "adapter_config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

RANK_MAP = {
    "o_proj": 32, "out_proj": 32, "k_proj": 32, "in_proj": 32,
    "down_proj": 24, "q_proj": 16, "v_proj": 16, "up_proj": 16,
    "gate_proj": 16, "x_proj": 16, "default": 24
}


def get_target_rank(name: str):
    for k, r in RANK_MAP.items():
        if k in name:
            return r
    return RANK_MAP["default"]


def _compress_lora_fast(B: torch.Tensor, A: torch.Tensor, target_rank: int):
    if A.shape[0] <= target_rank:
        return B, A
    Q_B, R_B = torch.linalg.qr(B.float())
    Q_A, R_A = torch.linalg.qr(A.float().T)
    core = R_B @ R_A.T
    U, S, Vh = torch.linalg.svd(core, full_matrices=False)
    new_B = (Q_B @ U[:, :target_rank]) * S[:target_rank].unsqueeze(0)
    new_A = Vh[:target_rank, :] @ Q_A.T
    return new_B.to(B.dtype).contiguous(), new_A.to(A.dtype).contiguous()


def trained_adapter_key_rename(k: str) -> str:
    return k.replace("base_model.model.model", "base_model.model.backbone")


model_key_shapes = {}
for model_safetensors in glob.glob(f"{MODEL_PATH}/*.safetensors"):
    with safe_open(model_safetensors, framework="pt", device="cpu") as f:
        for key in f.keys():
            model_key_shapes[key] = tuple(f.get_slice(key).get_shape())

adapter_tensors = {}
with safe_open(os.path.join(ADAPTER_PATH, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
    for key in f.keys():
        adapter_tensors[key] = f.get_tensor(key)

base_names = set(re.sub(r"\.lora_[AB]\.weight$", "", k) for k in adapter_tensors.keys())
mamba_merge_layers = {}
for base in base_names:
    for proj in ("gate_proj", "x_proj"):
        if f".{proj}" in base:
            mamba_merge_layers.setdefault(base.rsplit(f".{proj}", 1)[0], {})[proj] = base
mamba_merge_bases = set(v for p in mamba_merge_layers.values() for v in p.values())

tensors = {}
for base in sorted(base_names):
    lora_A, lora_B = adapter_tensors[f"{base}.lora_A.weight"], adapter_tensors[f"{base}.lora_B.weight"]
    renamed = trained_adapter_key_rename(base)
    if ".experts.w3" in base and lora_A.numel() == 0:
        continue
    if base in mamba_merge_bases:
        continue

    if ".experts.w1" in base or ".experts.w2" in base:
        if lora_A.shape[0] == 1:
            lora_A = lora_A.expand(lora_B.shape[0], -1, -1).contiguous()
        elif lora_B.shape[0] == 1:
            lora_B = lora_B.expand(lora_A.shape[0], -1, -1).contiguous()
        num_experts = lora_A.shape[0]
        proj_name = "up_proj" if ".w1" in base else "down_proj"
        for i in range(num_experts):
            exp_renamed = re.sub(r"\.experts\.w[12]", f".experts.{i}.{proj_name}", renamed)
            new_B, new_A = _compress_lora_fast(lora_B[i].contiguous(), lora_A[i].contiguous(), get_target_rank(proj_name))
            tensors[f"{exp_renamed}.lora_A.weight"], tensors[f"{exp_renamed}.lora_B.weight"] = new_A, new_B
        continue

    new_B, new_A = _compress_lora_fast(lora_B, lora_A, get_target_rank(renamed))
    tensors[f"{renamed}.lora_A.weight"], tensors[f"{renamed}.lora_B.weight"] = new_A, new_B

for layer_path, projs in sorted(mamba_merge_layers.items()):
    renamed_layer = trained_adapter_key_rename(layer_path)
    in_proj_base = f"{renamed_layer}.in_proj"
    in_proj_dim = model_key_shapes[renamed_layer.replace("base_model.model.", "") + ".in_proj.weight"][0]
    gate_A, gate_B = adapter_tensors[f"{projs['gate_proj']}.lora_A.weight"].float(), adapter_tensors[f"{projs['gate_proj']}.lora_B.weight"].float()
    x_A, x_B = adapter_tensors[f"{projs['x_proj']}.lora_A.weight"].float(), adapter_tensors[f"{projs['x_proj']}.lora_B.weight"].float()

    rank = gate_A.shape[0]
    A_cat = torch.cat([gate_A, x_A], dim=0)
    B_block = torch.zeros(in_proj_dim, 2 * rank)
    B_block[:gate_B.shape[0], :rank] = gate_B
    B_block[gate_B.shape[0]: gate_B.shape[0] + x_B.shape[0], rank:] = x_B
    new_B, new_A = _compress_lora_fast(B_block, A_cat, get_target_rank("in_proj"))
    tensors[f"{in_proj_base}.lora_A.weight"], tensors[f"{in_proj_base}.lora_B.weight"] = new_A, new_B

save_file(tensors, os.path.join(WORKING_ADAPTER_DIR, "adapter_model.safetensors"))
print(f"Weight surgery complete. Output tensors: {len(tensors)}")


def _infer_target_modules_from_safetensors(adapter_model_path: Path) -> list[str]:
    module_names = set()
    with safe_open(str(adapter_model_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.endswith(".lora_A.weight"):
                module_names.add(key[: -len(".lora_A.weight")].split(".")[-1])
    canonical = {"k_proj", "o_proj", "in_proj", "q_proj", "up_proj", "v_proj", "down_proj", "out_proj", "lm_head"}
    return sorted(m for m in module_names if m in canonical)


def reconcile_serving_metadata(adapter_dir: str | Path) -> dict:
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    model_path = adapter_dir / "adapter_model.safetensors"

    if not (config_path.exists() and model_path.exists()):
        source_candidates = [Path(os.environ.get("ADAPTER_PATH", "")), Path(ADAPTER_PATH)]
        source_candidates = [p for p in source_candidates if str(p)]
        for src in source_candidates:
            src_cfg, src_model = src / "adapter_config.json", src / "adapter_model.safetensors"
            if src_cfg.exists() and src_model.exists():
                adapter_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_cfg, config_path)
                shutil.copy2(src_model, model_path)
                break

    if not config_path.exists() or not model_path.exists():
        return {
            "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "alignment_scope": "B1 training-serving misalignment",
            "status": "BLOCKED",
            "reason": f"missing adapter files in {adapter_dir}",
        }

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    inferred = _infer_target_modules_from_safetensors(model_path)
    before_modules = cfg.get("target_modules", [])
    before_inference_mode = cfg.get("inference_mode")

    cfg["target_modules"] = inferred
    cfg["inference_mode"] = True

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    alignment = {
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "alignment_scope": "B1 training-serving misalignment",
        "status": "PASS",
        "target_modules_before": before_modules,
        "target_modules_after": inferred,
        "inference_mode_before": before_inference_mode,
        "inference_mode_after": True,
    }
    with open(adapter_dir / "serving_alignment.json", "w", encoding="utf-8") as f:
        json.dump(alignment, f, ensure_ascii=False, indent=2)
    return alignment


alignment = reconcile_serving_metadata(WORKING_ADAPTER_DIR)

# VRAMの安全解放
del adapter_tensors
del tensors
torch.cuda.empty_cache()


# ==========================================
# 3. Deterministic Hybrid Router
# ==========================================
def solve_deterministically(prompt: str) -> str | None:
    bit_match = re.search(r'bitwise (AND|OR|XOR) of (\d+) and (\d+)', prompt, re.IGNORECASE)
    if bit_match:
        op, a, b = bit_match.groups()
        a, b = int(a), int(b)
        if op.upper() == 'AND':
            return str(a & b)
        if op.upper() == 'OR':
            return str(a | b)
        if op.upper() == 'XOR':
            return str(a ^ b)

    roman_match = re.search(r'Roman numeral ([IVXLCDM]+)', prompt, re.IGNORECASE)
    if roman_match:
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total, prev_val = 0, 0
        for char in reversed(roman_match.group(1).upper()):
            val = roman_values.get(char, 0)
            total = total - val if val < prev_val else total + val
            prev_val = val
        return str(total)

    unit_match = re.search(r'(?:convert\s+)?([\d\.]+)\s*([a-zA-Z]+)\s+(?:to|in)\s+([a-zA-Z]+)', prompt, re.IGNORECASE)
    if unit_match:
        val_str, unit_from, unit_to = unit_match.groups()
        try:
            val = float(val_str)
            unit_from, unit_to = unit_from.lower(), unit_to.lower()
            if unit_from in ['celsius', 'c'] and unit_to in ['fahrenheit', 'f']:
                return f"{(val * 9/5 + 32):g}"
            if unit_from in ['fahrenheit', 'f'] and unit_to in ['celsius', 'c']:
                return f"{((val - 32) * 5/9):g}"
            if unit_from in ['km', 'kilometers'] and unit_to in ['m', 'meters']:
                return f"{(val * 1000):g}"
            if unit_from in ['m', 'meters'] and unit_to in ['km', 'kilometers']:
                return f"{(val / 1000):g}"
        except ValueError:
            pass

    eq_match = re.search(r'(?:solve for\s+([a-zA-Z])\s*[:,]\s*|find\s+([a-zA-Z])\s+if\s+)([^=]+)\s*=\s*([^?]+)', prompt, re.IGNORECASE)
    if eq_match:
        var1, var2, lhs, rhs = eq_match.groups()
        var_str = var1 if var1 else var2
        try:
            trans = (standard_transformations + (implicit_multiplication_application,))
            eq = sympy.Eq(parse_expr(lhs, transformations=trans), parse_expr(rhs, transformations=trans))
            solution = sympy.solve(eq, sympy.Symbol(var_str))
            if solution:
                return str(solution[0])
        except Exception:
            pass

    pct_match = re.search(r'([\d\.]+)\s*%\s*of\s*([\d\.]+)', prompt, re.IGNORECASE)
    if pct_match:
        try:
            return f"{((float(pct_match.group(1)) / 100) * float(pct_match.group(2))):g}"
        except Exception:
            pass

    return None


def extract_final_answer(text: str | None) -> str:
    if text is None:
        return "NOT_FOUND"
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        return [m.strip() for m in matches if m.strip()][-1]
    for pattern in [r"The final answer is:\s*([^\n]+)", r"Final answer\s*[:：]\s*([^\n]+)"]:
        if m := re.findall(pattern, text, re.IGNORECASE):
            return m[-1].strip()
    if m := re.findall(r"-?\d+(?:\.\d+)?", text):
        return m[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def cache_model(path: str | Path, exts: tuple[str, ...] = (".bin", ".pt", ".safetensors")):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def warmup_file(fpath: Path):
        try:
            with open(fpath, "rb") as f:
                while f.read(256 * 1024 * 1024):
                    pass
        except Exception:
            pass

    files = [p for p in Path(path).rglob("*") if p.is_file() and str(p).endswith(exts)]
    with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 8)) as pool:
        for _ in as_completed({pool.submit(warmup_file, f): f for f in files}):
            pass


# ==========================================
# 4. vLLM Inference
# ==========================================
def generate_predictions(test_df: pd.DataFrame, lora_path: str, row_id_col: str, **kwargs) -> pd.DataFrame:
    cache_model(MODEL_PATH)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=str(MODEL_PATH),
        tensor_parallel_size=1,
        max_num_seqs=kwargs['max_num_seqs'],
        gpu_memory_utilization=kwargs['gpu_memory_utilization'],
        max_model_len=kwargs['max_model_len'],
        enable_lora=True,
        max_lora_rank=kwargs['max_lora_rank'],
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=kwargs.get('n', 3),
        temperature=kwargs['temperature'],
        top_p=kwargs['top_p'],
        max_tokens=kwargs['max_tokens'],
    )

    tokenizer = llm.get_tokenizer()
    prompts_for_llm, llm_indices, hybrid_predictions = [], [], {}

    print("\n--- Routing Problems ---")
    for i, item in enumerate(test_df.itertuples(index=False)):
        user_content = item.prompt + "\nThink efficiently. Keep your `<think>` steps concise. \nPlease put your final answer inside `\\boxed{}`."
        deterministic_answer = solve_deterministically(item.prompt)
        if deterministic_answer is not None:
            hybrid_predictions[i] = deterministic_answer
        else:
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except Exception:
                prompt = user_content
            prompts_for_llm.append(prompt)
            llm_indices.append(i)

    if prompts_for_llm:
        outputs = llm.generate(prompts_for_llm, sampling_params=sampling_params, lora_request=LoRARequest("adapter", 1, lora_path))
        for idx, output in zip(llm_indices, outputs):
            candidate_answers = [extract_final_answer(out.text) for out in output.outputs]
            hybrid_predictions[idx] = Counter(candidate_answers).most_common(1)[0][0]

    predictions = [{"row_id_col": getattr(item, row_id_col), "prediction": hybrid_predictions[i]} for i, item in enumerate(test_df.itertuples(index=False))]
    return pd.DataFrame(predictions).rename(columns={"row_id_col": row_id_col})


# ==========================================
# 5. Execution & Submission
# ==========================================
if TEST_GENERATION:
    test_df = pd.read_csv(DATA_PATH / "test.csv")
    row_id_col = str(test_df.columns.to_list()[0])

    submission_df = generate_predictions(
        test_df=test_df,
        lora_path=WORKING_ADAPTER_DIR,
        row_id_col=row_id_col,
        max_lora_rank=32,
        max_tokens=4096,
        top_p=0.95,
        temperature=0.7,
        n=3,
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
    )

    submission_df.to_csv("submission.csv", index=False)
    print("\n--- submission.csv generated successfully ---")

    shutil.make_archive('/kaggle/working/submission', 'zip', WORKING_ADAPTER_DIR)
    print("--- submission.zip generated successfully ---")
    print("✅ All processes finished.")


# ==========================================
# 6. Day2 Evidence Collection
# ==========================================
def collect_day2_evidence(baseline_sha: str, alignment_info: dict):
    submission_zip = "/kaggle/working/submission.zip"
    exists = os.path.exists(submission_zip)

    evidence = {
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_of_truth": "Kaggle",
        "baseline_sha": baseline_sha,
        "experiment_scope": "B1: training-serving misalignment 修正",
        "one_variable_rule": True,
        "submission_zip": {
            "path": submission_zip,
            "exists": exists,
            "status": "PASS" if exists else "FAIL",
            "size_bytes": os.path.getsize(submission_zip) if exists else None,
            "file_count": 0,
            "file_list": [],
        },
        "submission_assets_preserved": False,
        "comparable_against_baseline": False,
        "worse_than_baseline": "UNCONFIRMED",
        "evidence_for_gt_086": "UNCONFIRMED",
        "provisional_verdict": "HOLD",
        "notes": [
            f"B1 serving alignment status={alignment_info.get('status', 'UNKNOWN')}",
            f"reason={alignment_info.get('reason', '')}",
        ],
    }

    if exists:
        with zipfile.ZipFile(submission_zip, "r") as zf:
            file_list = zf.namelist()
        evidence["submission_zip"]["file_count"] = len(file_list)
        evidence["submission_zip"]["file_list"] = file_list
        evidence["submission_assets_preserved"] = evidence["submission_zip"]["size_bytes"] not in [None, 0] and len(file_list) > 0

    with open("/kaggle/working/day2_evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    with open("/kaggle/working/day2_evidence.md", "w", encoding="utf-8") as f:
        f.write("# Day2 Evidence Report\n\n")
        for k in ["timestamp_utc", "source_of_truth", "baseline_sha", "experiment_scope", "one_variable_rule", "provisional_verdict", "comparable_against_baseline", "worse_than_baseline", "evidence_for_gt_086"]:
            f.write(f"- {k}: {evidence[k]}\n")
        f.write("\n## submission_zip\n")
        for k, v in evidence["submission_zip"].items():
            if isinstance(v, list):
                f.write(f"- {k}:\n")
                for item in v:
                    f.write(f"  - {item}\n")
            else:
                f.write(f"- {k}: {v}\n")


collect_day2_evidence("39f4bed90392567517b606d1301ae1c36a86a97c", alignment)
