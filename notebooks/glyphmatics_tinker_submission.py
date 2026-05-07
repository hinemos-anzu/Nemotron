# Competition-Grade GlyphMatics Tinker Submission — Gain 1.18
# Tags: #Nemotron #LoRA #SVD #TinkerCookbook #MoE #AdapterConversion
#       #Kaggle #OfflineInference #FusedProjection #RankCompression
#
# LB Score: 0.86
# Changes from previous: SVD_ENERGY_GAIN_CAP 1.10 -> 1.18, FORCED_FUSED_RANK=32 固定
#
# Required Kaggle Inputs:
#   1. NVIDIA Nemotron Model Reasoning Challenge
#   2. nemotron-3-nano-30b-a3b-bf16
#   3. huikang/nemotron-adapter
#   4. Tinker wheelhouse (tinker, tinker-cookbook, chz wheels)
#
# Competition Rules:
#   max_lora_rank=32, max_tokens=7680, top_p=1.0, temperature=0.0
#   max_num_seqs=64, gpu_memory_utilization=0.85, max_model_len=8192

# =============================================================================
# Cell 1: 環境確認
# =============================================================================
from pathlib import Path
import os
import shutil
import subprocess
import sys
import json
import importlib.util

print("Python:", sys.version)
print("Working dir:", Path.cwd())

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

if not KAGGLE_INPUT.exists():
    raise RuntimeError("This notebook is intended to run inside Kaggle.")

KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)

def print_inputs():
    print("\n[Inputs]")
    for p in sorted(KAGGLE_INPUT.iterdir()):
        print(" -", p)

print_inputs()

# =============================================================================
# Cell 2: Tinkerオフラインインストール
# ローカルホイールをスキャンしてインターネット不使用でインストール
# =============================================================================
def list_wheel_dirs():
    rows = []
    for root in [Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp")]:
        if not root.exists():
            continue
        for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
            wheels = sorted(d.glob("*.whl"))
            if wheels:
                rows.append((d, [w.name for w in wheels]))
    return rows

def score_tinker_dir(names):
    low = " ".join(n.lower() for n in names)
    score = 0
    for token in ["tinker_cookbook", "tinker-cookbook", "tinker_", "tinker-", "chz"]:
        if token in low:
            score += 1
    return score

def find_tinker_wheelhouse():
    candidates = []
    for d, names in list_wheel_dirs():
        score = score_tinker_dir(names)
        if score:
            candidates.append((score, len(names), d, names))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0] if candidates else None

print("[Tinker] scanning wheel folders...")
for d, names in list_wheel_dirs():
    interesting = [n for n in names if ("tinker" in n.lower() or "chz" in n.lower())]
    if interesting:
        print("\n[wheel-dir]", d)
        for name in interesting:
            print(" -", name)

if importlib.util.find_spec("tinker_cookbook") is not None:
    print("[Tinker] tinker_cookbook already installed; skipping wheel install")
else:
    explicit = os.environ.get("WHEEL_DIR")
    candidate = None

    if explicit and Path(explicit).exists():
        candidate = (999, 0, Path(explicit), [p.name for p in Path(explicit).glob("*.whl")])
    else:
        candidate = find_tinker_wheelhouse()

    if candidate is None:
        raise FileNotFoundError(
            "Could not find local tinker wheelhouse. Attach a Kaggle input containing "
            "tinker-cookbook/tinker/chz wheels or set WHEEL_DIR to that folder."
        )

    _, _, wheel_dir, names = candidate
    print("[Tinker] selected wheel_dir:", wheel_dir)

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-index",
        f"--find-links={wheel_dir}",
        "tinker-cookbook",
        "tinker",
    ]
    print("[Tinker] pip:", " ".join(cmd))
    subprocess.run(cmd, check=True)

import importlib.metadata as md
import tinker_cookbook
from tinker_cookbook import weights

print("[Tinker] ready:", tinker_cookbook.__file__)
print("[Tinker] tinker_cookbook version:", md.version("tinker-cookbook"))
print("[Tinker] tinker version:", md.version("tinker"))
print("[Tinker] has build_lora_adapter:", hasattr(weights, "build_lora_adapter"))

if not hasattr(weights, "build_lora_adapter"):
    raise RuntimeError("tinker_cookbook.weights.build_lora_adapter not found")

# =============================================================================
# Cell 3: アダプタ・ベースモデルパスの自動検出
# snapshot_download() を使わず完全オフラインで動作
# =============================================================================
def find_first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return str(Path(p))
    return None

ADAPTER_PATH_CANDIDATES = [
    "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20",
    "/kaggle/input/huikang/nemotron-adapter/transformers/default/20",
    "/kaggle/input/nemotron-adapter/transformers/default/20",
    "/kaggle/input/nemotron-adapter",
]

BASE_MODEL_CANDIDATES = [
    "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
    "/kaggle/input/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
    "/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
    "/kaggle/input/nemotron-3-nano-30b-a3b-bf16",
]

ADAPTER_PATH = find_first_existing(ADAPTER_PATH_CANDIDATES)
BASE_MODEL_PATH = find_first_existing(BASE_MODEL_CANDIDATES)

if ADAPTER_PATH is None:
    hits = []
    for cfg in Path("/kaggle/input").rglob("adapter_config.json"):
        folder = cfg.parent
        if (folder / "adapter_model.safetensors").exists():
            if "nemotron" in str(folder).lower() or "adapter" in str(folder).lower():
                hits.append(folder)
    if hits:
        ADAPTER_PATH = str(sorted(hits, key=lambda p: len(str(p)))[0])

if BASE_MODEL_PATH is None:
    hits = []
    for cfg in Path("/kaggle/input").rglob("config.json"):
        folder = cfg.parent
        s = str(folder).lower()
        if "nemotron" in s and ("30b" in s or "nano" in s):
            hits.append(folder)
    if hits:
        BASE_MODEL_PATH = str(sorted(hits, key=lambda p: len(str(p)))[0])

if ADAPTER_PATH is None:
    raise FileNotFoundError("Nemotron adapter input not found. Attach huikang/nemotron-adapter.")

if BASE_MODEL_PATH is None:
    raise FileNotFoundError("Local Nemotron base model not found. Attach nemotron-3-nano-30b-a3b-bf16.")

print("[Paths] ADAPTER_PATH:", ADAPTER_PATH)
print("[Paths] BASE_MODEL_PATH:", BASE_MODEL_PATH)

if not (Path(ADAPTER_PATH) / "adapter_config.json").exists():
    raise FileNotFoundError(f"adapter_config.json missing under {ADAPTER_PATH}")

if not (Path(BASE_MODEL_PATH) / "config.json").exists():
    raise FileNotFoundError(f"config.json missing under {BASE_MODEL_PATH}")

# =============================================================================
# Cell 4: GlyphMaticsパッチ適用
# SVDエネルギー復元付きLoRAランク圧縮 + Fused Projection Transport
# FORCED_FUSED_RANK=32 (evaluator limit), SVD_ENERGY_GAIN_CAP=1.18 (raised from 1.10)
# =============================================================================
from __future__ import annotations

from collections import Counter

import torch
import tinker_cookbook.weights._adapter as A

FORCED_FUSED_RANK = int(os.environ.get("FORCED_FUSED_RANK", "32"))
SVD_ENERGY_GAIN_CAP = float(os.environ.get("SVD_ENERGY_GAIN_CAP", "1.18"))


class GlyphmaticTransportLedger:
    def __init__(self):
        self.events = []
        self.counts = Counter()

    def emit(self, *, src, dst, op, alpha, beta=None, gamma=None):
        event = {
            "alpha": alpha,
            "source": str(src),
            "dest": str(dst),
            "op": str(op),
            "beta": beta or {},
            "gamma": gamma or {},
        }
        self.events.append(event)
        self.counts[(alpha, op)] += 1

    def markdown(self) -> str:
        lines = [
            "# GlyphMatics Transport Ledger",
            "",
            "Generated during tinker-cookbook adapter conversion.",
            "",
            "## Coordinate definition",
            "",
            "- **α**: module/transport family.",
            "- **β**: tensor geometry.",
            "- **γ**: preservation/compression action.",
            "",
            "## Event summary",
            "",
            "| α | op | count |",
            "|---|---:|---:|",
        ]
        for (alpha, op), count in sorted(self.counts.items()):
            lines.append(f"| `{alpha}` | `{op}` | {count} |")

        lines += [
            "",
            "## First 60 events",
            "",
            "| # | α | op | source | destination | γ |",
            "|---:|---|---|---|---|---|",
        ]
        for i, event in enumerate(self.events[:60], 1):
            gamma = json.dumps(event["gamma"], sort_keys=True)
            lines.append(
                f"| {i} | `{event['alpha']}` | `{event['op']}` | "
                f"`{event['source']}` | `{event['dest']}` | `{gamma}` |"
            )
        return "\n".join(lines) + "\n"

    def print_summary(self):
        print("[GlyphMatics ledger] events:", len(self.events))
        for (alpha, op), count in sorted(self.counts.items()):
            print(f"[GlyphMatics ledger] {alpha}:{op}={count}")


GLYPH_LEDGER = GlyphmaticTransportLedger()


def _compress_lora_pair_to_rank(B: torch.Tensor, A_mat: torch.Tensor, rank: int):
    """Compress Delta = B @ A to rank-k with conservative SVD energy restoration."""
    delta = B.float() @ A_mat.float()

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    total_mass = S.sum().clamp_min(1e-12)
    full_energy = torch.sqrt(torch.sum(S ** 2)).clamp_min(1e-12)

    U = U[:, :rank]
    S_k = S[:rank]
    Vh = Vh[:rank, :]

    kept_energy = torch.sqrt(torch.sum(S_k ** 2)).clamp_min(1e-12)
    gain = torch.clamp(full_energy / kept_energy, min=1.0, max=SVD_ENERGY_GAIN_CAP)

    sroot = torch.sqrt(S_k)
    B_new = (U * sroot.unsqueeze(0)) * gain
    A_new = sroot.unsqueeze(1) * Vh

    stats = {
        "rank_in": int(B.shape[1]),
        "rank_out": int(rank),
        "singular_mass_kept": float(S_k.sum() / total_mass),
        "energy_gain": float(gain),
    }

    return B_new.to(B.dtype).contiguous(), A_new.to(A_mat.dtype).contiguous(), stats


def patched_merge_fused_projections(
    fused_model_key: str,
    adapter_layer_prefix: str,
    components,
    model_state_shapes,
    peft_weights,
    target_modules,
    profile,
) -> int:
    fused_out_dim = model_state_shapes[fused_model_key][0]
    fused_target_name = fused_model_key.removesuffix(".weight").rsplit(".", 1)[-1]

    component_order = None
    for target, comps in profile.fused_projection_map:
        if target == fused_target_name:
            component_order = comps
            break
    assert component_order is not None

    comp_by_name = {name: (lora_A, lora_B) for name, lora_A, lora_B in components}

    lora_A_parts = []
    comp_slices = []
    merged_rank = 0
    row_offset = 0

    for comp_name in component_order:
        if comp_name not in comp_by_name:
            raise RuntimeError(
                f"Missing component {comp_name!r} for fused target {fused_model_key!r}"
            )

        lora_A, lora_B = comp_by_name[comp_name]
        r = lora_A.shape[0]
        out_dim = lora_B.shape[0]

        lora_A_parts.append(lora_A)
        comp_slices.append((row_offset, row_offset + out_dim, r, comp_name))
        row_offset += out_dim
        merged_rank += r

    merged_lora_A = torch.cat(lora_A_parts, dim=0)
    merged_lora_B = torch.zeros(
        fused_out_dim,
        merged_rank,
        dtype=merged_lora_A.dtype,
        device=merged_lora_A.device,
    )

    rank_offset = 0
    for row_start, row_end, r, comp_name in comp_slices:
        _, lora_B = comp_by_name[comp_name]
        merged_lora_B[row_start:row_end, rank_offset:rank_offset + r] = lora_B
        rank_offset += r

    final_rank = merged_rank
    compression_stats = {
        "rank_in": int(merged_rank),
        "rank_out": int(merged_rank),
        "preservation": "exact",
    }

    if merged_rank > FORCED_FUSED_RANK:
        merged_lora_B, merged_lora_A, svd_stats = _compress_lora_pair_to_rank(
            merged_lora_B,
            merged_lora_A,
            FORCED_FUSED_RANK,
        )
        final_rank = FORCED_FUSED_RANK
        compression_stats = {
            **svd_stats,
            "preservation": "rank32_svd_energy_restored",
            "gain_cap": SVD_ENERGY_GAIN_CAP,
        }

    peft_target_key = f"{adapter_layer_prefix}.{fused_target_name}.weight"

    GLYPH_LEDGER.emit(
        src=f"{adapter_layer_prefix}.{{{','.join(component_order)}}}",
        dst=peft_target_key,
        op="fused_projection_transport",
        alpha="mamba_or_fused_projection",
        beta={
            "fused_out_dim": int(fused_out_dim),
            "component_count": len(component_order),
            "component_order": list(component_order),
        },
        gamma=compression_stats,
    )

    A._add_peft_weight(peft_target_key, merged_lora_A, merged_lora_B, peft_weights, target_modules)
    return final_rank


A._merge_fused_projections = patched_merge_fused_projections

print("[GlyphMatics] patched:", A._merge_fused_projections.__name__)
print("[GlyphMatics] FORCED_FUSED_RANK:", FORCED_FUSED_RANK)
print("[GlyphMatics] SVD_ENERGY_GAIN_CAP:", SVD_ENERGY_GAIN_CAP)

# =============================================================================
# Cell 5: アダプタビルド
# =============================================================================
OUTPUT_DIR = Path("/kaggle/working/nemotron-adapter-ready-to-submit")

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

print("[Build] adapter_path:", ADAPTER_PATH)
print("[Build] base_model_path:", BASE_MODEL_PATH)
print("[Build] output_dir:", OUTPUT_DIR)

weights.build_lora_adapter(
    base_model=str(BASE_MODEL_PATH),
    adapter_path=str(ADAPTER_PATH),
    output_path=str(OUTPUT_DIR),
)

if (OUTPUT_DIR / "adapter_config.json").exists() and (OUTPUT_DIR / "adapter_model.safetensors").exists():
    (OUTPUT_DIR / "checkpoint_complete").write_text("ok\n", encoding="utf-8")
else:
    raise FileNotFoundError("Adapter build did not produce adapter_config.json and adapter_model.safetensors")

GLYPH_LEDGER.print_summary()
ledger_text = GLYPH_LEDGER.markdown()
(OUTPUT_DIR / "GLYPHMATICS_TRANSPORT_LEDGER.md").write_text(ledger_text, encoding="utf-8")

readme = OUTPUT_DIR / "README.md"
if readme.exists():
    with open(readme, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(ledger_text)
else:
    readme.write_text(
        "# Nemotron Adapter Submission\n\n" + ledger_text,
        encoding="utf-8",
    )

print("[Build] output files:")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(" -", p.name, p.stat().st_size)

# =============================================================================
# Cell 6: submission.zip の検証・作成
# =============================================================================
import zipfile

ZIP_PATH = Path("/kaggle/working/submission.zip")

required = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "README.md",
    "checkpoint_complete",
]

missing = [name for name in required if not (OUTPUT_DIR / name).exists()]
if missing:
    raise FileNotFoundError(f"Missing required submission files: {missing}")

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

include = required + ["GLYPHMATICS_TRANSPORT_LEDGER.md"]

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in include:
        p = OUTPUT_DIR / name
        if p.exists():
            zf.write(p, arcname=name)

print("[Zip] wrote:", ZIP_PATH)
print("[Zip] size:", ZIP_PATH.stat().st_size)
print("[Zip] contents:")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    for name in zf.namelist():
        print(" -", name)

assert ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 0

# =============================================================================
# Cell 7: 最終ファイル一覧確認
# After this cell, open the Output tab and submit submission.zip
# =============================================================================
import subprocess as _sp
_sp.run(["ls", "-lah",
         "/kaggle/working/submission.zip",
         "/kaggle/working/nemotron-adapter-ready-to-submit"])

# =============================================================================
# Cell 8: singular_mass_kept 分析
# GLYPH_LEDGERのイベントを分析し、レイヤー別のSVD圧縮損失を可視化する
# Cell 5 より後に実行すること（GLYPH_LEDGERがビルド完了後に全イベントを持つため）
# =============================================================================
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

LEDGER_MD = OUTPUT_DIR / "GLYPHMATICS_TRANSPORT_LEDGER.md"

try:
    events = GLYPH_LEDGER.events
    print(f"[Source] in-memory GLYPH_LEDGER  ({len(events)} events)")
except NameError:
    print("[Source] GLYPH_LEDGER not found — parsing", LEDGER_MD)
    events = []
    pattern = re.compile(r'\|\s*`(\{.*?\})`\s*\|?\s*$')
    for line in LEDGER_MD.read_text(encoding="utf-8").splitlines():
        m = pattern.search(line)
        if m:
            try:
                gamma = json.loads(m.group(1))
                cols = [c.strip().strip("`") for c in line.split("|") if c.strip()]
                events.append({
                    "source": cols[3] if len(cols) > 3 else "",
                    "dest":   cols[4] if len(cols) > 4 else "",
                    "gamma":  gamma,
                })
            except (json.JSONDecodeError, IndexError):
                pass
    print(f"[Source] parsed {len(events)} events from Markdown")

rows = []
for i, ev in enumerate(events):
    g = ev.get("gamma", {})
    if "singular_mass_kept" not in g:
        continue
    rows.append({
        "idx":                i + 1,
        "dest":               ev.get("dest", ""),
        "rank_in":            g.get("rank_in", "?"),
        "rank_out":           g.get("rank_out", "?"),
        "singular_mass_kept": float(g["singular_mass_kept"]),
        "energy_gain":        float(g.get("energy_gain", 1.0)),
        "preservation":       g.get("preservation", ""),
    })

rows.sort(key=lambda r: r["singular_mass_kept"])

GAIN_CAP_CURRENT = 1.18

print(f"\n{'#':>3}  {'singular_mass_kept':>20}  {'energy_gain':>12}  "
      f"{'rank_in':>8}  {'rank_out':>9}  dest")
print("-" * 90)

risky = []
for r in rows:
    flag = ""
    if r["singular_mass_kept"] < 0.80:
        flag = "  ⚠ CRITICAL"
        risky.append(r)
    elif r["singular_mass_kept"] < 0.90:
        flag = "  △ low"
        risky.append(r)
    print(f"{r['idx']:>3}  {r['singular_mass_kept']:>20.4f}  {r['energy_gain']:>12.4f}  "
          f"{str(r['rank_in']):>8}  {str(r['rank_out']):>9}  {r['dest']}{flag}")

masses = np.array([r["singular_mass_kept"] for r in rows])
gains  = np.array([r["energy_gain"]        for r in rows])

print(f"\n{'─'*50}")
print(f"  count              : {len(masses)}")
print(f"  singular_mass_kept : min={masses.min():.4f}  mean={masses.mean():.4f}  max={masses.max():.4f}")
print(f"  energy_gain        : min={gains.min():.4f}   mean={gains.mean():.4f}  max={gains.max():.4f}")
print(f"  gain cap 使用率     : {(gains >= GAIN_CAP_CURRENT).sum()}/{len(gains)} レイヤーが上限に到達")
print(f"  mass_kept < 0.90   : {len(risky)} レイヤー → gain cap 引き上げ候補")
print(f"{'─'*50}\n")

SAFETY_CAP = 1.30

print("推奨レイヤー別 gain cap (安全上限 1.30):")
print(f"  {'dest':<60}  {'ideal_gain':>10}  {'recommended_cap':>16}")
print("  " + "-" * 90)

for r in rows:
    if r["energy_gain"] >= GAIN_CAP_CURRENT * 0.999:
        ideal = r["energy_gain"]
        rec   = min(ideal * 1.10, SAFETY_CAP)
        print(f"  {r['dest']:<60}  {ideal:>10.4f}  {rec:>16.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("GlyphMatics Ledger — singular_mass_kept Analysis", fontsize=13, fontweight="bold")

ax = axes[0]
ax.hist(masses, bins=10, color="#4C72B0", edgecolor="white")
ax.axvline(0.90, color="orange", linestyle="--", linewidth=1.5, label="threshold 0.90")
ax.axvline(0.80, color="red",    linestyle="--", linewidth=1.5, label="threshold 0.80")
ax.set_xlabel("singular_mass_kept")
ax.set_ylabel("count")
ax.set_title("(a) Distribution")
ax.legend(fontsize=8)

ax = axes[1]
x = np.arange(len(rows))
ax.scatter(x, masses, c=masses, cmap="RdYlGn", vmin=0.7, vmax=1.0, s=60, zorder=3)
ax.axhline(0.90, color="orange", linestyle="--", linewidth=1.2)
ax.axhline(0.80, color="red",    linestyle="--", linewidth=1.2)
ax.set_xlabel("layer index (sorted by mass_kept)")
ax.set_ylabel("singular_mass_kept")
ax.set_title("(b) Per-layer (sorted)")
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

ax = axes[2]
sc = ax.scatter(masses, gains, c=masses, cmap="RdYlGn", vmin=0.7, vmax=1.0, s=60)
ax.axvline(0.90, color="orange", linestyle="--", linewidth=1.2)
ax.axhline(GAIN_CAP_CURRENT, color="steelblue", linestyle=":", linewidth=1.5,
           label=f"current cap {GAIN_CAP_CURRENT}")
ax.set_xlabel("singular_mass_kept")
ax.set_ylabel("energy_gain (applied)")
ax.set_title("(c) mass_kept vs energy_gain")
ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="mass_kept")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "singular_mass_analysis.png", dpi=120, bbox_inches="tight")
plt.show()
print("[Plot] saved:", OUTPUT_DIR / "singular_mass_analysis.png")
