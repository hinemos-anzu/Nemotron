#!/usr/bin/env python3
"""Phase 3 Golden-Adapter Artifact Audit (Tasks 1-3, CPU-only).

SAFETY CONTRACT:
  - READ-ONLY w.r.t. every existing adapter directory under /kaggle/input.
  - The only WRITE this script performs is extracting B3's submission.zip
    into a NEW directory (--b3-dest, default /kaggle/working/golden_b3_adapter).
    The zip itself and all /kaggle/input sources are never modified.
  - Does NOT change adapter_config.json, rank maps, target_modules, dtype,
    or LoRA structure of anything. It only *reads* and *reports*.
  - Does NOT run the model, train, or create submission.zip.
  - No GPU / torch required.

What this does:
  Task 1: locate B3's submission.zip and extract it to --b3-dest.
  Task 2: audit the extracted B3 adapter (tensor_count, prefix_counts,
          target_modules, rank_histogram, dtype_counts, shape_summary,
          bad_pattern_counts, max_rank, rank_gt_32_count) against the
          expected values from the task spec. STOPS (exit 1) if the
          artifact does not look like the expected B3 Golden adapter,
          unless --force is given.
  Task 3: audit the OLD adapter currently used by phase3_run_golden_validation.py
          (--old-adapter) the same way, and write a 3-column config diff
          ("phase3_baseline_script" / "phase3_executed_notebook" /
          "b3_golden_measured_or_unknown"). Sets artifact_mismatch=True
          (by construction - Phase3's ADAPTER_PATH is B3's pre-conversion
          *input*, not its submitted *output*; see
          bab0532c-b3nemotronsvd26042701.ipynb) and reports whether this
          session's tensor-level measurements of --old-adapter are
          consistent with that. Also sets prompt_harness_mismatch=True
          (the executed notebook applies ChatML chat-template wrapping;
          phase3_run_golden_validation.py does not - flagged as an
          out-of-scope follow-up, not changed here).

Outputs (under --output-dir, default phase3_token_budget_audit/):
  phase3_config_audit/golden_b3_adapter_audit.json
  phase3_config_audit/golden_b3_safetensors_summary.csv
  phase3_config_audit/golden_b3_adapter_config.json
  phase3_config_audit/golden_b3_artifact_report.md
  phase3_config_audit/old_adapter_audit.json
  phase3_config_audit/old_adapter_safetensors_summary.csv
  phase3_config_audit/old_adapter_artifact_report.md
  phase3_config_audit/phase3_vs_golden_config_diff.md
  phase3_config_audit/phase3_vs_golden_config_diff.csv
  phase3_config_audit/STAGE1_STATUS.json   (consumed by phase3_token_budget_compare.py)

Usage (Kaggle, CPU session is sufficient for this stage):
  python phase3_adapter_artifact_audit.py \
      --b3-dest /kaggle/working/golden_b3_adapter \
      --old-adapter /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20 \
      --output-dir phase3_token_budget_audit
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Expected values (from task spec)
# ---------------------------------------------------------------------------

EXPECTED_TENSOR_COUNT = 12010
DEFAULT_TENSOR_COUNT_TOLERANCE = 0.02  # +/- 2%
EXPECTED_PREFIX_SUBSTRING = "base_model.model.backbone"
EXPECTED_TARGET_MODULES = {
    "k_proj", "o_proj", "in_proj", "q_proj", "up_proj",
    "v_proj", "down_proj", "out_proj", "lm_head",
}
EXPECTED_MAX_RANK = 32

# ---------------------------------------------------------------------------
# B3 conversion ground truth, taken verbatim from the user-supplied
# b3-nemotron-svd-26042701.ipynb (B3_INPROJ_SPLIT_GATE_X_16_16 + clean
# submission.zip). Used to make the Task 2 audit of the EXTRACTED B3 adapter
# precise rather than relying only on the generic spec values above.
#
# RANK_MAP: per-module target rank used by compress_lora_fast(). Lookup is
# "first key that is a substring of the (renamed) tensor's module name wins";
# "default" is the fallback (covers e.g. lm_head).
# NOTE: in_proj is special-cased in the conversion - gate_proj and x_proj are
# each compressed to 16 and concatenated, so the OUTPUT in_proj lora_A rank is
# 2*16=32, not a direct application of RANK_MAP["in_proj"].
B3_RANK_MAP = {
    "o_proj": 32,
    "out_proj": 32,
    "k_proj": 32,
    "in_proj": 32,
    "down_proj": 22,
    "up_proj": 22,
    "q_proj": 32,
    "v_proj": 32,
    "gate_proj": 16,
    "x_proj": 16,
    "default": 24,
}

# Patterns that MUST be ABSENT from a correctly-converted B3 adapter
# (validate_adapter()'s BAD_PATTERNS in b3-nemotron-svd-26042701.ipynb).
# .gate_proj./.x_proj. are merged into in_proj; .experts.w1/w2/w3. are
# unfused into per-expert up_proj/down_proj.
B3_BAD_PATTERNS = [".experts.w1.", ".experts.w2.", ".experts.w3.", ".gate_proj.", ".x_proj."]

# Patterns expected to be PRESENT post key-rename
# (validate_adapter()'s GOOD_PATTERNS in b3-nemotron-svd-26042701.ipynb).
B3_GOOD_PATTERNS = [
    ".in_proj.", ".out_proj.", ".up_proj.", ".down_proj.",
    ".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.", ".lm_head.",
]

# trained_adapter_key_rename() in b3-nemotron-svd-26042701.ipynb does:
#   key.replace("base_model.model.model", "base_model.model.backbone")
# A correctly-converted B3 adapter must contain ZERO keys with the OLD
# prefix substring below (it should all have become EXPECTED_PREFIX_SUBSTRING).
B3_OLD_PREFIX_SUBSTRING = "base_model.model.model"

SUBMISSION_ZIP_CANDIDATES = [
    "/kaggle/input/notebooks/hinemos/b3-nemotron-svd-26042701/submission.zip",
    "/kaggle/input/notebooks/hinemos/*b3*nemotron*svd*/submission.zip",
    "/kaggle/input/*b3*nemotron*svd*/submission.zip",
    "/kaggle/input/*/submission.zip",
    "/kaggle/input/*/*/submission.zip",
    "/kaggle/input/*/*/*/submission.zip",
]

DEFAULT_OLD_ADAPTER = "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20"
DEFAULT_B3_DEST = "/kaggle/working/golden_b3_adapter"

MODULE_PATTERNS = [
    re.compile(r"\.([a-zA-Z0-9_]+)\.lora_([AB])\.weight$"),
    re.compile(r"\.([a-zA-Z0-9_]+)\.modules_to_save\.[^.]+\.weight$"),
    re.compile(r"\.([a-zA-Z0-9_]+)\.original_module\.weight$"),
]


# ---------------------------------------------------------------------------
# Low-level helpers (no torch / safetensors package required)
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_safetensors_header(path: Path) -> Dict[str, Any]:
    """Parse a .safetensors header without the safetensors package or torch.

    Format: 8-byte little-endian u64 header length, followed by that many
    bytes of UTF-8 JSON. Each non-metadata key maps to
    {"dtype": str, "shape": [int, ...], "data_offsets": [int, int]}.
    """
    with path.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header_json = fh.read(header_len).decode("utf-8")
    header = json.loads(header_json)
    header.pop("__metadata__", None)
    return header


def extract_module_name(key: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (module_name, lora_side) for a tensor key, lora_side in {"A","B",None}."""
    for pat in MODULE_PATTERNS:
        m = pat.search(key)
        if m:
            groups = m.groups()
            if len(groups) == 2:
                return groups[0], groups[1]
            return groups[0], None
    return None, None


# ---------------------------------------------------------------------------
# Task 2: adapter artifact audit (generic — used for old + B3 adapters)
# ---------------------------------------------------------------------------

def audit_adapter_dir(adapter_dir: Path) -> Dict[str, Any]:
    audit: Dict[str, Any] = {"adapter_dir": str(adapter_dir)}

    cfg_path = adapter_dir / "adapter_config.json"
    audit["adapter_config_exists"] = cfg_path.exists()
    if cfg_path.exists():
        try:
            adapter_config = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            adapter_config = {"_read_error": f"{type(exc).__name__}: {exc}"}
        audit["adapter_config"] = adapter_config
        audit["target_modules_config"] = sorted(adapter_config.get("target_modules") or [])
        audit["r_config"] = adapter_config.get("r")
        audit["lora_alpha_config"] = adapter_config.get("lora_alpha")
    else:
        audit["adapter_config"] = None
        audit["target_modules_config"] = []
        audit["r_config"] = None
        audit["lora_alpha_config"] = None

    st_path = adapter_dir / "adapter_model.safetensors"
    audit["adapter_model_exists"] = st_path.exists()
    if not st_path.exists():
        audit["error"] = "adapter_model.safetensors not found at adapter root"
        return audit

    audit["file_size_bytes"] = st_path.stat().st_size
    audit["sha256"] = sha256_file(st_path)

    header = read_safetensors_header(st_path)
    audit["tensor_count"] = len(header)

    dtype_counts: Counter = Counter()
    prefix_counts: Counter = Counter()
    module_tensor_counts: Counter = Counter()
    rank_counter: Counter = Counter()
    shape_counter: Counter = Counter()
    lora_a_keys = set()
    lora_b_keys = set()
    total_params = 0
    per_tensor_rows: List[Dict[str, Any]] = []
    b3_bad_pattern_counter: Counter = Counter()
    b3_good_pattern_counter: Counter = Counter()
    old_prefix_count = 0
    new_prefix_count = 0
    rank_by_module: Dict[str, Counter] = {}

    for key, meta in header.items():
        dtype = meta.get("dtype", "?")
        shape = tuple(meta.get("shape", []))
        dtype_counts[dtype] += 1
        shape_counter[shape] += 1
        numel = 1
        for d in shape:
            numel *= d
        total_params += numel

        parts = key.split(".")
        prefix = ".".join(parts[:4])
        prefix_counts[prefix] += 1

        for pat in B3_BAD_PATTERNS:
            if pat in key:
                b3_bad_pattern_counter[pat] += 1
        for pat in B3_GOOD_PATTERNS:
            if pat in key:
                b3_good_pattern_counter[pat] += 1
        if B3_OLD_PREFIX_SUBSTRING in key:
            old_prefix_count += 1
        if EXPECTED_PREFIX_SUBSTRING in key:
            new_prefix_count += 1

        module_name, lora_side = extract_module_name(key)
        rank = None
        if module_name:
            module_tensor_counts[module_name] += 1
            base_key = key.rsplit(".", 2)[0]
            if lora_side == "A":
                lora_a_keys.add(base_key)
                rank = shape[0] if shape else None
                if rank is not None:
                    rank_counter[rank] += 1
                    rank_by_module.setdefault(module_name, Counter())[rank] += 1
            elif lora_side == "B":
                lora_b_keys.add(base_key)
                rank = shape[1] if len(shape) > 1 else None

        per_tensor_rows.append({
            "tensor_name": key,
            "dtype": dtype,
            "shape": "x".join(str(d) for d in shape),
            "numel": numel,
            "prefix": prefix,
            "module_name": module_name or "",
            "lora_side": lora_side or "",
            "rank": rank if rank is not None else "",
        })

    audit["dtype_counts"] = dict(dtype_counts)
    audit["prefix_counts"] = dict(prefix_counts.most_common(30))
    audit["target_modules_in_weights"] = sorted(module_tensor_counts.keys())
    audit["target_module_tensor_counts"] = dict(module_tensor_counts)
    audit["rank_histogram"] = {str(k): v for k, v in sorted(rank_counter.items())}
    audit["max_rank"] = max(rank_counter.keys()) if rank_counter else None
    audit["rank_gt_32_count"] = sum(c for r, c in rank_counter.items() if r > EXPECTED_MAX_RANK)
    audit["total_params"] = total_params
    audit["shape_summary"] = {
        "unique_shapes": len(shape_counter),
        "top_shapes": [
            {"shape": "x".join(str(d) for d in s), "count": c}
            for s, c in shape_counter.most_common(10)
        ],
    }

    orphan_a = lora_a_keys - lora_b_keys
    orphan_b = lora_b_keys - lora_a_keys
    known_modules = set(audit["target_modules_config"]) | EXPECTED_TARGET_MODULES
    unexpected_modules = set(module_tensor_counts) - known_modules
    audit["bad_pattern_counts"] = {
        "orphan_lora_A": len(orphan_a),
        "orphan_lora_B": len(orphan_b),
        "unexpected_target_module_keys": sum(
            module_tensor_counts[m] for m in unexpected_modules
        ),
        "rank_gt_32": audit["rank_gt_32_count"],
    }

    # B3-conversion-specific fields (ground truth from
    # b3-nemotron-svd-26042701.ipynb's validate_adapter()). Meaningful for the
    # B3-extracted adapter; for the OLD (pre-conversion) adapter these are
    # EXPECTED to be non-zero/non-empty (see check_against_expected's
    # is_b3_converted flag).
    audit["b3_bad_pattern_hits"] = {p: b3_bad_pattern_counter.get(p, 0) for p in B3_BAD_PATTERNS}
    audit["b3_good_pattern_hits"] = {p: b3_good_pattern_counter.get(p, 0) for p in B3_GOOD_PATTERNS}
    audit["old_prefix_count"] = old_prefix_count  # keys still containing "base_model.model.model"
    audit["new_prefix_count"] = new_prefix_count  # keys containing "base_model.model.backbone"
    audit["rank_by_module"] = {
        m: {str(r): c for r, c in sorted(counter.items())}
        for m, counter in sorted(rank_by_module.items())
    }
    audit["max_rank_by_module"] = {m: max(counter) for m, counter in rank_by_module.items()}

    audit["_per_tensor_rows"] = per_tensor_rows  # consumed by CSV writer, stripped from JSON
    return audit


def check_against_expected(
    audit: Dict[str, Any],
    tensor_count_tolerance: float,
    is_b3_converted: bool = True,
) -> Dict[str, Any]:
    """Check an audited adapter against expected values.

    is_b3_converted=True (default; use for the B3-extracted adapter): also
    checks the conversion-specific invariants from
    b3-nemotron-svd-26042701.ipynb (B3_BAD_PATTERNS absent, old key prefix
    absent, per-module rank <= B3_RANK_MAP).

    is_b3_converted=False (use for the OLD/pre-conversion adapter): skips
    those conversion-specific checks, since the pre-conversion adapter is
    EXPECTED to contain gate_proj/x_proj/.experts.w1-3./old key prefix - their
    presence there is normal, not a failure.
    """
    failures: List[str] = []

    if not audit.get("adapter_config_exists"):
        failures.append("adapter_config.json not found")
    if not audit.get("adapter_model_exists"):
        failures.append("adapter_model.safetensors not found")
    if failures:
        return {"pass": False, "failures": failures}

    tc = audit.get("tensor_count")
    lo = EXPECTED_TENSOR_COUNT * (1 - tensor_count_tolerance)
    hi = EXPECTED_TENSOR_COUNT * (1 + tensor_count_tolerance)
    if tc is None or not (lo <= tc <= hi):
        failures.append(
            f"tensor_count={tc} outside expected range "
            f"[{lo:.0f}, {hi:.0f}] (~{EXPECTED_TENSOR_COUNT}, "
            f"tolerance={tensor_count_tolerance:.0%})"
        )

    prefixes = audit.get("prefix_counts", {})
    if not any(EXPECTED_PREFIX_SUBSTRING in p for p in prefixes):
        failures.append(
            f"no tensor prefix contains '{EXPECTED_PREFIX_SUBSTRING}' "
            f"(saw prefixes: {sorted(prefixes)[:10]})"
        )

    tm = set(audit.get("target_modules_in_weights", [])) | set(audit.get("target_modules_config", []))
    missing_tm = EXPECTED_TARGET_MODULES - tm
    if missing_tm:
        failures.append(f"missing target_modules: {sorted(missing_tm)} (saw: {sorted(tm)})")

    max_rank = audit.get("max_rank")
    if max_rank is not None and max_rank > EXPECTED_MAX_RANK:
        failures.append(f"max_rank={max_rank} > {EXPECTED_MAX_RANK}")
    if audit.get("rank_gt_32_count", 0) > 0:
        failures.append(f"rank_gt_32_count={audit['rank_gt_32_count']} (expected 0)")

    if is_b3_converted:
        # 1. B3_BAD_PATTERNS must be entirely absent (gate_proj/x_proj merged
        #    into in_proj; MoE experts.w1/w2/w3 unfused into per-expert
        #    up_proj/down_proj).
        bad_hits = audit.get("b3_bad_pattern_hits", {})
        nonzero_bad = {p: c for p, c in bad_hits.items() if c}
        if nonzero_bad:
            failures.append(
                f"B3_BAD_PATTERNS present in converted adapter (expected all-zero): {nonzero_bad}"
            )

        # 2. Old key prefix ("base_model.model.model") must be fully renamed
        #    to "base_model.model.backbone".
        if audit.get("old_prefix_count", 0) > 0:
            failures.append(
                f"old_prefix_count={audit['old_prefix_count']} keys still contain "
                f"'{B3_OLD_PREFIX_SUBSTRING}' (trained_adapter_key_rename not fully applied)"
            )

        # 3. Per-module rank must not exceed B3_RANK_MAP[module] (in_proj's
        #    output rank is 32 via the gate/x 16+16 merge, which equals
        #    B3_RANK_MAP["in_proj"]=32, so the same upper-bound check applies).
        rank_violations = {}
        for module, observed_max in audit.get("max_rank_by_module", {}).items():
            expected = B3_RANK_MAP.get(module, B3_RANK_MAP["default"])
            if observed_max > expected:
                rank_violations[module] = {"observed_max_rank": observed_max, "expected_max_rank": expected}
        if rank_violations:
            failures.append(f"per-module rank exceeds B3_RANK_MAP: {rank_violations}")

    return {"pass": len(failures) == 0, "failures": failures}


# ---------------------------------------------------------------------------
# Task 1: locate + extract B3 submission.zip
# ---------------------------------------------------------------------------

def find_submission_zip(explicit: Optional[str]) -> Tuple[Optional[Path], List[str]]:
    """Return (path_or_None, log_lines) describing the search."""
    log: List[str] = []
    if explicit:
        p = Path(explicit)
        log.append(f"--submission-zip explicitly given: {p}")
        if p.is_file():
            return p, log
        log.append(f"  -> NOT FOUND at explicit path")
        return None, log

    for pattern in SUBMISSION_ZIP_CANDIDATES:
        matches = sorted(glob.glob(pattern))
        log.append(f"glob '{pattern}' -> {len(matches)} match(es)")
        for m in matches:
            log.append(f"    {m}")
        for m in matches:
            p = Path(m)
            if p.is_file():
                return p, log

    log.append("Falling back to recursive search under /kaggle/input/**/submission.zip ...")
    matches = sorted(glob.glob("/kaggle/input/**/submission.zip", recursive=True))
    log.append(f"recursive glob -> {len(matches)} match(es)")
    for m in matches:
        log.append(f"    {m}")
    for m in matches:
        p = Path(m)
        if p.is_file():
            return p, log

    return None, log


def extract_submission_zip(zip_path: Path, dest_dir: Path) -> List[str]:
    log: List[str] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        log.append(f"submission.zip contains {len(names)} entries")
        zf.extractall(dest_dir)

    if (dest_dir / "adapter_model.safetensors").exists():
        log.append(f"adapter_model.safetensors found at root of {dest_dir}")
        return log

    # Not at root - find it nested and copy the bundle up to root.
    found = list(dest_dir.rglob("adapter_model.safetensors"))
    if not found:
        log.append("WARNING: adapter_model.safetensors not found anywhere in extracted zip")
        return log

    src_dir = found[0].parent
    log.append(f"adapter_model.safetensors found nested at {src_dir} - copying bundle to root")
    import shutil
    for fname in ("adapter_model.safetensors", "adapter_config.json", "README.md", "checkpoint_complete"):
        src = src_dir / fname
        if src.exists() and src != dest_dir / fname:
            shutil.copy2(src, dest_dir / fname)
            log.append(f"  copied {fname} -> {dest_dir / fname}")
    return log


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_safetensors_summary_csv(audit: Dict[str, Any], path: Path) -> None:
    rows = audit.get("_per_tensor_rows", [])
    fields = ["tensor_name", "dtype", "shape", "numel", "prefix", "module_name", "lora_side", "rank"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_to_json(audit: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in audit.items() if k != "_per_tensor_rows"}
    return out


def format_artifact_report_md(audit: Dict[str, Any], check: Dict[str, Any], title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- adapter_dir: `{audit.get('adapter_dir')}`")
    lines.append(f"- adapter_config.json exists: **{audit.get('adapter_config_exists')}**")
    lines.append(f"- adapter_model.safetensors exists: **{audit.get('adapter_model_exists')}**")
    if audit.get("error"):
        lines.append(f"- ERROR: {audit['error']}")
        lines.append("")
        lines.append(f"## Check result: {'PASS' if check['pass'] else 'FAIL'}")
        for f in check["failures"]:
            lines.append(f"- FAIL: {f}")
        return "\n".join(lines) + "\n"

    lines.append(f"- file_size_bytes: {audit.get('file_size_bytes')}")
    lines.append(f"- sha256: `{audit.get('sha256')}`")
    lines.append(f"- tensor_count: **{audit.get('tensor_count')}** (expected ~{EXPECTED_TENSOR_COUNT})")
    lines.append(f"- total_params: {audit.get('total_params')}")
    lines.append(f"- max_rank: **{audit.get('max_rank')}** (expected <= {EXPECTED_MAX_RANK})")
    lines.append(f"- rank_gt_32_count: **{audit.get('rank_gt_32_count')}** (expected 0)")
    lines.append("")

    lines.append("## adapter_config.json (key fields)")
    lines.append(f"- r: {audit.get('r_config')}")
    lines.append(f"- lora_alpha: {audit.get('lora_alpha_config')}")
    lines.append(f"- target_modules (config): {audit.get('target_modules_config')}")
    lines.append("")

    lines.append("## target_modules found in weights")
    tm_in = audit.get("target_modules_in_weights", [])
    counts = audit.get("target_module_tensor_counts", {})
    lines.append(f"- modules: {tm_in}")
    for m in tm_in:
        lines.append(f"  - {m}: {counts.get(m)} tensors")
    missing = EXPECTED_TARGET_MODULES - (set(tm_in) | set(audit.get("target_modules_config", [])))
    if missing:
        lines.append(f"- MISSING vs expected: {sorted(missing)}")
    lines.append("")

    lines.append("## rank_histogram")
    for r, c in audit.get("rank_histogram", {}).items():
        lines.append(f"- rank {r}: {c} lora_A tensors")
    lines.append("")

    lines.append("## dtype_counts")
    for dt, c in audit.get("dtype_counts", {}).items():
        lines.append(f"- {dt}: {c}")
    lines.append("")

    lines.append("## prefix_counts (top entries)")
    for p, c in audit.get("prefix_counts", {}).items():
        lines.append(f"- `{p}`: {c}")
    lines.append("")

    lines.append("## shape_summary")
    ss = audit.get("shape_summary", {})
    lines.append(f"- unique_shapes: {ss.get('unique_shapes')}")
    for s in ss.get("top_shapes", []):
        lines.append(f"  - {s['shape']}: {s['count']} tensors")
    lines.append("")

    lines.append("## bad_pattern_counts")
    lines.append(
        "Definitions: orphan_lora_A/B = lora_A without matching lora_B (or vice versa); "
        "unexpected_target_module_keys = tensors whose module name is in neither "
        "adapter_config.target_modules nor the spec's expected target_modules list; "
        "rank_gt_32 = lora_A tensors with rank > 32."
    )
    for k, v in audit.get("bad_pattern_counts", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## B3 conversion checks (from b3-nemotron-svd-26042701.ipynb ground truth)")
    lines.append(f"- old_prefix_count (`{B3_OLD_PREFIX_SUBSTRING}` keys, expect 0 if converted): {audit.get('old_prefix_count')}")
    lines.append(f"- new_prefix_count (`{EXPECTED_PREFIX_SUBSTRING}` keys): {audit.get('new_prefix_count')}")
    lines.append("- B3_BAD_PATTERNS hits (expect all 0 if converted):")
    for p, c in audit.get("b3_bad_pattern_hits", {}).items():
        lines.append(f"  - `{p}`: {c}")
    lines.append("- B3_GOOD_PATTERNS hits:")
    for p, c in audit.get("b3_good_pattern_hits", {}).items():
        lines.append(f"  - `{p}`: {c}")
    lines.append("- max_rank_by_module vs B3_RANK_MAP:")
    for m, observed in sorted(audit.get("max_rank_by_module", {}).items()):
        expected = B3_RANK_MAP.get(m, B3_RANK_MAP["default"])
        flag = "OK" if observed <= expected else "EXCEEDS EXPECTED"
        lines.append(f"  - {m}: observed_max_rank={observed}, B3_RANK_MAP_expected={expected} [{flag}]")
    lines.append("")

    lines.append(f"## Check result: {'PASS' if check['pass'] else 'FAIL'}")
    if check["failures"]:
        for f in check["failures"]:
            lines.append(f"- FAIL: {f}")
    else:
        lines.append("- All expected-value checks passed.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Task 3: config diff
#   columns: phase3_baseline_script | phase3_executed_notebook | b3_golden_measured_or_unknown
# ---------------------------------------------------------------------------
#
# Three artifacts are compared:
#
#   phase3_baseline_script   = phase3_run_golden_validation.py in this repo.
#       This is the script phase3_token_budget_compare.py (Stage 2) drives
#       for Tasks 4-7, varying ONLY adapter_path / max_new_tokens per the
#       task spec's "ONLY allowed verification-time changes" list.
#
#   phase3_executed_notebook = the notebook the user ran (RTX Pro 5000) that
#       produced the n=20/30, 20% (4/20) result motivating this audit
#       (supplied separately from this repo; hardcoded below from a direct
#       reading of its source - see EXECUTED_NOTEBOOK_NAME).
#
#   b3_golden_measured_or_unknown = measured from the B3-extracted adapter
#       (Task 1/2) for adapter-structural rows.
#       b3-nemotron-svd-26042701.ipynb (also supplied separately) performs
#       ADAPTER WEIGHT CONVERSION ONLY (SVD compression / key rename / expert
#       unfuse / gate-x merge) and contains NO inference/generation/prompt/
#       parser code, so generation/harness rows cannot be derived from it -
#       the actual Kaggle submission inference notebook (Public LB 0.86) is a
#       separate, not-provided artifact for those rows.

EXECUTED_NOTEBOOK_NAME = "claude-rtx-nemotron-2606-05-01_4.ipynb (RTX Pro 5000 'Golden Baseline Inference' notebook)"

# Hardcoded reference for the "phase3_executed_notebook" column, derived from
# a direct reading of that notebook's source (it is not part of this repo).
EXECUTED_NOTEBOOK_REF: Dict[str, str] = {
    "adapter_path": "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20 "
                    "(hardcoded constant, IDENTICAL string to phase3_baseline_script default "
                    "-> same artifact as old_adapter_audit.json)",
    "adapter_config": "same artifact as old_adapter (see old_adapter_audit.json)",
    "adapter_model_tensor_count": "same artifact as old_adapter (see old_adapter_audit.json)",
    "prefix": "same artifact as old_adapter (see old_adapter_audit.json)",
    "target_modules": "same artifact as old_adapter (see old_adapter_audit.json)",
    "rank": "same artifact as old_adapter (see old_adapter_audit.json)",
    "dtype": "same artifact as old_adapter (see old_adapter_audit.json)",
    "base_model_path": "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1 "
                       "(hardcoded path; same model id as phase3_baseline_script's kagglehub default)",
    "base_model_quantization": "CONDITIONAL: 4-bit NF4 (BitsAndBytesConfig(load_in_4bit=True, "
                     "bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True, "
                     "bnb_4bit_quant_type='nf4')) ONLY if total GPU VRAM < 50GB; OTHERWISE "
                     "(>= 50GB) loads the base model in FULL bfloat16/fp16 with NO "
                     "quantization_config at all. On RTX Pro 5000 (>= 50GB) this took the "
                     ">=50GB branch -> ran in FULL BF16, no 4-bit quantization. "
                     "DIFFERS from phase3_baseline_script -> see quantization_mismatch verdict.",
    "prompt_template": "build_prompt(): IDENTICAL text to phase3_baseline_script "
                       "('Solve the following problem carefully...' + Question + "
                       "'Think step by step. Put your final answer inside \\\\boxed{}.')",
    "chat_template": "APPLIES ChatML wrapping before tokenizing: "
                     "tokenizer.apply_chat_template([{role:user,content:_user_text}], "
                     "add_generation_prompt=True) if available, else manual "
                     "'<|im_start|>user\\n{_user_text}\\n<|im_end|>\\n<|im_start|>assistant\\n'. "
                     "DIFFERS from phase3_baseline_script -> see prompt_harness_mismatch verdict.",
    "max_new_tokens": "2048 (GOLDEN_GENERATION_CONFIG, identical dict to phase3_baseline_script)",
    "temperature": "0.0 (identical)",
    "top_p": "(unset; do_sample=False so top_p has no effect) (identical)",
    "num_beams": "1 (default, not overridden) (identical)",
    "do_sample": "False (identical)",
    "stop_condition": "['<|endoftext|>', '<|im_end|>'] (identical)",
    "eos_token_id": "tokenizer.eos_token_id plus chat-stop ids via same _eos_ids construction (identical logic)",
    "parser": "extract_answer(): IDENTICAL BOXED_RE / THEREFORE_RE / LAST_LINE_RE regexes to phase3_baseline_script",
    "output_scores": "True (model.generate(..., output_scores=True, return_dict_in_generate=True)) "
                     "-> DIFFERS from phase3_baseline_script (False). See Hypothesis 4 / logprob_fields.",
    "return_dict_in_generate": "True -> DIFFERS from phase3_baseline_script (False).",
    "logprob_fields": "computes token_logprobs / min_logprob / mean_logprob per generated token "
                      "via output.scores (requires output_scores=True/return_dict_in_generate=True "
                      "above); written to golden_validation_predictions.jsonl + min_logprob_analysis.csv",
    "mamba_patch_mechanism": "USE_MAMBA_PATCH=False for RTX Pro 5000 (so _apply_mamba_patch() is "
                            "NOT called), but a SEPARATE unconditional inline mamba_ssm stub "
                            "(pure-PyTorch rmsnorm_fn) is injected before model load regardless. "
                            "DIFFERS in mechanism from phase3_baseline_script's unconditional "
                            "_apply_mamba_patch() call, but both aim to avoid the broken CUDA "
                            "mamba_ssm extension; environment-dependent (T4x2 vs RTX Pro 5000), "
                            "not expected to change generation correctness on a working GPU.",
    "validation_source": "PROBLEMS_PATH = /kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv "
                         "(same competition file phase3_baseline_script's --problems should point at)",
    "MAX_PROBLEMS": "30",
    "problem_selection": "random.seed(SEED=42); random.shuffle(problems); problems[:MAX_PROBLEMS] "
                         "-> DIFFERS from phase3_baseline_script (problems[:N], first N, NO shuffle). "
                         "Affects WHICH problem_ids were in the audited n=20/30 set; not addressed by "
                         "this ticket. Stage 2 (Task 5) selects its subset BY problem_id from the "
                         "existing golden_validation_predictions.jsonl, so this does not block Stage 2.",
    "input_truncation": "_MAX_INPUT_TOKENS = 768 (identical to phase3_baseline_script)",
    "seed": "42 (identical)",
}


def get_phase3_config() -> Dict[str, str]:
    """Static facts about phase3_run_golden_validation.py (the
    "phase3_baseline_script" column). Does not require torch/GPU."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import phase3_run_golden_validation as p3  # type: ignore
        cfg = p3.GOLDEN_GENERATION_CONFIG
        seed = p3.SEED
    except Exception as exc:  # pragma: no cover - defensive
        cfg = {}
        seed = "?"

    return {
        "adapter_path": DEFAULT_OLD_ADAPTER,
        "adapter_config": "see old_adapter_audit.json",
        "adapter_model_tensor_count": "see old_adapter_audit.json",
        "prefix": "see old_adapter_audit.json",
        "target_modules": "see old_adapter_audit.json",
        "rank": "see old_adapter_audit.json",
        "dtype": "see old_adapter_audit.json",
        "base_model_path": "auto-resolved via MODEL_PATH env / kagglehub "
                            "(metric/nemotron-3-nano-30b-a3b-bf16)",
        "base_model_quantization": "ALWAYS 4-bit NF4 via bitsandbytes "
                     "(BitsAndBytesConfig(load_in_4bit=True, "
                     "bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True, "
                     "bnb_4bit_quant_type='nf4')), torch_dtype=compute_dtype "
                     "(bf16 if SM>=8 else fp16), device_map='auto', max_memory=80%/GPU. "
                     "HARD REQUIREMENT: raises RuntimeError if bitsandbytes is not "
                     "importable, or if total GPU VRAM < 16.5GB (lines ~362-457). NOT "
                     "conditional on VRAM beyond that floor - i.e. ALWAYS 4-bit even on "
                     "GPUs with ample VRAM for full bf16. Explicitly tuned/commented for "
                     "Kaggle T4x2 (29.2GB).",
        "prompt_template": "build_prompt(): 'Solve the following problem carefully...' "
                            "+ Question + 'Think step by step. Put your final answer "
                            "inside \\\\boxed{}.' (raw text prompt)",
        "chat_template": "NOT applied - tokenizer(build_prompt(record), ...) is called "
                         "directly on the raw build_prompt() string "
                         "(phase3_run_golden_validation.py lines ~736-737). No "
                         "apply_chat_template / <|im_start|> wrapping.",
        "max_new_tokens": cfg.get("max_new_tokens", "?"),
        "temperature": cfg.get("temperature", "?"),
        "top_p": cfg.get("top_p", "(unset; do_sample=False so top_p has no effect)"),
        "num_beams": cfg.get("num_beams", "1 (default, not overridden)"),
        "do_sample": cfg.get("do_sample", "?"),
        "stop_condition": str(cfg.get("stop", "?")),
        "eos_token_id": "tokenizer.eos_token_id plus chat-stop ids "
                         "(see _eos_ids construction in run_inference_transformers)",
        "parser": "extract_answer(): \\\\boxed{} -> 'therefore/answer is' regex -> last non-empty line",
        "output_scores": "False (model.generate() called without output_scores)",
        "return_dict_in_generate": "False (model.generate() called without return_dict_in_generate)",
        "logprob_fields": "none (output_scores/return_dict_in_generate not requested)",
        "mamba_patch_mechanism": "_apply_mamba_patch() called unconditionally before model load "
                                 "(line ~318), regardless of GPU type",
        "validation_source": "--problems file (jsonl/csv) via load_problems()",
        "MAX_PROBLEMS": "--max-problems CLI flag (0 = all)",
        "problem_selection": "problems[:N] via --max-problems (first N in file order, no shuffle)",
        "input_truncation": "_MAX_INPUT_TOKENS = 768 (see _was_truncated flag in output records)",
        "seed": str(seed),
    }


CONFIG_DIFF_ROWS = [
    "adapter_path", "adapter_config", "adapter_model_tensor_count", "prefix",
    "target_modules", "rank", "dtype", "base_model_path", "base_model_quantization",
    "prompt_template", "chat_template", "max_new_tokens", "temperature", "top_p",
    "num_beams", "do_sample", "stop_condition", "eos_token_id", "parser",
    "output_scores", "return_dict_in_generate", "logprob_fields",
    "mamba_patch_mechanism", "validation_source", "MAX_PROBLEMS",
    "problem_selection", "input_truncation", "seed",
]


def golden_side_value(row: str, golden_audit: Dict[str, Any]) -> str:
    if row == "adapter_path":
        return golden_audit.get("adapter_dir", "?")
    if row == "adapter_config":
        return json.dumps(golden_audit.get("adapter_config"), ensure_ascii=False)[:300]
    if row == "adapter_model_tensor_count":
        return str(golden_audit.get("tensor_count"))
    if row == "prefix":
        return ", ".join(sorted(golden_audit.get("prefix_counts", {}))[:5])
    if row == "target_modules":
        return ", ".join(golden_audit.get("target_modules_in_weights", []))
    if row == "rank":
        return f"max={golden_audit.get('max_rank')}, histogram={golden_audit.get('rank_histogram')}"
    if row == "dtype":
        return ", ".join(golden_audit.get("dtype_counts", {}))
    # Generation / harness-level rows: b3-nemotron-svd-26042701.ipynb is
    # ADAPTER-WEIGHT-CONVERSION-ONLY (confirmed by direct reading - see module
    # comment above). It has no inference/generation/prompt/parser code, so
    # this row cannot be derived from it.
    return ("N/A - b3-nemotron-svd-26042701.ipynb is adapter-weight-conversion-only "
            "(SVD/key-rename/expert-unfuse/gate-x-merge); it has no inference/"
            "generation/prompt/parser code. The Kaggle submission inference "
            "notebook (Public LB 0.86) is a separate artifact not provided in "
            "this session.")


def write_config_diff(
    phase3_cfg: Dict[str, str],
    old_audit: Dict[str, Any],
    golden_audit: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    rows = []
    for r in CONFIG_DIFF_ROWS:
        rows.append({
            "item": r,
            "phase3_baseline_script": phase3_cfg.get(r, "?"),
            "phase3_executed_notebook": EXECUTED_NOTEBOOK_REF.get(r, "?"),
            "b3_golden_measured_or_unknown": golden_side_value(r, golden_audit),
        })

    # =====================================================================
    # Verdict 1: artifact_mismatch (LoRA adapter weights / Hypothesis 1)
    # =====================================================================
    artifact_reasons: List[str] = []

    # STATIC / BY CONSTRUCTION (true regardless of what this session
    # measures - established by direct reading of
    # bab0532c-b3nemotronsvd26042701.ipynb, see module comment above):
    artifact_reasons.append(
        "STATIC/BY CONSTRUCTION: phase3_baseline_script's default "
        f"ADAPTER_PATH ({phase3_cfg.get('adapter_path')}) and "
        "phase3_executed_notebook's ADAPTER_PATH (identical hardcoded "
        "string) both equal b3-nemotron-svd-26042701.ipynb's ADAPTER_PATH - "
        "i.e. that notebook's CONVERSION INPUT (the pre-SVD trained "
        "adapter). B3's submission.zip is built from "
        "WORKING_ADAPTER_DIR=/kaggle/working/adapter, the OUTPUT of "
        "compress_lora_fast() (per-module SVD via B3_RANK_MAP) + MoE "
        "expert-unfuse (.experts.w1/w2 -> per-expert up_proj/down_proj) + "
        "Mamba in_proj gate/x split-merge + "
        "'base_model.model.model'->'base_model.model.backbone' key rename. "
        "Phase3 (both the script Tasks 4-7 will drive and the notebook that "
        "produced the audited 20% result) therefore loads B3's CONVERSION "
        "INPUT, not B3's submitted (Public LB 0.86) adapter."
    )

    if old_audit.get("error") or not old_audit.get("adapter_model_exists"):
        artifact_reasons.append(
            "MEASURED (this session): old adapter "
            f"({old_audit.get('adapter_dir')}) could not be audited "
            f"({old_audit.get('error', 'adapter_model.safetensors missing')}). "
            "The STATIC finding above stands on its own and does not depend "
            "on this measurement."
        )
    else:
        diffs: List[str] = []
        if old_audit.get("tensor_count") != golden_audit.get("tensor_count"):
            diffs.append(
                f"tensor_count differs (old={old_audit.get('tensor_count')} "
                f"vs B3={golden_audit.get('tensor_count')})"
            )
        old_tm = set(old_audit.get("target_modules_in_weights", []))
        b3_tm = set(golden_audit.get("target_modules_in_weights", []))
        if old_tm != b3_tm:
            diffs.append(f"target_modules differ (old={sorted(old_tm)} vs B3={sorted(b3_tm)})")
        old_prefixes = set(old_audit.get("prefix_counts", {}))
        b3_prefixes = set(golden_audit.get("prefix_counts", {}))
        if old_prefixes != b3_prefixes:
            diffs.append(
                f"tensor key prefixes differ (old={sorted(old_prefixes)} vs "
                f"B3={sorted(b3_prefixes)}, consistent with the key rename)"
            )
        if old_audit.get("max_rank") != golden_audit.get("max_rank"):
            diffs.append(
                f"max_rank differs (old={old_audit.get('max_rank')} vs "
                f"B3={golden_audit.get('max_rank')})"
            )

        old_bad_hits = sum(old_audit.get("b3_bad_pattern_hits", {}).values())
        old_prefix_hits = old_audit.get("old_prefix_count", 0)
        if old_prefix_hits > 0 or old_bad_hits > 0:
            artifact_reasons.append(
                "MEASURED (this session), CONFIRMS STATIC finding: old adapter "
                f"contains {old_prefix_hits} tensor key(s) with the "
                f"pre-rename prefix '{B3_OLD_PREFIX_SUBSTRING}' and "
                f"{old_bad_hits} tensor key(s) matching B3_BAD_PATTERNS "
                "(.experts.w1/w2/w3./.gate_proj./.x_proj.) - i.e. the old "
                "adapter is consistent with being B3's pre-conversion "
                "(conversion-input) artifact."
            )

        if old_audit.get("sha256") and old_audit.get("sha256") == golden_audit.get("sha256"):
            artifact_reasons.append(
                "MEASURED (this session): old adapter and B3 adapter have "
                "IDENTICAL sha256 - UNEXPECTED given the conversion pipeline "
                "(the key rename alone changes every tensor name, so a "
                "converted adapter cannot be byte-identical to its input). "
                "This may indicate --old-adapter in THIS run was pointed at "
                "an already-converted directory rather than Phase3's "
                f"configured ADAPTER_PATH ({phase3_cfg.get('adapter_path')}). "
                "Does not override the STATIC finding above; verify "
                "--old-adapter resolves to that literal path before relying "
                "on this."
            )
        elif diffs:
            artifact_reasons.append(
                "MEASURED (this session), CONSISTENT with STATIC finding: " + "; ".join(diffs)
            )

    # artifact_mismatch is True by construction (the STATIC finding above),
    # independent of what this session's tensor audit measures.
    artifact_mismatch = True

    # =====================================================================
    # Verdict 2: prompt_harness_mismatch (Hypothesis 3-adjacent; chat
    # template / generation-harness divergence between
    # phase3_baseline_script and phase3_executed_notebook)
    # =====================================================================
    prompt_harness_mismatch = True
    harness_reasons = [
        "STATIC/BY CONSTRUCTION: phase3_executed_notebook (the notebook that "
        "produced the audited n=20/30, 20% result) wraps build_prompt(record) "
        "in ChatML before tokenizing - tokenizer.apply_chat_template("
        "[{role: user, content: <prompt>}], add_generation_prompt=True) if "
        "available, else manually "
        "'<|im_start|>user\\n{prompt}\\n<|im_end|>\\n<|im_start|>assistant\\n'. "
        "phase3_baseline_script (driven by phase3_token_budget_compare.py for "
        "Tasks 4-7) instead calls "
        "tokenizer(build_prompt(record), return_tensors='pt') directly on "
        "the RAW prompt text - no chat-template wrapping at all "
        "(phase3_run_golden_validation.py lines ~736-737, confirmed by grep: "
        "no apply_chat_template / <|im_start|> anywhere in that file).",
        "OUT OF SCOPE for this ticket: the task spec's 'ONLY allowed "
        "verification-time changes' are adapter load path / evaluation "
        "subset / max_new_tokens / logprob ON-OFF. Adding chat-template "
        "wrapping to phase3_run_golden_validation.py is not in that list, "
        "and would itself be a second simultaneous variable change (already "
        "prohibited). Tasks 4-7 below therefore run "
        "phase3_run_golden_validation.py AS-IS (raw-text prompt, no "
        "chat-template) on this dimension. This is recorded here as a "
        "candidate follow-up ticket, not actioned now.",
        "Relevance flagged for Section 6 (Diagnosis): "
        "GOLDEN_GENERATION_CONFIG's stop list "
        "(['<|endoftext|>', '<|im_end|>']) includes a ChatML end-of-turn "
        "token. phase3_executed_notebook opens an '<|im_start|>assistant' "
        "turn before generating, so '<|im_end|>' is a meaningful stop "
        "condition there. phase3_baseline_script never opens an assistant "
        "turn (raw-text prompt), so the model must reach '<|endoftext|>' "
        "or max_new_tokens to stop - if this makes phase3_baseline_script "
        "MORE prone to finish_reason=length than phase3_executed_notebook "
        "was, Tasks 5-7's token-budget comparison is run against "
        "phase3_baseline_script as specified, and any such effect will "
        "show up directly in the case-level finish_reason data rather than "
        "being assumed.",
    ]

    # =====================================================================
    # Verdict 3: quantization_mismatch (base-model load precision divergence
    # between phase3_baseline_script and phase3_executed_notebook)
    # =====================================================================
    quantization_mismatch = True
    quantization_reasons = [
        "STATIC/BY CONSTRUCTION: phase3_baseline_script ALWAYS loads the base "
        "model with 4-bit NF4 quantization via bitsandbytes "
        "(BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', "
        "bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)) "
        "and RAISES if bitsandbytes is unavailable or total GPU VRAM < 16.5GB "
        "(phase3_run_golden_validation.py lines ~362-457; explicitly tuned/"
        "commented for Kaggle T4x2, 29.2GB total VRAM). "
        "phase3_executed_notebook instead branches on total GPU VRAM: 4-bit "
        "NF4 only if < 50GB, else loads the base model in FULL bfloat16/fp16 "
        "with NO quantization_config. On RTX Pro 5000 (>= 50GB) it took the "
        "FULL-BF16 branch - i.e. the n=20/30, 20% result was produced WITHOUT "
        "4-bit quantization.",
        "IMPLICATION: if Tasks 4-7 (driven by phase3_baseline_script, "
        "unmodified per the prohibition list) are run on the SAME "
        "RTX-Pro-5000-class GPU (>= 50GB VRAM) that produced the n=20/30, 20% "
        "result, phase3_baseline_script will load the base model in 4-bit NF4 "
        "- a PRECISION CHANGE not present in the original run, IN ADDITION TO "
        "Task 4's intended adapter-path fix. If instead run on a < 50GB GPU "
        "(e.g. Kaggle T4x2), both scripts would agree on 4-bit NF4, but then "
        "Tasks 4-7 run on DIFFERENT HARDWARE than the RTX-Pro-5000 20% "
        "baseline. Either way, Stage 2's absolute accuracy/finish_reason "
        "numbers are NOT a clean 'adapter-path-only' delta versus the "
        "original n=20/30, 20% result.",
        "DOES NOT BLOCK Tasks 6-7's A/B/C max_new_tokens comparison: "
        "quantization is HELD CONSTANT across conditions A (2048), B (4096), "
        "and C (6144) - all three use the SAME phase3_baseline_script on the "
        "SAME hardware, so only max_new_tokens varies between them, as "
        "required. This verdict affects ONLY the interpretation of Stage 2's "
        "absolute numbers relative to the pre-existing n=20/30, 20% baseline, "
        "not the internal A-vs-B-vs-C comparison.",
        "OUT OF SCOPE for this ticket: 'base-model quantization' is not in "
        "the task spec's 'ONLY allowed verification-time changes' list "
        "(adapter load path / evaluation subset / max_new_tokens / logprob "
        "ON-OFF), and changing it would be an additional simultaneous "
        "variable change (already prohibited). phase3_baseline_script's "
        "quantization logic is left AS-IS. Recorded here as a critical "
        "caveat for Section 6 (Diagnosis) and as a candidate follow-up "
        "ticket (e.g. make 4-bit-vs-bf16 conditional on VRAM in "
        "phase3_baseline_script, matching phase3_executed_notebook, as a "
        "SEPARATE single-variable change).",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phase3_vs_golden_config_diff.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["item", "phase3_baseline_script", "phase3_executed_notebook", "b3_golden_measured_or_unknown"],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["# Phase3 vs B3 Golden Config Diff", ""]
    md_lines.append(f"**artifact_mismatch = {artifact_mismatch}**")
    md_lines.append("")
    for r in artifact_reasons:
        md_lines.append(f"- {r}")
    md_lines.append("")
    md_lines.append(
        "> Phase3 (both phase3_run_golden_validation.py's default "
        "ADAPTER_PATH and the executed notebook's ADAPTER_PATH) is loading "
        "B3's PRE-conversion (trained) adapter, not B3's submitted "
        "(Public LB 0.86) adapter. Task 4 changes ADAPTER_PATH to the "
        f"B3-extracted adapter at `{golden_audit.get('adapter_dir')}` so "
        "Tasks 5-7 evaluate against the correct artifact."
    )
    md_lines.append("")
    md_lines.append(f"**prompt_harness_mismatch = {prompt_harness_mismatch}**")
    md_lines.append("")
    for r in harness_reasons:
        md_lines.append(f"- {r}")
    md_lines.append("")
    md_lines.append(f"**quantization_mismatch = {quantization_mismatch}**")
    md_lines.append("")
    for r in quantization_reasons:
        md_lines.append(f"- {r}")
    md_lines.append("")
    md_lines.append(
        "| item | phase3_baseline_script | phase3_executed_notebook | "
        "b3_golden_measured_or_unknown |"
    )
    md_lines.append("|---|---|---|---|")
    for row in rows:
        a = str(row["phase3_baseline_script"]).replace("|", "\\|").replace("\n", " ")
        b = str(row["phase3_executed_notebook"]).replace("|", "\\|").replace("\n", " ")
        c = str(row["b3_golden_measured_or_unknown"]).replace("|", "\\|").replace("\n", " ")
        md_lines.append(f"| {row['item']} | {a} | {b} | {c} |")

    (out_dir / "phase3_vs_golden_config_diff.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {
        "artifact_mismatch": artifact_mismatch,
        "artifact_mismatch_reasons": artifact_reasons,
        "prompt_harness_mismatch": prompt_harness_mismatch,
        "prompt_harness_mismatch_reasons": harness_reasons,
        "quantization_mismatch": quantization_mismatch,
        "quantization_mismatch_reasons": quantization_reasons,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission-zip", default=None, help="Explicit path to B3's submission.zip (skip search)")
    ap.add_argument("--b3-dest", default=DEFAULT_B3_DEST, help="Where to extract the B3 adapter")
    ap.add_argument("--old-adapter", default=DEFAULT_OLD_ADAPTER, help="Phase3's current ADAPTER_PATH")
    ap.add_argument("--output-dir", default="phase3_token_budget_audit", help="Output root directory")
    ap.add_argument("--tensor-count-tolerance", type=float, default=DEFAULT_TENSOR_COUNT_TOLERANCE)
    ap.add_argument("--skip-old-adapter-audit", action="store_true",
                    help="Skip auditing --old-adapter (e.g. if not present in this session)")
    ap.add_argument("--force", action="store_true",
                    help="Do not exit(1) on Task2 expected-value failures; still write all reports")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    audit_dir = out_dir / "phase3_config_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Task 1: locate + extract B3 submission.zip")
    print("=" * 70)
    zip_path, search_log = find_submission_zip(args.submission_zip)
    for line in search_log:
        print(line)

    if zip_path is None:
        status = {
            "stage": "1-2",
            "pass": False,
            "artifact_mismatch": True,
            "reason": "submission.zip not found",
            "search_log": search_log,
        }
        (audit_dir / "STAGE1_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        (audit_dir / "golden_b3_artifact_report.md").write_text(
            "# B3 Golden Adapter Artifact Report\n\n"
            "## Check result: FAIL\n\n"
            "- FAIL: submission.zip not found under any candidate path.\n\n"
            "Search log:\n```\n" + "\n".join(search_log) + "\n```\n",
            encoding="utf-8",
        )
        print("\nSTOP: submission.zip not found. See "
              f"{audit_dir / 'golden_b3_artifact_report.md'}")
        if not args.force:
            sys.exit(1)
        return

    print(f"\nFound: {zip_path}")
    b3_dest = Path(args.b3_dest)
    extract_log = extract_submission_zip(zip_path, b3_dest)
    for line in extract_log:
        print(line)

    print("\n" + "=" * 70)
    print("Task 2: audit B3 extracted adapter")
    print("=" * 70)
    golden_audit = audit_adapter_dir(b3_dest)
    golden_check = check_against_expected(golden_audit, args.tensor_count_tolerance)
    print(json.dumps({k: v for k, v in golden_audit.items() if k != "_per_tensor_rows"}, indent=2, default=str)[:2000])
    print(f"\nCheck: {'PASS' if golden_check['pass'] else 'FAIL'}")
    for f in golden_check["failures"]:
        print(f"  FAIL: {f}")

    (audit_dir / "golden_b3_adapter_audit.json").write_text(
        json.dumps(audit_to_json(golden_audit), indent=2, default=str), encoding="utf-8"
    )
    write_safetensors_summary_csv(golden_audit, audit_dir / "golden_b3_safetensors_summary.csv")
    if golden_audit.get("adapter_config") is not None:
        (audit_dir / "golden_b3_adapter_config.json").write_text(
            json.dumps(golden_audit["adapter_config"], indent=2), encoding="utf-8"
        )
    (audit_dir / "golden_b3_artifact_report.md").write_text(
        format_artifact_report_md(golden_audit, golden_check, "B3 Golden Adapter Artifact Report"),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("Task 3: audit OLD adapter (Phase3 current ADAPTER_PATH) + config diff")
    print("=" * 70)
    if args.skip_old_adapter_audit:
        old_audit: Dict[str, Any] = {"adapter_dir": args.old_adapter, "error": "skipped via --skip-old-adapter-audit",
                                      "adapter_config_exists": False, "adapter_model_exists": False}
        old_check = {"pass": False, "failures": ["audit skipped"]}
    else:
        old_path = Path(args.old_adapter)
        if not old_path.exists():
            old_audit = {"adapter_dir": args.old_adapter, "error": "path does not exist in this session",
                          "adapter_config_exists": False, "adapter_model_exists": False}
            old_check = {"pass": False, "failures": ["old adapter path does not exist"]}
        else:
            old_audit = audit_adapter_dir(old_path)
            # is_b3_converted=False: the OLD adapter is the PRE-conversion
            # (trained) adapter. It is EXPECTED to contain gate_proj/x_proj/
            # .experts.w1-3./the old "base_model.model.model" prefix - those
            # are normal here, not failures.
            old_check = check_against_expected(old_audit, args.tensor_count_tolerance, is_b3_converted=False)

    (audit_dir / "old_adapter_audit.json").write_text(
        json.dumps(audit_to_json(old_audit), indent=2, default=str), encoding="utf-8"
    )
    if "_per_tensor_rows" in old_audit:
        write_safetensors_summary_csv(old_audit, audit_dir / "old_adapter_safetensors_summary.csv")
    (audit_dir / "old_adapter_artifact_report.md").write_text(
        format_artifact_report_md(old_audit, old_check, "Old Adapter (Phase3 ADAPTER_PATH) Artifact Report"),
        encoding="utf-8",
    )

    phase3_cfg = get_phase3_config()
    verdict = write_config_diff(phase3_cfg, old_audit, golden_audit, audit_dir)
    print(f"\nartifact_mismatch = {verdict['artifact_mismatch']}")
    for r in verdict["artifact_mismatch_reasons"]:
        print(f"  - {r}")
    print(f"\nprompt_harness_mismatch = {verdict['prompt_harness_mismatch']}")
    for r in verdict["prompt_harness_mismatch_reasons"]:
        print(f"  - {r}")
    print(f"\nquantization_mismatch = {verdict['quantization_mismatch']}")
    for r in verdict["quantization_mismatch_reasons"]:
        print(f"  - {r}")

    overall_pass = golden_check["pass"]
    status = {
        "stage": "1-3",
        "pass": overall_pass,
        "artifact_mismatch": verdict["artifact_mismatch"],
        "prompt_harness_mismatch": verdict["prompt_harness_mismatch"],
        "quantization_mismatch": verdict["quantization_mismatch"],
        "golden_check": golden_check,
        "old_check": old_check,
        "b3_adapter_dir": str(b3_dest),
        "old_adapter_dir": args.old_adapter,
    }
    (audit_dir / "STAGE1_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    if overall_pass:
        print("RESULT: B3 adapter artifact PASSES expected-value checks. "
              "Proceed to phase3_token_budget_compare.py.")
    else:
        print("RESULT: B3 adapter artifact FAILS expected-value checks (see "
              f"{audit_dir / 'golden_b3_artifact_report.md'}).")
        if not args.force:
            print("STOPPING here as instructed. Re-run with --force to proceed anyway.")
            sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
