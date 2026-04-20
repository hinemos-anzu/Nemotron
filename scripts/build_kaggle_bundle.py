#!/usr/bin/env python3
"""
scripts/build_kaggle_bundle.py
Builds artifacts/kaggle/Nemotron_kaggle_bundle/ and its ZIP.
"""
import csv, json, os, shutil, zipfile
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).parent.parent.resolve()
BUNDLE_ROOT = REPO / "artifacts" / "kaggle" / "Nemotron_kaggle_bundle"
ARTIFACTS   = REPO / "artifacts" / "kaggle"

# ── commit SHAs captured from git log ──────────────────────────────────────
SHAS = {
    "scripts/debug_harness.py":                          "70ef3d91b507f11018cc2259cd56d6d1e14d7247",
    "kaggle/run_baseline_with_debug.py":                 "70ef3d91b507f11018cc2259cd56d6d1e14d7247",
    "kaggle/original-nemotron-asymmetric-svd-26041602.py":"70ef3d91b507f11018cc2259cd56d6d1e14d7247",
    "data/eval/quick_gate_v1.jsonl":                     "9a3046f1e42560667a615c921814bf6ec29a398e",
    "data/eval/diagnostic_v1.jsonl":                     "9a3046f1e42560667a615c921814bf6ec29a398e",
    "data/eval/promotion_v1.jsonl":                      "9a3046f1e42560667a615c921814bf6ec29a398e",
    "data/eval/category_manifest_v1.csv":                "9a3046f1e42560667a615c921814bf6ec29a398e",
    "reports/eval/baseline_reference_v1.md":             "9a3046f1e42560667a615c921814bf6ec29a398e",
    "tickets/TICKET_S1_5_baseline_measured_reference_v1.md":
        "6193e3eaa9214558b30306ceea0154ba270ae6d6",
    "tickets/TICKET_S1_6A_kaggle_runtime_execution_enablement_for_baseline_path.md":
        "6e778d3263cd20b4c21a2b3d9a421323c9376689",
}
SOURCE_BRANCHES = {
    "scripts/debug_harness.py":                           "planner/debug-output-hardening-v1",
    "kaggle/run_baseline_with_debug.py":                  "planner/debug-output-hardening-v1",
    "kaggle/original-nemotron-asymmetric-svd-26041602.py":"planner/debug-output-hardening-v1",
    "data/eval/quick_gate_v1.jsonl":                      "claude/nemotron-experiment-framework-7K8yG",
    "data/eval/diagnostic_v1.jsonl":                      "claude/nemotron-experiment-framework-7K8yG",
    "data/eval/promotion_v1.jsonl":                       "claude/nemotron-experiment-framework-7K8yG",
    "data/eval/category_manifest_v1.csv":                 "claude/nemotron-experiment-framework-7K8yG",
    "reports/eval/baseline_reference_v1.md":              "claude/nemotron-experiment-framework-7K8yG",
    "tickets/TICKET_S1_5_baseline_measured_reference_v1.md":
        "claude/nemotron-experiment-framework-7K8yG",
    "tickets/TICKET_S1_6A_kaggle_runtime_execution_enablement_for_baseline_path.md":
        "claude/nemotron-experiment-framework-7K8yG",
}


# ── step 1: JSONL → CSV conversion ─────────────────────────────────────────

def jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> int:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not rows:
        return 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)

eval_dir = REPO / "data" / "eval"
conversions = [
    ("quick_gate_v1.jsonl",  "quick_gate_v1.csv"),
    ("diagnostic_v1.jsonl",  "diagnostic_v1.csv"),
    ("promotion_v1.jsonl",   "promotion_v1.csv"),
]
for src_name, dst_name in conversions:
    n = jsonl_to_csv(eval_dir / src_name, eval_dir / dst_name)
    print(f"  converted {src_name} → {dst_name}  ({n} rows)")


# ── step 2: placeholder reports for files not yet generated ────────────────

def write_placeholder(path: Path, title: str, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {title}\n\n"
        f"**Status:** PENDING\n\n"
        f"**Reason:** {reason}\n\n"
        "This file is a placeholder.  "
        "It will be replaced with real content after the corresponding ticket is executed.\n",
        encoding="utf-8",
    )

placeholder_reports = {
    "reports/eval/baseline_reference_v1_measured.md": (
        "Baseline Reference v1 — Measured",
        "Requires TICKET_S1_5 model inference run. "
        "Execute `python kaggle/run_baseline_with_debug.py` on the Kaggle source-of-truth "
        "environment and populate this report with measured accuracy values.",
    ),
    "reports/eval/baseline_reference_v1_measured_kaggle.md": (
        "Baseline Reference v1 — Measured (Kaggle Source-of-Truth)",
        "Requires TICKET_S1_6 / S1_6A Kaggle environment execution. "
        "Must be populated after Kaggle execution role runs the baseline path "
        "and returns per-sample results.",
    ),
}
for rel_path, (title, reason) in placeholder_reports.items():
    full = REPO / rel_path
    if not full.exists():
        write_placeholder(full, title, reason)
        print(f"  created placeholder: {rel_path}")
    else:
        print(f"  already exists: {rel_path}")


# ── step 3: build bundle directory ─────────────────────────────────────────

if BUNDLE_ROOT.exists():
    shutil.rmtree(BUNDLE_ROOT)
BUNDLE_ROOT.mkdir(parents=True)

# (repo_relative_src, bundle_relative_dst)
BUNDLE_FILES = [
    # required
    ("scripts/debug_harness.py",                               "scripts/debug_harness.py"),
    ("kaggle/run_baseline_with_debug.py",                      "kaggle/run_baseline_with_debug.py"),
    ("kaggle/original-nemotron-asymmetric-svd-26041602.py",    "kaggle/original-nemotron-asymmetric-svd-26041602.py"),
    ("data/eval/quick_gate_v1.csv",                            "data/eval/quick_gate_v1.csv"),
    ("data/eval/diagnostic_v1.csv",                            "data/eval/diagnostic_v1.csv"),
    ("data/eval/promotion_v1.csv",                             "data/eval/promotion_v1.csv"),
    ("data/eval/category_manifest_v1.csv",                     "data/eval/category_manifest_v1.csv"),
    # optional / reference
    ("reports/eval/baseline_reference_v1.md",                  "reports/eval/baseline_reference_v1.md"),
    ("reports/eval/baseline_reference_v1_measured.md",         "reports/eval/baseline_reference_v1_measured.md"),
    ("reports/eval/baseline_reference_v1_measured_kaggle.md",  "reports/eval/baseline_reference_v1_measured_kaggle.md"),
    ("tickets/TICKET_S1_5_baseline_measured_reference_v1.md",
     "tickets/TICKET_S1_5_baseline_measured_reference_v1.md"),
    ("tickets/TICKET_S1_6A_kaggle_runtime_execution_enablement_for_baseline_path.md",
     "tickets/TICKET_S1_6A_kaggle_runtime_execution_enablement_for_baseline_path.md"),
]

for repo_rel, bundle_rel in BUNDLE_FILES:
    src = REPO / repo_rel
    dst = BUNDLE_ROOT / bundle_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  bundle: {bundle_rel}")


# ── step 4: README_KAGGLE_BUNDLE.md ────────────────────────────────────────

readme_bundle = """# Nemotron Kaggle Bundle
**Version:** v1
**Generated:** {ts}
**Repository:** hinemos-anzu/Nemotron
**Branch:** planner/kaggle-bundle-materialization-v1

---

## Purpose

This bundle provides all files required for the Kaggle execution role to run the
Nemotron baseline evaluation path in the Kaggle source-of-truth environment.
The bundle is designed to be uploaded as a Kaggle Dataset and mounted without
any additional file-finding steps by the execution role.

---

## Key Files

| Role | Path in Bundle |
|---|---|
| **Entrypoint** (run this) | `kaggle/run_baseline_with_debug.py` |
| Baseline script (do not change logic) | `kaggle/original-nemotron-asymmetric-svd-26041602.py` |
| Debug harness library | `scripts/debug_harness.py` |
| Eval set — Quick Gate v1 | `data/eval/quick_gate_v1.csv` |
| Eval set — Diagnostic v1 | `data/eval/diagnostic_v1.csv` |
| Eval set — Promotion v1 | `data/eval/promotion_v1.csv` |
| Category manifest v1 | `data/eval/category_manifest_v1.csv` |
| Baseline reference report | `reports/eval/baseline_reference_v1.md` |
| Execution ticket | `tickets/TICKET_S1_6A_kaggle_runtime_execution_enablement_for_baseline_path.md` |

---

## Kaggle Notebook — Assumed Execution Commands

```bash
# 1. Set working directory to bundle root
cd /kaggle/working/Nemotron

# 2. Run entrypoint (Quick Gate set, default)
python kaggle/run_baseline_with_debug.py --eval-set quick_gate_v1

# 3. Or run specific eval set
python kaggle/run_baseline_with_debug.py --eval-set diagnostic_v1
python kaggle/run_baseline_with_debug.py --eval-set promotion_v1
```

---

## Script Roles

- **`kaggle/run_baseline_with_debug.py`**
  Entry point. Initialises `KaggleRunHarness`, dumps environment info,
  runs the baseline script as a subprocess with stdout/stderr tee,
  and writes all 9 debug report files automatically.

- **`kaggle/original-nemotron-asymmetric-svd-26041602.py`**
  Baseline execution script — original Nemotron asymmetric-SVD submission path.
  PROTECTED ASSET: do not change conversion flow, SVD surgery, key rename,
  merge logic, expert unfuse, or submission.zip generation.
  Currently contains a placeholder body; Kaggle execution role inserts real
  notebook content at the marked section.

- **`scripts/debug_harness.py`**
  Debug harness library. Provides `KaggleRunHarness`, `StageLogger`,
  `EnvironmentDumper`, `InputManifestWriter`, `OutputArtifactTracker`,
  `ErrorReporter`, `ReproStepsWriter`, `RunSummaryWriter`.
  On every run (success or failure) it auto-generates:
  `reports/kaggle_run/run_summary.md`,
  `reports/kaggle_run/environment_info.md`,
  `reports/kaggle_run/error_report.md`,
  `logs/kaggle_run/stage_log.jsonl`, and more.

---

## Debug Output Location After Run

| File | Location |
|---|---|
| Run summary | `reports/kaggle_run/run_summary.md` |
| Environment info | `reports/kaggle_run/environment_info.md` |
| Error report | `reports/kaggle_run/error_report.md` |
| Stage log | `logs/kaggle_run/stage_log.jsonl` |
| Stdout | `logs/kaggle_run/stdout.log` |
| Stderr | `logs/kaggle_run/stderr.log` |

Return all files under `reports/kaggle_run/` and `logs/kaggle_run/` to Generator
on failure, or on success along with submission artifacts.
""".format(ts=datetime.now(timezone.utc).isoformat(timespec="seconds"))

(BUNDLE_ROOT / "README_KAGGLE_BUNDLE.md").write_text(readme_bundle, encoding="utf-8")
print("  created README_KAGGLE_BUNDLE.md")


# ── step 5: create ZIP ──────────────────────────────────────────────────────

zip_path = ARTIFACTS / "Nemotron_kaggle_bundle.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fpath in sorted(BUNDLE_ROOT.rglob("*")):
        if fpath.is_file():
            arcname = "Nemotron_kaggle_bundle/" + str(fpath.relative_to(BUNDLE_ROOT))
            zf.write(fpath, arcname)
            print(f"  zip: {arcname}")

# Verify root folder presence
with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    has_root = any(n.startswith("Nemotron_kaggle_bundle/") for n in names)
    print(f"\n  ZIP entries: {len(names)}, root folder present: {has_root}")
    assert has_root, "ZIP root folder missing!"

print(f"\n  ZIP written: {zip_path}  ({zip_path.stat().st_size:,} bytes)")


# ── step 6: inventory for manifest ─────────────────────────────────────────

REQUIRED_FOR_RUN = {
    "scripts/debug_harness.py": True,
    "kaggle/run_baseline_with_debug.py": True,
    "kaggle/original-nemotron-asymmetric-svd-26041602.py": True,
    "data/eval/quick_gate_v1.csv": True,
    "data/eval/diagnostic_v1.csv": False,
    "data/eval/promotion_v1.csv": False,
    "data/eval/category_manifest_v1.csv": True,
    "reports/eval/baseline_reference_v1.md": False,
    "reports/eval/baseline_reference_v1_measured.md": False,
    "reports/eval/baseline_reference_v1_measured_kaggle.md": False,
    "tickets/TICKET_S1_5_baseline_measured_reference_v1.md": False,
    "tickets/TICKET_S1_6A_kaggle_runtime_execution_enablement_for_baseline_path.md": False,
}

manifest_rows = []
missing_files = []

for repo_rel, bundle_rel in BUNDLE_FILES:
    src = REPO / repo_rel
    src_exists = src.exists()
    bundle_exists = (BUNDLE_ROOT / bundle_rel).exists()

    # Use source branch SHA; CSV files share SHA with their JSONL source
    sha_key = repo_rel
    if repo_rel.endswith(".csv") and repo_rel.startswith("data/eval/") and "manifest" not in repo_rel:
        sha_key = repo_rel.replace(".csv", ".jsonl")
    sha = SHAS.get(sha_key, "GENERATED_THIS_BRANCH")
    source_branch = SOURCE_BRANCHES.get(sha_key, "planner/kaggle-bundle-materialization-v1")

    if not src_exists:
        missing_files.append(repo_rel)

    manifest_rows.append({
        "logical_name": Path(repo_rel).name,
        "repo_path": repo_rel,
        "bundle_path": f"Nemotron_kaggle_bundle/{bundle_rel}",
        "repository": "hinemos-anzu/Nemotron",
        "branch": source_branch,
        "commit_sha": sha,
        "exists_on_github": "YES" if SHAS.get(sha_key) else "YES_PLACEHOLDER",
        "included_in_bundle": "YES" if bundle_exists else "NO",
        "included_in_zip": "YES" if bundle_exists else "NO",
        "required_for_run": "YES" if REQUIRED_FOR_RUN.get(repo_rel) else "NO",
        "notes": (
            "converted from .jsonl" if repo_rel.endswith(".csv") and repo_rel.startswith("data/eval/") and "manifest" not in repo_rel
            else "placeholder — PENDING" if "measured" in repo_rel
            else ""
        ),
    })

# ── step 7: write manifest.csv ──────────────────────────────────────────────

manifest_csv_path = ARTIFACTS / "Nemotron_kaggle_bundle_manifest.csv"
fields = ["logical_name","repo_path","bundle_path","repository","branch",
          "commit_sha","exists_on_github","included_in_bundle","included_in_zip",
          "required_for_run","notes"]
with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(manifest_rows)
print(f"\n  manifest.csv: {len(manifest_rows)} rows")

# Print manifest for record
for r in manifest_rows:
    print(f"  {r['repo_path']:60s} sha={r['commit_sha'][:12]}  exist={r['exists_on_github']}")

print("\nBuild complete.")
