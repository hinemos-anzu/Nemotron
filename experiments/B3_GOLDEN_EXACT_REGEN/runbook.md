# B3_GOLDEN_EXACT_REGEN Runbook

This runbook prepares the later Golden exact execution. It is intentionally written as a controlled procedure; do not execute artifact-generating steps until explicitly approved.

## Phase 0: Preparation-only status for this PR

Status for this change:

- Adapter regeneration: **not run**
- Root `submission.zip` regeneration: **not run**
- Kaggle submission: **not run**
- SFT / cryptarithm patch creation: **not run**
- Rank/SVD/weight surgery: **not run**

## Phase 1: Preflight documentation check

From the repository root, verify that the preparation files exist:

```bash
python - <<'PY'
from pathlib import Path
required = [
    'experiments/B3_GOLDEN_EXACT_REGEN/README.md',
    'experiments/B3_GOLDEN_EXACT_REGEN/preflight_checklist.md',
    'experiments/B3_GOLDEN_EXACT_REGEN/golden_fixed_conditions.md',
    'experiments/B3_GOLDEN_EXACT_REGEN/diagnostics_template.md',
    'experiments/B3_GOLDEN_EXACT_REGEN/runbook.md',
]
for path in required:
    assert Path(path).exists(), path
print('B3_GOLDEN_EXACT_REGEN preparation files present')
PY
```

## Phase 2: Notebook path for Golden-compatible local validation

Use one of the Golden Baseline inference notebooks/scripts without changing the Golden configuration:

1. `phase3_rtx_analysis.ipynb`, generated from `gen_phase3_notebook.py`, for the current self-contained Phase 3 analysis path.
2. `nemotron_rtx_inference.ipynb`, generated from `gen_notebook.py`, for the RTX Golden Baseline inference path.
3. `phase3_run_all.py` only for analysis/validation flows that explicitly report no adapter, training, or submission changes.

Golden default paths to preserve unless the execution environment requires equivalent mounted paths:

```text
ADAPTER_PATH=/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20
MODEL_PATH=/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
PROBLEMS_PATH=/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv
OUTPUT_DIR=/kaggle/working/phase3_analysis
SEED=42
```

Do not edit the Golden generation configuration while running this control.

## Phase 3: Later adapter diagnostics command template

Use this template only after an approved execution phase has a concrete adapter path to inspect. It is a diagnostics command, not a conversion or regeneration command.

```bash
python - <<'PY'
from pathlib import Path
import json
import os
from safetensors.torch import load_file

adapter_path = Path(os.environ['ADAPTER_PATH']) / 'adapter_model.safetensors'
state = load_file(str(adapter_path))

num_tensors = len(state)
expected_rank_ceiling = 32
max_rank_seen = 0
rank_violations = []
non_finite = []
keys = sorted(state)

for key, tensor in state.items():
    if tensor.ndim >= 2 and ('lora_A' in key or 'lora_B' in key):
        rank_dim = min(int(tensor.shape[-1]), int(tensor.shape[-2]))
        max_rank_seen = max(max_rank_seen, rank_dim)
        if rank_dim > expected_rank_ceiling:
            rank_violations.append({'key': key, 'rank_dim': rank_dim})
    if not tensor.isfinite().all().item():
        non_finite.append(key)

report = {
    'gate_x_split_expected': '16/16',
    'num_tensors': num_tensors,
    'max_rank_seen': max_rank_seen,
    'expected_rank_ceiling': expected_rank_ceiling,
    'rank_violation_count': len(rank_violations),
    'rank_violations': rank_violations[:20],
    'non_finite_tensor_count': len(non_finite),
    'non_finite_tensor_keys': non_finite[:20],
    'first_keys': keys[:20],
    'last_keys': keys[-20:],
}
print(json.dumps(report, indent=2, sort_keys=True))
PY
```

Copy the observed values into `diagnostics_template.md` or a filled run-specific diagnostics file.

## Phase 4: Later root archive diagnostics command template

Use this only after explicit approval to generate `submission.zip` in a later phase.

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import json
import zipfile

zip_path = Path('submission.zip')
with zipfile.ZipFile(zip_path) as zf:
    names = sorted(zf.namelist())
sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
print(json.dumps({
    'artifact': str(zip_path),
    'sha256': sha256,
    'num_entries': len(names),
    'entries': names,
}, indent=2))
PY
```

Record the archive SHA256 and contents in the diagnostics template before any Kaggle submission.

## Phase 5: Promotion rule

Promote the regenerated control only if all of the following hold:

- Golden fixed conditions remain unchanged.
- Gate/X split is `16/16`.
- Rank violations are absent.
- NaN/Inf values are absent.
- Adapter keys/modules match Golden expectations.
- Root `submission.zip` contents match Golden packaging expectations, if generated.
- Public LB reproduces `0.86` equivalent after an explicitly approved submission.

If any item fails, reject or hold the control and investigate environment, conversion, adapter, packaging, or submission drift before starting SFT.
