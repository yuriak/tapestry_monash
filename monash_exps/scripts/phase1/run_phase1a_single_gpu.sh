#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
cd "${workspace_root}"

# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Phase 1A requires a CUDA GPU in the interactive allocation")
count = torch.cuda.device_count()
if count != 1:
    raise SystemExit(f"Phase 1A requires exactly one visible GPU, got {count}")
print(f"Phase 1A interactive GPU: {torch.cuda.get_device_name(0)}")
PY

run_id="${1:-$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID:-interactive}_phase1a}"
run_dir="${experiment_root}/artifacts/phase1/${run_id}/phase1a-single-gpu"
checkpoint_dir="${run_dir}/checkpoints"
mkdir -p "${run_dir}"

python "${experiment_root}/src/phase1/prepare_experiment.py" \
    --template "${experiment_root}/configs/phase1/phase1a_single_gpu.yaml" \
    --run-dir "${run_dir}" \
    --checkpoint-dir "${checkpoint_dir}" \
    --run-name "${run_id}-phase1a"

python "${experiment_root}/src/phase1/launch_training.py" \
    --config "${run_dir}/resolved-config.yaml" \
    --num-workers 1 \
    --run-dir "${run_dir}" \
    2>&1 | tee "${run_dir}/train.log"

python "${experiment_root}/src/phase1/verify_training.py" \
    --mode phase1a \
    --run-dir "${run_dir}" \
    --expected-workers 1

echo "Phase 1A artifact: ${run_dir}"
