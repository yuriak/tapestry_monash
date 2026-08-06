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
    raise SystemExit("Phase 2 requires a CUDA GPU")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"Phase 2 requires exactly one visible GPU, got {torch.cuda.device_count()}")
print(f"Phase 2 GPU: {torch.cuda.get_device_name(0)}")
PY

run_id="${1:-$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID:-interactive}_phase2}"
artifact_root="${experiment_root}/artifacts/phase2/${run_id}"
local_dir="${artifact_root}/local-training"
update_dir="${artifact_root}/update"
continue_dir="${artifact_root}/continue-training"
initial_adapter="${artifact_root}/initial_adapter.safetensors"
continuation_initial="${artifact_root}/continuation_initial_adapter.safetensors"
mkdir -p "${local_dir}" "${update_dir}" "${continue_dir}"

python "${experiment_root}/src/phase1/prepare_experiment.py" \
    --template "${experiment_root}/configs/phase1/phase1a_single_gpu.yaml" \
    --run-dir "${local_dir}" \
    --checkpoint-dir "${local_dir}/checkpoints" \
    --run-name "${run_id}-local"

python "${experiment_root}/src/phase1/launch_training.py" \
    --config "${local_dir}/resolved-config.yaml" \
    --num-workers 1 \
    --run-dir "${local_dir}" \
    --capture-initial-adapter "${initial_adapter}" \
    2>&1 | tee "${local_dir}/train.log"

python "${experiment_root}/src/phase1/verify_training.py" \
    --mode phase1a \
    --run-dir "${local_dir}" \
    --expected-workers 1 \
    --expected-final-step 4

trained_adapter="${local_dir}/checkpoints/step_0000004/adapter_model.safetensors"
test -f "${trained_adapter}"

python "${experiment_root}/src/phase2/adapter_delta.py" create \
    --initial "${initial_adapter}" \
    --trained "${trained_adapter}" \
    --config "${local_dir}/resolved-config.yaml" \
    --output-dir "${update_dir}"

python "${experiment_root}/src/phase1/prepare_experiment.py" \
    --template "${experiment_root}/configs/phase2/continue_one_step.yaml" \
    --run-dir "${continue_dir}" \
    --checkpoint-dir "${continue_dir}/checkpoints" \
    --run-name "${run_id}-continue" \
    --lora-resume-path "${update_dir}/applied_adapter.pth"

python "${experiment_root}/src/phase1/launch_training.py" \
    --config "${continue_dir}/resolved-config.yaml" \
    --num-workers 1 \
    --run-dir "${continue_dir}" \
    --capture-initial-adapter "${continuation_initial}" \
    2>&1 | tee "${continue_dir}/train.log"

python "${experiment_root}/src/phase1/verify_training.py" \
    --mode phase1a \
    --run-dir "${continue_dir}" \
    --expected-workers 1 \
    --expected-final-step 1

python "${experiment_root}/src/phase2/adapter_delta.py" finalize \
    --update-dir "${update_dir}" \
    --trained "${trained_adapter}" \
    --continuation-initial "${continuation_initial}" \
    --continuation-summary "${continue_dir}/verification-summary.json" \
    --output "${artifact_root}/verification-summary.json"

echo "Phase 2 artifact: ${artifact_root}"
