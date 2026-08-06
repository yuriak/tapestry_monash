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
    raise SystemExit("Phase 4 requires a CUDA GPU")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"Phase 4 requires exactly one visible GPU, got {torch.cuda.device_count()}")
print(f"Phase 4 GPU: {torch.cuda.get_device_name(0)}")
PY

run_id="${1:-$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID:-interactive}_phase4}"
artifact_root="${experiment_root}/artifacts/phase4/${run_id}"
mkdir -p "${artifact_root}/global-0"

run_client() {
    local round_number="$1"
    local client_name="$2"
    local row_indices="$3"
    local resume_path="$4"
    local client_dir="${artifact_root}/round-${round_number}/${client_name}"
    mkdir -p "${client_dir}"

    local prepare_args=(
        "${experiment_root}/src/phase1/prepare_experiment.py"
        --template "${experiment_root}/configs/phase4/client_two_steps.yaml"
        --run-dir "${client_dir}"
        --checkpoint-dir "${client_dir}/checkpoints"
        --run-name "${run_id}-round${round_number}-${client_name}"
        --row-indices "${row_indices}"
        --data-label "phase4-${client_name}"
    )
    if [[ -n "${resume_path}" ]]; then
        prepare_args+=(--lora-resume-path "${resume_path}")
    fi
    python "${prepare_args[@]}" 2>&1 | tee "${client_dir}/prepare.log"

    python "${experiment_root}/src/phase1/launch_training.py" \
        --config "${client_dir}/resolved-config.yaml" \
        --num-workers 1 \
        --run-dir "${client_dir}" \
        --capture-initial-adapter "${client_dir}/initial_adapter.safetensors" \
        2>&1 | tee "${client_dir}/train.log"

    python "${experiment_root}/src/phase1/verify_training.py" \
        --mode phase1a \
        --run-dir "${client_dir}" \
        --expected-workers 1 \
        --expected-final-step 2 \
        2>&1 | tee "${client_dir}/verify.log"
}

echo "=== Round 1: client A establishes the shared initialization ==="
run_client 1 client-a "0,2,4,6" ""

python "${experiment_root}/src/phase4/fedavg.py" bridge \
    --adapter "${artifact_root}/round-1/client-a/initial_adapter.safetensors" \
    --canonical-output "${artifact_root}/global-0/global_adapter.safetensors" \
    --output "${artifact_root}/global-0/global_adapter.pth"

echo "=== Round 1: client B starts from the identical initialization ==="
run_client 1 client-b "1,3,5,7" "${artifact_root}/global-0/global_adapter.pth"

python "${experiment_root}/src/phase4/fedavg.py" aggregate \
    --round 1 \
    --base "${artifact_root}/global-0/global_adapter.safetensors" \
    --client client-a \
        "${artifact_root}/round-1/client-a/initial_adapter.safetensors" \
        "${artifact_root}/round-1/client-a/checkpoints/step_0000002/adapter_model.safetensors" 4 \
    --client client-b \
        "${artifact_root}/round-1/client-b/initial_adapter.safetensors" \
        "${artifact_root}/round-1/client-b/checkpoints/step_0000002/adapter_model.safetensors" 4 \
    --output-dir "${artifact_root}/round-1/aggregation"

round1_global="${artifact_root}/round-1/aggregation/global_adapter.pth"
echo "=== Round 2: both clients warm-start from Round 1 FedAvg ==="
run_client 2 client-a "0,2,4,6" "${round1_global}"
run_client 2 client-b "1,3,5,7" "${round1_global}"

python "${experiment_root}/src/phase4/fedavg.py" aggregate \
    --round 2 \
    --base "${artifact_root}/round-1/aggregation/global_adapter.safetensors" \
    --client client-a \
        "${artifact_root}/round-2/client-a/initial_adapter.safetensors" \
        "${artifact_root}/round-2/client-a/checkpoints/step_0000002/adapter_model.safetensors" 4 \
    --client client-b \
        "${artifact_root}/round-2/client-b/initial_adapter.safetensors" \
        "${artifact_root}/round-2/client-b/checkpoints/step_0000002/adapter_model.safetensors" 4 \
    --output-dir "${artifact_root}/round-2/aggregation"

python "${experiment_root}/src/phase4/verify_phase4.py" \
    --artifact-root "${artifact_root}" \
    --output "${artifact_root}/verification-summary.json"

if [[ -n "$(git -C Slakshna status --short)" ]]; then
    echo "Slakshna submodule became dirty" >&2
    git -C Slakshna status --short >&2
    exit 1
fi
{
    echo "revision=$(git -C Slakshna rev-parse HEAD)"
    echo "status=clean"
} > "${artifact_root}/source-integrity.txt"

echo "Phase 4 artifact: ${artifact_root}"
