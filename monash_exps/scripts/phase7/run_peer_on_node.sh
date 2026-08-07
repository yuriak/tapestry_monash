#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 10 ]]; then
    echo "Usage: run_peer_on_node.sh PEER ARTIFACT RUN_ID NODE_IP P2P WS API SEED EPOCH CONFIG" >&2
    exit 2
fi

peer_name="$1"
artifact_root="$2"
run_id="$3"
node_ip="$4"
p2p_port="$5"
ws_port="$6"
api_port="$7"
seed_peer="$8"
epoch_duration="$9"
config_name="${10}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
cd "${workspace_root}"

# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

if [[ "${peer_name}" != peer-a && "${peer_name}" != peer-b ]]; then
    echo "invalid peer name: ${peer_name}" >&2
    exit 2
fi
visible_gpu="${CUDA_VISIBLE_DEVICES:-}"
if [[ ! "${visible_gpu}" =~ ^[0-9]+$ ]]; then
    echo "Phase 7 requires one numeric scheduler GPU id, got CUDA_VISIBLE_DEVICES=${visible_gpu}" >&2
    exit 2
fi
python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"Phase 7 node task requires exactly one visible GPU, got {torch.cuda.device_count()}")
print(f"Phase 7 node GPU: {torch.cuda.get_device_name(0)}")
PY

peer_root="${artifact_root}/${peer_name}"
runtime_dir="${peer_root}/runtime"
data_dir="${peer_root}/node-data"
prepare_args=(
    --template "${experiment_root}/configs/phase7/two_node.toml"
    --bridge "${experiment_root}/src/phase7/ml_bridge.py"
    --runtime-dir "${runtime_dir}"
    --data-dir "${data_dir}"
    --peer-name "${peer_name}"
    --gpu-id "${visible_gpu}"
    --p2p-port "${p2p_port}"
    --ws-port "${ws_port}"
    --api-port "${api_port}"
    --epoch-duration "${epoch_duration}"
    --sync-deadline "$((epoch_duration - 30))"
    --config-name "${config_name}"
)
if [[ "${seed_peer}" != none ]]; then
    prepare_args+=(--seed-peer "${seed_peer}")
fi
python "${experiment_root}/src/phase5/prepare_runtime.py" "${prepare_args[@]}"

python "${experiment_root}/src/phase6/node_probe.py" metadata \
    --output "${peer_root}/node-execution.json" \
    --peer-name "${peer_name}" \
    --node-ip "${node_ip}" \
    --seed-peer "${seed_peer}" \
    --gpu-id "${visible_gpu}"

rust_binary="${experiment_root}/.runtime/cargo-target/slakshna/release/iiitd"
test -x "${rust_binary}"
cd "${runtime_dir}"
exec env \
    LD_LIBRARY_PATH="${SLAKSHNA_RUST_LD_LIBRARY_PATH:-}" \
    SLAKSHNA_PHASE7_ARTIFACT_ROOT="${artifact_root}" \
    SLAKSHNA_EXPERIMENT_ROOT="${experiment_root}" \
    SLAKSHNA_PHASE7_RUN_ID="${run_id}" \
    SLAKSHNA_PHASE7_PEER_NAME="${peer_name}" \
    SLAKSHNA_PHASE7_GLOBAL0="${artifact_root}/global-0/global_adapter.safetensors" \
    SLAKSHNA_PHASE7_GLOBAL0_RESUME="${artifact_root}/global-0/global_adapter.pth" \
    SLAKSHNA_CPU_LIMIT="${SLURM_CPUS_PER_TASK:-20}" \
    SLAKSHNA_RAY_CONTROL_CPUS=4 \
    "${rust_binary}" --config "${config_name}"
