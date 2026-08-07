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
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise SystemExit(f"Phase 7 single-node mode requires exactly two visible GPUs, got {torch.cuda.device_count()}")
for index in range(2):
    if "A100" not in torch.cuda.get_device_name(index).upper():
        raise SystemExit(f"Phase 7 expected A100 at logical GPU {index}: {torch.cuda.get_device_name(index)}")
    print(f"Phase 7 logical GPU {index}: {torch.cuda.get_device_name(index)}")
PY

run_id="${1:-$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID:-interactive}_m3_phase7}"
artifact_base="${experiment_root}/artifacts/phase7"
artifact_scope="${SLAKSHNA_PHASE7_ARTIFACT_SCOPE:-}"
if [[ -n "${artifact_scope}" ]]; then
    if [[ ! "${artifact_scope}" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
        echo "Invalid SLAKSHNA_PHASE7_ARTIFACT_SCOPE: ${artifact_scope}" >&2
        exit 2
    fi
    artifact_base="${artifact_base}/${artifact_scope}"
fi
artifact_root="${artifact_base}/${run_id}"
mkdir -p "${artifact_root}/global-0" "${artifact_root}/bootstrap" \
    "${artifact_root}/peer-a" "${artifact_root}/peer-b" "${artifact_root}/evaluation"

allocated_cpus="${SLURM_CPUS_ON_NODE:-$(nproc)}"
allocated_cpus="${allocated_cpus%%(*}"
if [[ ! "${allocated_cpus}" =~ ^[0-9]+$ ]] || (( allocated_cpus < 12 )); then
    echo "Phase 7 requires at least 12 allocated CPUs, got ${allocated_cpus}" >&2
    exit 2
fi
peer_cpus=$(( (allocated_cpus - 4) / 2 ))
if (( peer_cpus < 4 )); then peer_cpus=4; fi
echo "Phase 7 CPU split: allocation=${allocated_cpus}, per_peer=${peer_cpus}"

cargo_home="${experiment_root}/.runtime/cargo"
rustup_home="${experiment_root}/.runtime/rustup"
if [[ -x "${cargo_home}/bin/cargo" ]]; then
    export CARGO_HOME="${cargo_home}"
    export RUSTUP_HOME="${rustup_home}"
    export PATH="${CARGO_HOME}/bin:${PATH}"
fi
command -v cargo >/dev/null 2>&1 || {
    echo "cargo is unavailable; install the workspace-local Rust toolchain first" >&2
    exit 2
}
export CARGO_TARGET_DIR="${experiment_root}/.runtime/cargo-target/slakshna"
mkdir -p "${CARGO_TARGET_DIR}"
module_gcc_runtime="$(dirname "$(realpath "$(gcc -print-file-name=libstdc++.so.6)")")"
rust_ld_library_path=""
IFS=':' read -r -a ld_entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${ld_entries[@]}"; do
    if [[ -z "${entry}" || "${entry}" == "${module_gcc_runtime}" ]]; then continue; fi
    [[ -z "${rust_ld_library_path}" ]] || rust_ld_library_path+=":"
    rust_ld_library_path+="${entry}"
done
libclang_path="/usr/lib64/llvm21/lib64"
test -f "${libclang_path}/libclang.so.21.1.8"

{
    rustc --version
    cargo --version
    echo "slakshna_revision=$(git -C Slakshna rev-parse HEAD)"
    echo "cargo_lock_sha256=$(sha256sum Slakshna/Cargo.lock | awk '{print $1}')"
    echo "placement=single-node-two-gpu"
} > "${artifact_root}/rust-toolchain.txt"

echo "=== Build pinned Slakshna ==="
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u LIBRARY_PATH -u CPATH \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ LIBCLANG_PATH="${libclang_path}" \
    LD_LIBRARY_PATH="${rust_ld_library_path}" \
    cargo build --locked --release --manifest-path Slakshna/Cargo.toml \
    2>&1 | tee "${artifact_root}/cargo-build.log"
rust_binary="${CARGO_TARGET_DIR}/release/iiitd"
test -x "${rust_binary}"

echo "=== Prepare pinned non-IID data and tokenized shards ==="
SLAKSHNA_CPU_LIMIT="${peer_cpus}" python \
    "${experiment_root}/src/phase7/prepare_data.py" \
    --template "${experiment_root}/configs/phase7/client_ten_epochs.yaml" \
    --artifact-root "${artifact_root}" \
    2>&1 | tee "${artifact_root}/prepare-data.log"

echo "=== Create common G0 ==="
CUDA_VISIBLE_DEVICES=0 python "${experiment_root}/src/phase5/create_initial_adapter.py" \
    --config "${artifact_root}/bootstrap/peer-a/resolved-config.yaml" \
    --output "${artifact_root}/global-0/global_adapter.safetensors" \
    --resume-output "${artifact_root}/global-0/global_adapter.pth" \
    --audit-output "${artifact_root}/global-0/initialization-audit.json"

read -r a_p2p a_ws a_api b_p2p b_ws b_api < <(python - <<'PY'
import socket
sockets, ports = [], []
for _ in range(6):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sockets.append(sock)
    ports.append(sock.getsockname()[1])
print(*ports)
for sock in sockets:
    sock.close()
PY
)

epoch_duration="${SLAKSHNA_PHASE7_EPOCH_DURATION:-600}"
if [[ ! "${epoch_duration}" =~ ^[0-9]+$ ]] || (( epoch_duration < 300 )); then
    echo "SLAKSHNA_PHASE7_EPOCH_DURATION must be an integer of at least 300" >&2
    exit 2
fi
a_runtime="${artifact_root}/peer-a/runtime"
b_runtime="${artifact_root}/peer-b/runtime"
a_data="${artifact_root}/peer-a/node-data"
b_data="${artifact_root}/peer-b/node-data"
python "${experiment_root}/src/phase5/prepare_runtime.py" \
    --template "${experiment_root}/configs/phase7/two_node.toml" \
    --bridge "${experiment_root}/src/phase7/ml_bridge.py" \
    --runtime-dir "${a_runtime}" --data-dir "${a_data}" \
    --peer-name peer-a --gpu-id 0 \
    --p2p-port "${a_p2p}" --ws-port "${a_ws}" --api-port "${a_api}" \
    --epoch-duration "${epoch_duration}" --sync-deadline "$((epoch_duration - 30))"

a_pid=""
b_pid=""
stop_pid() {
    local pid="$1"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${pid}" 2>/dev/null; then kill -KILL "${pid}" 2>/dev/null || true; fi
        wait "${pid}" 2>/dev/null || true
    fi
}
cleanup() {
    stop_pid "${b_pid}"
    stop_pid "${a_pid}"
}
trap cleanup EXIT INT TERM

start_peer() {
    local peer_name="$1" runtime_dir="$2" gpu_id="$3" log_path="$4"
    (
        cd "${runtime_dir}"
        exec env \
            LD_LIBRARY_PATH="${rust_ld_library_path}" \
            SLAKSHNA_PHASE7_ARTIFACT_ROOT="${artifact_root}" \
            SLAKSHNA_EXPERIMENT_ROOT="${experiment_root}" \
            SLAKSHNA_PHASE7_RUN_ID="${run_id}" \
            SLAKSHNA_PHASE7_PEER_NAME="${peer_name}" \
            SLAKSHNA_PHASE7_GLOBAL0="${artifact_root}/global-0/global_adapter.safetensors" \
            SLAKSHNA_PHASE7_GLOBAL0_RESUME="${artifact_root}/global-0/global_adapter.pth" \
            SLAKSHNA_CPU_LIMIT="${peer_cpus}" \
            SLAKSHNA_RAY_CONTROL_CPUS=4 \
            CUDA_VISIBLE_DEVICES="${gpu_id}" \
            "${rust_binary}" --config node.toml
    ) > "${log_path}" 2>&1 &
    started_pid=$!
}

wait_api() {
    local pid="$1" url="$2" output="$3"
    for _ in $(seq 1 120); do
        kill -0 "${pid}" 2>/dev/null || return 1
        if curl --silent --fail --connect-timeout 2 "${url}" > "${output}.tmp"; then
            mv "${output}.tmp" "${output}"
            return 0
        fi
        sleep 1
    done
    return 1
}

seconds_until_boundary=$((epoch_duration - ($(date +%s) % epoch_duration)))
if (( seconds_until_boundary < 60 )); then
    echo "Waiting ${seconds_until_boundary}s to avoid a partial first federation interval"
    sleep $((seconds_until_boundary + 1))
fi

echo "=== Start Peer A ==="
started_pid=""
start_peer peer-a "${a_runtime}" 0 "${artifact_root}/peer-a/rust-node.log"
a_pid="${started_pid}"
a_base="http://127.0.0.1:${a_api}"
if ! wait_api "${a_pid}" "${a_base}/status" "${artifact_root}/peer-a/api-status.initial.json"; then
    tail -n 180 "${artifact_root}/peer-a/rust-node.log" >&2
    exit 1
fi
a_endpoint="$(jq -r '.endpoint_id' "${artifact_root}/peer-a/api-status.initial.json")"
[[ -n "${a_endpoint}" && "${a_endpoint}" != null ]]
direct_seed="${a_endpoint}@127.0.0.1:${a_p2p}"

python "${experiment_root}/src/phase5/prepare_runtime.py" \
    --template "${experiment_root}/configs/phase7/two_node.toml" \
    --bridge "${experiment_root}/src/phase7/ml_bridge.py" \
    --runtime-dir "${b_runtime}" --data-dir "${b_data}" \
    --peer-name peer-b --gpu-id 1 \
    --p2p-port "${b_p2p}" --ws-port "${b_ws}" --api-port "${b_api}" \
    --seed-peer "${direct_seed}" \
    --epoch-duration "${epoch_duration}" --sync-deadline "$((epoch_duration - 30))"

echo "=== Start Peer B with Peer A's direct address ==="
start_peer peer-b "${b_runtime}" 1 "${artifact_root}/peer-b/rust-node.log"
b_pid="${started_pid}"
b_base="http://127.0.0.1:${b_api}"
if ! wait_api "${b_pid}" "${b_base}/status" "${artifact_root}/peer-b/api-status.initial.json"; then
    tail -n 180 "${artifact_root}/peer-b/rust-node.log" >&2
    exit 1
fi
b_endpoint="$(jq -r '.endpoint_id' "${artifact_root}/peer-b/api-status.initial.json")"
[[ -n "${b_endpoint}" && "${b_endpoint}" != null ]]

host="$(hostname -s)"
jq -n \
    --arg job_id "${SLURM_JOB_ID:-interactive}" --arg run_id "${run_id}" \
    --arg host "${host}" --arg a_endpoint "${a_endpoint}" --arg b_endpoint "${b_endpoint}" \
    --arg seed "${direct_seed}" --argjson a_p2p "${a_p2p}" --argjson b_p2p "${b_p2p}" \
    '{schema_version:1,placement_mode:"single-node-two-gpu",slurm_job_id:$job_id,
      slurm_job_num_nodes:1,run_id:$run_id,
      nodes:{"peer-a":{hostname:$host,ip:"127.0.0.1",endpoint_id:$a_endpoint,p2p_port:$a_p2p},
             "peer-b":{hostname:$host,ip:"127.0.0.1",endpoint_id:$b_endpoint,p2p_port:$b_p2p}},
      direct_seed:$seed,discovery:{mdns:false,dht:false,dns:false,relay:false}}' \
    > "${artifact_root}/cluster-topology.json"
CUDA_VISIBLE_DEVICES=0 python "${experiment_root}/src/phase6/node_probe.py" metadata \
    --output "${artifact_root}/peer-a/node-execution.json" --peer-name peer-a \
    --node-ip 127.0.0.1 --seed-peer none --gpu-id 0
CUDA_VISIBLE_DEVICES=1 python "${experiment_root}/src/phase6/node_probe.py" metadata \
    --output "${artifact_root}/peer-b/node-execution.json" --peer-name peer-b \
    --node-ip 127.0.0.1 --seed-peer "${direct_seed}" --gpu-id 1

echo "=== Require bidirectional mesh membership ==="
mesh_ready=0
for _ in $(seq 1 120); do
    if ! kill -0 "${a_pid}" 2>/dev/null || ! kill -0 "${b_pid}" 2>/dev/null; then
        echo "A Phase 7 peer exited while forming the mesh" >&2
        exit 1
    fi
    curl --silent --fail "${a_base}/peers" > "${artifact_root}/peer-a/api-peers.tmp.json" || true
    curl --silent --fail "${b_base}/peers" > "${artifact_root}/peer-b/api-peers.tmp.json" || true
    if jq -e --arg id "${b_endpoint}" '.connected == [$id]' "${artifact_root}/peer-a/api-peers.tmp.json" >/dev/null 2>&1 && \
       jq -e --arg id "${a_endpoint}" '.connected == [$id]' "${artifact_root}/peer-b/api-peers.tmp.json" >/dev/null 2>&1; then
        mesh_ready=1
        break
    fi
    sleep 1
done
[[ "${mesh_ready}" == 1 ]] || { echo "Phase 7 mesh did not form" >&2; exit 1; }

echo "=== Wait for five training rounds and final aggregation ==="
history_ready=0
for poll in $(seq 1 2880); do
    if ! kill -0 "${a_pid}" 2>/dev/null || ! kill -0 "${b_pid}" 2>/dev/null; then
        echo "A Phase 7 peer exited before completion" >&2
        exit 1
    fi
    if grep -Eq "Python ML Engine failed|Failed to start Python process" \
        "${artifact_root}/peer-a/rust-node.log" "${artifact_root}/peer-b/rust-node.log"; then
        tail -n 200 "${artifact_root}/peer-a/rust-node.log" >&2
        tail -n 200 "${artifact_root}/peer-b/rust-node.log" >&2
        exit 1
    fi
    curl --silent --fail "${a_base}/updates" > "${artifact_root}/peer-a/api-updates.tmp.json" || true
    curl --silent --fail "${b_base}/updates" > "${artifact_root}/peer-b/api-updates.tmp.json" || true
    if (( poll % 3 == 0 )); then
        a_round="$(jq -r '.training_rounds_completed // 0' "${artifact_root}/peer-a/bridge-state.json" 2>/dev/null || echo 0)"
        b_round="$(jq -r '.training_rounds_completed // 0' "${artifact_root}/peer-b/bridge-state.json" 2>/dev/null || echo 0)"
        echo "Phase 7 progress: elapsed=$((poll * 10))s rounds(A/B)=${a_round}/${b_round}"
    fi
    if jq -e '([.updates[].kind.ModelUpdate?|select(.!=null)]|length==12) and ([.updates[].kind.PeerReview?|select(.!=null)]|length==20)' "${artifact_root}/peer-a/api-updates.tmp.json" >/dev/null 2>&1 && \
       jq -e '([.updates[].kind.ModelUpdate?|select(.!=null)]|length==12) and ([.updates[].kind.PeerReview?|select(.!=null)]|length==20)' "${artifact_root}/peer-b/api-updates.tmp.json" >/dev/null 2>&1 && \
       jq -e '.finalized == true' "${artifact_root}/peer-a/bridge-state.json" >/dev/null 2>&1 && \
       jq -e '.finalized == true' "${artifact_root}/peer-b/bridge-state.json" >/dev/null 2>&1; then
        history_ready=1
        break
    fi
    sleep 10
done
[[ "${history_ready}" == 1 ]] || { echo "Phase 7 did not complete in eight hours" >&2; exit 1; }

for peer_name in peer-a peer-b; do
    if [[ "${peer_name}" == peer-a ]]; then api_base="${a_base}"; else api_base="${b_base}"; fi
    mv "${artifact_root}/${peer_name}/api-updates.tmp.json" "${artifact_root}/${peer_name}/api-updates.json"
    curl --silent --fail "${api_base}/status" > "${artifact_root}/${peer_name}/api-status.json"
    curl --silent --fail "${api_base}/peers" > "${artifact_root}/${peer_name}/api-peers.json"
done

echo "=== Stop peers after G5 is complete ==="
stop_pid "${b_pid}"; b_pid=""
stop_pid "${a_pid}"; a_pid=""
trap - EXIT INT TERM

echo "=== Evaluate G0 through G5 in fresh processes ==="
a_validation="$(jq -r '.peers["peer-a"].validation.path' "${artifact_root}/data-manifest.json")"
b_validation="$(jq -r '.peers["peer-b"].validation.path' "${artifact_root}/data-manifest.json")"
eval_config="${artifact_root}/bootstrap/peer-a/resolved-config.yaml"
for global_number in $(seq 0 5); do
    if (( global_number == 0 )); then
        adapter="${artifact_root}/global-0/global_adapter.safetensors"
        resume="${artifact_root}/global-0/global_adapter.pth"
    else
        adapter="${artifact_root}/peer-a/global-${global_number}/global_adapter.safetensors"
        resume="${artifact_root}/peer-a/global-${global_number}/global_adapter.pth"
    fi
    eval_args=(
        --config "${eval_config}" --adapter "${adapter}" --adapter-resume "${resume}"
        --peer-a-validation "${a_validation}" --peer-b-validation "${b_validation}"
        --global-number "${global_number}"
        --output "${artifact_root}/evaluation/global-${global_number}.json"
    )
    if (( global_number == 5 )); then eval_args+=(--generate); fi
    CUDA_VISIBLE_DEVICES=0 python "${experiment_root}/src/phase7/evaluate_adapter.py" "${eval_args[@]}" \
        2>&1 | tee "${artifact_root}/evaluation/global-${global_number}.log"
done

sleep 3
python "${experiment_root}/src/phase6/node_probe.py" clean \
    --job-id "${SLURM_JOB_ID:-interactive}" --output "${artifact_root}/peer-a/process-residue.json"
cp "${artifact_root}/peer-a/process-residue.json" "${artifact_root}/peer-b/process-residue.json"

if [[ -n "$(git -C Slakshna status --short)" ]]; then
    echo "Slakshna submodule became dirty" >&2
    git -C Slakshna status --short >&2
    exit 1
fi
{
    echo "revision=$(git -C Slakshna rev-parse HEAD)"
    echo "status=clean"
} > "${artifact_root}/source-integrity.txt"

python "${experiment_root}/src/phase7/verify_phase7.py" \
    --artifact-root "${artifact_root}" --output "${artifact_root}/verification-summary.json"
printf '%s\n' "${artifact_root}" > "${artifact_base}/latest-${SLURM_JOB_ID:-interactive}.txt"
echo "Phase 7 artifact: ${artifact_root}"
