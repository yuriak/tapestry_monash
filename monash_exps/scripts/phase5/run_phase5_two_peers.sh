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
    raise SystemExit("Phase 5 requires CUDA")
if torch.cuda.device_count() != 2:
    raise SystemExit(f"Phase 5 requires exactly two visible GPUs, got {torch.cuda.device_count()}")
print("Phase 5 GPUs:")
for index in range(2):
    print(f"  logical {index}: {torch.cuda.get_device_name(index)}")
PY

cargo_home="${experiment_root}/.runtime/cargo"
rustup_home="${experiment_root}/.runtime/rustup"
if [[ -x "${cargo_home}/bin/cargo" ]]; then
    export CARGO_HOME="${cargo_home}"
    export RUSTUP_HOME="${rustup_home}"
    export PATH="${CARGO_HOME}/bin:${PATH}"
fi
if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo is unavailable. Run monash_exps/scripts/environment/06_install_rust.sh first." >&2
    exit 2
fi

run_id="${1:-$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID:-interactive}_phase5}"
artifact_root="${experiment_root}/artifacts/phase5/${run_id}"
mkdir -p "${artifact_root}/global-0" "${artifact_root}/bootstrap"

allocated_cpus="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc)}}"
allocated_cpus="${allocated_cpus%%(*}"
if [[ ! "${allocated_cpus}" =~ ^[0-9]+$ ]] || (( allocated_cpus < 8 )); then
    echo "Phase 5 requires at least 8 allocated CPUs, got ${allocated_cpus}" >&2
    exit 2
fi
peer_cpus=$(( (allocated_cpus - 2) / 2 ))
if (( peer_cpus < 3 )); then
    peer_cpus=3
fi
echo "Phase 5 CPU split: allocation=${allocated_cpus}, per_peer=${peer_cpus}"

export CARGO_TARGET_DIR="${experiment_root}/.runtime/cargo-target/slakshna"
mkdir -p "${CARGO_TARGET_DIR}"
module_gcc_runtime="$(dirname "$(realpath "$(gcc -print-file-name=libstdc++.so.6)")")"
rust_ld_library_path=""
IFS=':' read -r -a ld_entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${ld_entries[@]}"; do
    if [[ -z "${entry}" || "${entry}" == "${module_gcc_runtime}" ]]; then
        continue
    fi
    if [[ -n "${rust_ld_library_path}" ]]; then
        rust_ld_library_path+=":"
    fi
    rust_ld_library_path+="${entry}"
done
libclang_path="/usr/lib64/llvm21/lib64"
test -f "${libclang_path}/libclang.so.21.1.8"

{
    rustc --version
    cargo --version
    echo "cargo_target_dir=${CARGO_TARGET_DIR}"
    echo "slakshna_revision=$(git -C Slakshna rev-parse HEAD)"
    echo "cargo_lock_sha256=$(sha256sum Slakshna/Cargo.lock | awk '{print $1}')"
    echo "rust_cc=/usr/bin/gcc"
    echo "rust_cxx=/usr/bin/g++"
    echo "libclang_path=${libclang_path}"
    echo "excluded_module_gcc_runtime=${module_gcc_runtime}"
} > "${artifact_root}/rust-toolchain.txt"

echo "=== Verify/build pinned Slakshna Rust binary ==="
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u LIBRARY_PATH -u CPATH \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    LIBCLANG_PATH="${libclang_path}" \
    LD_LIBRARY_PATH="${rust_ld_library_path}" \
    cargo build --locked --release --manifest-path Slakshna/Cargo.toml \
    2>&1 | tee "${artifact_root}/cargo-build.log"
rust_binary="${CARGO_TARGET_DIR}/release/iiitd"
test -x "${rust_binary}"

echo "=== Prewarm both deterministic data shards ==="
for peer_name in peer-a peer-b; do
    if [[ "${peer_name}" == "peer-a" ]]; then
        row_indices="0,2,4,6"
    else
        row_indices="1,3,5,7"
    fi
    bootstrap_dir="${artifact_root}/bootstrap/${peer_name}"
    SLAKSHNA_CPU_LIMIT="${peer_cpus}" python \
        "${experiment_root}/src/phase1/prepare_experiment.py" \
        --template "${experiment_root}/configs/phase4/client_two_steps.yaml" \
        --run-dir "${bootstrap_dir}" \
        --checkpoint-dir "${bootstrap_dir}/unused-checkpoints" \
        --run-name "${run_id}-bootstrap-${peer_name}" \
        --row-indices "${row_indices}" \
        --data-label "phase5-${peer_name}" \
        2>&1 | tee "${bootstrap_dir}.log"
done

echo "=== Create common G0 without an optimizer step ==="
CUDA_VISIBLE_DEVICES=0 python "${experiment_root}/src/phase5/create_initial_adapter.py" \
    --config "${artifact_root}/bootstrap/peer-a/resolved-config.yaml" \
    --output "${artifact_root}/global-0/global_adapter.safetensors" \
    --resume-output "${artifact_root}/global-0/global_adapter.pth" \
    --audit-output "${artifact_root}/global-0/initialization-audit.json"

read -r a_p2p a_ws a_api b_p2p b_ws b_api < <(python - <<'PY'
import socket

sockets = []
ports = []
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

a_runtime="${artifact_root}/peer-a/runtime"
b_runtime="${artifact_root}/peer-b/runtime"
a_data="${artifact_root}/peer-a/node-data"
b_data="${artifact_root}/peer-b/node-data"
python "${experiment_root}/src/phase5/prepare_runtime.py" \
    --template "${experiment_root}/configs/phase5/two_peer.toml" \
    --bridge "${experiment_root}/src/phase5/ml_bridge.py" \
    --runtime-dir "${a_runtime}" \
    --data-dir "${a_data}" \
    --peer-name peer-a --gpu-id 0 \
    --p2p-port "${a_p2p}" --ws-port "${a_ws}" --api-port "${a_api}"

a_pid=""
b_pid=""
recovery_pid=""
stop_pid() {
    local pid="$1"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
}
cleanup() {
    stop_pid "${recovery_pid}"
    stop_pid "${b_pid}"
    stop_pid "${a_pid}"
}
trap cleanup EXIT INT TERM

started_pid=""
start_peer() {
    local peer_name="$1"
    local runtime_dir="$2"
    local config_name="$3"
    local row_indices="$4"
    local log_path="$5"
    (
        cd "${runtime_dir}"
        exec env \
            LD_LIBRARY_PATH="${rust_ld_library_path}" \
            SLAKSHNA_PHASE5_ARTIFACT_ROOT="${artifact_root}" \
            SLAKSHNA_EXPERIMENT_ROOT="${experiment_root}" \
            SLAKSHNA_PHASE5_RUN_ID="${run_id}" \
            SLAKSHNA_PHASE5_PEER_NAME="${peer_name}" \
            SLAKSHNA_PHASE5_ROW_INDICES="${row_indices}" \
            SLAKSHNA_PHASE5_GLOBAL0="${artifact_root}/global-0/global_adapter.safetensors" \
            SLAKSHNA_PHASE5_GLOBAL0_RESUME="${artifact_root}/global-0/global_adapter.pth" \
            SLAKSHNA_CPU_LIMIT="${peer_cpus}" \
            SLAKSHNA_RAY_CONTROL_CPUS=4 \
            "${rust_binary}" --config "${config_name}"
    ) > "${log_path}" 2>&1 &
    started_pid=$!
}

wait_api() {
    local pid="$1"
    local url="$2"
    local output="$3"
    for _ in $(seq 1 60); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            return 1
        fi
        if curl --silent --fail "${url}" > "${output}.tmp"; then
            mv "${output}.tmp" "${output}"
            return 0
        fi
        sleep 1
    done
    return 1
}

# Start immediately after a wall-clock boundary if the next 120-second boundary
# is too close. This leaves ample time to obtain A's EndpointId and start B.
seconds_until_boundary=$((120 - ($(date +%s) % 120)))
if (( seconds_until_boundary < 25 )); then
    echo "Waiting ${seconds_until_boundary}s to avoid a partial first epoch..."
    sleep $((seconds_until_boundary + 1))
fi

echo "=== Start Peer A and obtain its real EndpointId ==="
start_peer peer-a "${a_runtime}" node.toml "0,2,4,6" "${artifact_root}/peer-a/rust-node.log"
a_pid="${started_pid}"
a_base="http://127.0.0.1:${a_api}"
if ! wait_api "${a_pid}" "${a_base}/status" "${artifact_root}/peer-a/api-status.initial.json"; then
    tail -n 160 "${artifact_root}/peer-a/rust-node.log" >&2
    exit 1
fi
a_endpoint="$(jq -r '.endpoint_id' "${artifact_root}/peer-a/api-status.initial.json")"
if [[ -z "${a_endpoint}" || "${a_endpoint}" == "null" ]]; then
    echo "Peer A did not expose an EndpointId" >&2
    exit 1
fi

python "${experiment_root}/src/phase5/prepare_runtime.py" \
    --template "${experiment_root}/configs/phase5/two_peer.toml" \
    --bridge "${experiment_root}/src/phase5/ml_bridge.py" \
    --runtime-dir "${b_runtime}" \
    --data-dir "${b_data}" \
    --peer-name peer-b --gpu-id 1 \
    --p2p-port "${b_p2p}" --ws-port "${b_ws}" --api-port "${b_api}" \
    --seed-peer "${a_endpoint}@127.0.0.1:${a_p2p}"

echo "=== Start Peer B through explicit Iroh direct addressing ==="
start_peer peer-b "${b_runtime}" node.toml "1,3,5,7" "${artifact_root}/peer-b/rust-node.log"
b_pid="${started_pid}"
b_base="http://127.0.0.1:${b_api}"
if ! wait_api "${b_pid}" "${b_base}/status" "${artifact_root}/peer-b/api-status.initial.json"; then
    tail -n 160 "${artifact_root}/peer-b/rust-node.log" >&2
    exit 1
fi

echo "=== Require bidirectional mesh membership before training ==="
mesh_ready=0
for _ in $(seq 1 60); do
    if ! kill -0 "${a_pid}" 2>/dev/null || ! kill -0 "${b_pid}" 2>/dev/null; then
        echo "A peer exited while forming the mesh" >&2
        exit 1
    fi
    curl --silent --fail "${a_base}/peers" > "${artifact_root}/peer-a/api-peers.tmp.json" || true
    curl --silent --fail "${b_base}/peers" > "${artifact_root}/peer-b/api-peers.tmp.json" || true
    if jq -e '(.connected | length) == 1' "${artifact_root}/peer-a/api-peers.tmp.json" >/dev/null 2>&1 && \
       jq -e '(.connected | length) == 1' "${artifact_root}/peer-b/api-peers.tmp.json" >/dev/null 2>&1; then
        mesh_ready=1
        break
    fi
    sleep 1
done
if [[ "${mesh_ready}" != 1 ]]; then
    echo "The two peers did not form a bidirectional mesh" >&2
    exit 1
fi

echo "=== Wait for two rounds, four updates, and four reviews on both peers ==="
history_ready=0
for poll in $(seq 1 900); do
    if ! kill -0 "${a_pid}" 2>/dev/null || ! kill -0 "${b_pid}" 2>/dev/null; then
        echo "A peer exited before the two-round history converged" >&2
        tail -n 120 "${artifact_root}/peer-a/rust-node.log" >&2
        tail -n 120 "${artifact_root}/peer-b/rust-node.log" >&2
        exit 1
    fi
    if grep -Eq "Python ML Engine failed|Failed to start Python process" \
          "${artifact_root}/peer-a/rust-node.log" \
          "${artifact_root}/peer-b/rust-node.log"; then
        echo "A Phase 5 ML child failed; aborting instead of waiting for history convergence" >&2
        tail -n 120 "${artifact_root}/peer-a/rust-node.log" >&2
        tail -n 120 "${artifact_root}/peer-b/rust-node.log" >&2
        exit 1
    fi
    curl --silent --fail "${a_base}/updates" > "${artifact_root}/peer-a/api-updates.tmp.json" || true
    curl --silent --fail "${b_base}/updates" > "${artifact_root}/peer-b/api-updates.tmp.json" || true
    if (( poll % 15 == 0 )); then
        a_count="$(jq -r '.updates | length' "${artifact_root}/peer-a/api-updates.tmp.json" 2>/dev/null || echo '?')"
        b_count="$(jq -r '.updates | length' "${artifact_root}/peer-b/api-updates.tmp.json" 2>/dev/null || echo '?')"
        a_round="$(jq -r '.rounds_completed' "${artifact_root}/peer-a/bridge-state.json" 2>/dev/null || echo 0)"
        b_round="$(jq -r '.rounds_completed' "${artifact_root}/peer-b/bridge-state.json" 2>/dev/null || echo 0)"
        echo "Phase 5 progress: elapsed=${poll}s rounds(A/B)=${a_round}/${b_round} records(A/B)=${a_count}/${b_count}"
    fi
    if jq -e '[.updates[].kind.ModelUpdate? | select(. != null)] | length == 4' \
           "${artifact_root}/peer-a/api-updates.tmp.json" >/dev/null 2>&1 && \
       jq -e '[.updates[].kind.PeerReview? | select(. != null)] | length == 4' \
           "${artifact_root}/peer-a/api-updates.tmp.json" >/dev/null 2>&1 && \
       jq -e '[.updates[].kind.ModelUpdate? | select(. != null)] | length == 4' \
           "${artifact_root}/peer-b/api-updates.tmp.json" >/dev/null 2>&1 && \
       jq -e '[.updates[].kind.PeerReview? | select(. != null)] | length == 4' \
           "${artifact_root}/peer-b/api-updates.tmp.json" >/dev/null 2>&1; then
        history_ready=1
        break
    fi
    sleep 1
done
if [[ "${history_ready}" != 1 ]]; then
    echo "Two-round peer histories did not converge within 15 minutes" >&2
    exit 1
fi

for peer_name in peer-a peer-b; do
    if [[ "${peer_name}" == "peer-a" ]]; then
        api_base="${a_base}"
    else
        api_base="${b_base}"
    fi
    mv "${artifact_root}/${peer_name}/api-updates.tmp.json" \
       "${artifact_root}/${peer_name}/api-updates.json"
    curl --silent --fail "${api_base}/status" > "${artifact_root}/${peer_name}/api-status.json"
    curl --silent --fail "${api_base}/peers" > "${artifact_root}/${peer_name}/api-peers.json"
    curl --silent --fail "${api_base}/leaderboard" > "${artifact_root}/${peer_name}/api-leaderboard.json"
done

echo "=== Stop both training peers after exactly two rounds ==="
stop_pid "${b_pid}"
b_pid=""
stop_pid "${a_pid}"
a_pid=""

echo "=== Restart Peer A to verify persistent identity and known-peer state ==="
python "${experiment_root}/src/phase5/prepare_runtime.py" \
    --template "${experiment_root}/configs/phase5/two_peer.toml" \
    --bridge "${experiment_root}/src/phase5/ml_bridge.py" \
    --runtime-dir "${a_runtime}" \
    --data-dir "${a_data}" \
    --peer-name peer-a --gpu-id 0 \
    --p2p-port "${a_p2p}" --ws-port "${a_ws}" --api-port "${a_api}" \
    --epoch-duration 1000000000 --sync-deadline 999999999 \
    --config-name recovery.toml --reuse-runtime
start_peer peer-a "${a_runtime}" recovery.toml "0,2,4,6" "${artifact_root}/peer-a/recovery-rust-node.log"
recovery_pid="${started_pid}"
if ! wait_api "${recovery_pid}" "${a_base}/status" "${artifact_root}/peer-a/recovery-status.json"; then
    tail -n 160 "${artifact_root}/peer-a/recovery-rust-node.log" >&2
    exit 1
fi
curl --silent --fail "${a_base}/peers" > "${artifact_root}/peer-a/recovery-peers.json"
curl --silent --fail "${a_base}/updates" > "${artifact_root}/peer-a/recovery-updates.json"
stop_pid "${recovery_pid}"
recovery_pid=""
trap - EXIT INT TERM

python "${experiment_root}/src/phase5/verify_phase5.py" \
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

echo "Phase 5 artifact: ${artifact_root}"
