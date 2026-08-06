#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
cd "${workspace_root}"

# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Phase 6 must run inside a two-node Slurm allocation" >&2
    exit 2
fi
job_nodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}"
if [[ "${job_nodes}" != 2 ]]; then
    echo "Phase 6 requires exactly two allocated nodes, got ${job_nodes}" >&2
    exit 2
fi
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#nodes[@]}" -ne 2 || "${nodes[0]}" == "${nodes[1]}" ]]; then
    echo "Phase 6 could not resolve two distinct nodes: ${nodes[*]}" >&2
    exit 2
fi
a_host="${nodes[0]}"
b_host="${nodes[1]}"

resolve_node_ip() {
    local host="$1"
    srun --nodes=1 --ntasks=1 --nodelist="${host}" --exclusive --exact \
        --cpus-per-task=1 python -c \
        'import socket; print(socket.gethostbyname(socket.gethostname()))'
}
a_ip="$(resolve_node_ip "${a_host}" | tail -n 1)"
b_ip="$(resolve_node_ip "${b_host}" | tail -n 1)"
python - "${a_ip}" "${b_ip}" <<'PY'
import ipaddress
import sys

addresses = [ipaddress.ip_address(value) for value in sys.argv[1:]]
if any(value.version != 4 or value.is_loopback for value in addresses):
    raise SystemExit(f"Phase 6 requires non-loopback IPv4 node addresses: {addresses}")
if addresses[0] == addresses[1]:
    raise SystemExit(f"Phase 6 node addresses are identical: {addresses}")
PY
echo "Phase 6 nodes: peer-a=${a_host}/${a_ip}, peer-b=${b_host}/${b_ip}"

python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"Phase 6 batch controller requires one visible GPU on node A, got {torch.cuda.device_count()}")
print(f"Phase 6 controller GPU: {torch.cuda.get_device_name(0)}")
PY

run_id="${1:-$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID}_m3_phase6}"
artifact_root="${experiment_root}/artifacts/phase6/${run_id}"
mkdir -p "${artifact_root}/global-0" "${artifact_root}/bootstrap" \
    "${artifact_root}/peer-a" "${artifact_root}/peer-b"

peer_cpus="${SLURM_CPUS_PER_TASK:-0}"
peer_cpus="${peer_cpus%%(*}"
if [[ ! "${peer_cpus}" =~ ^[0-9]+$ ]] || (( peer_cpus < 8 )); then
    echo "Phase 6 requires at least 8 CPUs per node task, got ${peer_cpus}" >&2
    exit 2
fi

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
    if [[ -z "${entry}" || "${entry}" == "${module_gcc_runtime}" ]]; then
        continue
    fi
    [[ -z "${rust_ld_library_path}" ]] || rust_ld_library_path+=":"
    rust_ld_library_path+="${entry}"
done
export SLAKSHNA_RUST_LD_LIBRARY_PATH="${rust_ld_library_path}"
libclang_path="/usr/lib64/llvm21/lib64"
test -f "${libclang_path}/libclang.so.21.1.8"

{
    rustc --version
    cargo --version
    echo "slakshna_revision=$(git -C Slakshna rev-parse HEAD)"
    echo "cargo_lock_sha256=$(sha256sum Slakshna/Cargo.lock | awk '{print $1}')"
    echo "nodes=${a_host},${b_host}"
    echo "node_ips=${a_ip},${b_ip}"
} > "${artifact_root}/rust-toolchain.txt"

echo "=== Build pinned Slakshna once on the shared filesystem ==="
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u LIBRARY_PATH -u CPATH \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    LIBCLANG_PATH="${libclang_path}" \
    LD_LIBRARY_PATH="${rust_ld_library_path}" \
    cargo build --locked --release --manifest-path Slakshna/Cargo.toml \
    2>&1 | tee "${artifact_root}/cargo-build.log"
test -x "${CARGO_TARGET_DIR}/release/iiitd"

echo "=== Prepare the two deterministic shards ==="
for peer_name in peer-a peer-b; do
    if [[ "${peer_name}" == peer-a ]]; then rows="0,2,4,6"; else rows="1,3,5,7"; fi
    bootstrap_dir="${artifact_root}/bootstrap/${peer_name}"
    SLAKSHNA_CPU_LIMIT="${peer_cpus}" python \
        "${experiment_root}/src/phase1/prepare_experiment.py" \
        --template "${experiment_root}/configs/phase4/client_two_steps.yaml" \
        --run-dir "${bootstrap_dir}" \
        --checkpoint-dir "${bootstrap_dir}/unused-checkpoints" \
        --run-name "${run_id}-bootstrap-${peer_name}" \
        --row-indices "${rows}" --data-label "phase6-${peer_name}" \
        2>&1 | tee "${bootstrap_dir}.log"
done

echo "=== Create common G0 on node A ==="
python "${experiment_root}/src/phase5/create_initial_adapter.py" \
    --config "${artifact_root}/bootstrap/peer-a/resolved-config.yaml" \
    --output "${artifact_root}/global-0/global_adapter.safetensors" \
    --resume-output "${artifact_root}/global-0/global_adapter.pth" \
    --audit-output "${artifact_root}/global-0/initialization-audit.json"

port_offset=$((SLURM_JOB_ID % 8000))
p2p_port=$((30000 + port_offset))
ws_port=$((40000 + port_offset))
api_port=$((50000 + port_offset))
epoch_duration=120
worker="${experiment_root}/scripts/phase6/run_peer_on_node.sh"

a_step_pid=""
b_step_pid=""
recovery_step_pid=""
stop_step() {
    local pid="$1"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
        fi
        wait "${pid}" 2>/dev/null || true
    fi
}
cleanup() {
    stop_step "${recovery_step_pid}"
    stop_step "${b_step_pid}"
    stop_step "${a_step_pid}"
}
trap cleanup EXIT INT TERM

start_peer_step() {
    local host="$1" peer_name="$2" rows="$3" node_ip="$4" seed="$5"
    local config_name="$6" reuse="$7" epoch="$8" log_path="$9"
    srun --nodes=1 --ntasks=1 --nodelist="${host}" --exclusive --exact \
        --cpus-per-task="${peer_cpus}" --gres=gpu:1 \
        bash "${worker}" "${peer_name}" "${artifact_root}" "${run_id}" \
        "${rows}" "${node_ip}" "${p2p_port}" "${ws_port}" "${api_port}" \
        "${seed}" "${config_name}" "${reuse}" "${epoch}" \
        > "${log_path}" 2>&1 &
    started_step_pid=$!
}

wait_api() {
    local pid="$1" url="$2" output="$3"
    for _ in $(seq 1 90); do
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
if (( seconds_until_boundary < 35 )); then
    echo "Waiting ${seconds_until_boundary}s to avoid a partial first epoch"
    sleep $((seconds_until_boundary + 1))
fi

echo "=== Start Peer A on ${a_host} ==="
started_step_pid=""
start_peer_step "${a_host}" peer-a "0,2,4,6" "${a_ip}" none \
    node.toml 0 "${epoch_duration}" "${artifact_root}/peer-a/rust-node.log"
a_step_pid="${started_step_pid}"
a_base="http://${a_ip}:${api_port}"
if ! wait_api "${a_step_pid}" "${a_base}/status" "${artifact_root}/peer-a/api-status.initial.json"; then
    tail -n 180 "${artifact_root}/peer-a/rust-node.log" >&2
    exit 1
fi
a_endpoint="$(jq -r '.endpoint_id' "${artifact_root}/peer-a/api-status.initial.json")"
[[ -n "${a_endpoint}" && "${a_endpoint}" != null ]]
direct_seed="${a_endpoint}@${a_ip}:${p2p_port}"

echo "=== Start Peer B on ${b_host} with direct cross-node seed ==="
start_peer_step "${b_host}" peer-b "1,3,5,7" "${b_ip}" "${direct_seed}" \
    node.toml 0 "${epoch_duration}" "${artifact_root}/peer-b/rust-node.log"
b_step_pid="${started_step_pid}"
b_base="http://${b_ip}:${api_port}"
if ! wait_api "${b_step_pid}" "${b_base}/status" "${artifact_root}/peer-b/api-status.initial.json"; then
    tail -n 180 "${artifact_root}/peer-b/rust-node.log" >&2
    exit 1
fi
b_endpoint="$(jq -r '.endpoint_id' "${artifact_root}/peer-b/api-status.initial.json")"
[[ -n "${b_endpoint}" && "${b_endpoint}" != null ]]

jq -n \
    --arg job_id "${SLURM_JOB_ID}" --arg run_id "${run_id}" \
    --arg a_host "${a_host}" --arg b_host "${b_host}" \
    --arg a_ip "${a_ip}" --arg b_ip "${b_ip}" \
    --arg a_endpoint "${a_endpoint}" --arg b_endpoint "${b_endpoint}" \
    --arg seed "${direct_seed}" --argjson p2p_port "${p2p_port}" \
    '{schema_version:1,slurm_job_id:$job_id,slurm_job_num_nodes:2,run_id:$run_id,
      nodes:{"peer-a":{hostname:$a_host,ip:$a_ip,endpoint_id:$a_endpoint,p2p_port:$p2p_port},
             "peer-b":{hostname:$b_host,ip:$b_ip,endpoint_id:$b_endpoint,p2p_port:$p2p_port}},
      direct_seed:$seed,discovery:{mdns:false,dht:false,dns:false,relay:false}}' \
    > "${artifact_root}/cluster-topology.json"

echo "=== Require bidirectional cross-node mesh membership ==="
mesh_ready=0
for _ in $(seq 1 120); do
    if ! kill -0 "${a_step_pid}" 2>/dev/null || ! kill -0 "${b_step_pid}" 2>/dev/null; then
        echo "A Phase 6 peer exited while forming the mesh" >&2
        exit 1
    fi
    curl --silent --fail --connect-timeout 2 "${a_base}/peers" > "${artifact_root}/peer-a/api-peers.tmp.json" || true
    curl --silent --fail --connect-timeout 2 "${b_base}/peers" > "${artifact_root}/peer-b/api-peers.tmp.json" || true
    if jq -e --arg id "${b_endpoint}" '.connected == [$id]' "${artifact_root}/peer-a/api-peers.tmp.json" >/dev/null 2>&1 && \
       jq -e --arg id "${a_endpoint}" '.connected == [$id]' "${artifact_root}/peer-b/api-peers.tmp.json" >/dev/null 2>&1; then
        mesh_ready=1
        break
    fi
    sleep 1
done
[[ "${mesh_ready}" == 1 ]] || { echo "Cross-node mesh did not form" >&2; exit 1; }

echo "=== Wait for two cross-node rounds ==="
history_ready=0
for poll in $(seq 1 1200); do
    if ! kill -0 "${a_step_pid}" 2>/dev/null || ! kill -0 "${b_step_pid}" 2>/dev/null; then
        echo "A Phase 6 peer exited before history convergence" >&2
        exit 1
    fi
    if grep -Eq "Python ML Engine failed|Failed to start Python process" \
        "${artifact_root}/peer-a/rust-node.log" "${artifact_root}/peer-b/rust-node.log"; then
        tail -n 180 "${artifact_root}/peer-a/rust-node.log" >&2
        tail -n 180 "${artifact_root}/peer-b/rust-node.log" >&2
        exit 1
    fi
    curl --silent --fail --connect-timeout 2 "${a_base}/updates" > "${artifact_root}/peer-a/api-updates.tmp.json" || true
    curl --silent --fail --connect-timeout 2 "${b_base}/updates" > "${artifact_root}/peer-b/api-updates.tmp.json" || true
    if (( poll % 15 == 0 )); then
        a_count="$(jq -r '.updates|length' "${artifact_root}/peer-a/api-updates.tmp.json" 2>/dev/null || echo '?')"
        b_count="$(jq -r '.updates|length' "${artifact_root}/peer-b/api-updates.tmp.json" 2>/dev/null || echo '?')"
        a_round="$(jq -r '.rounds_completed' "${artifact_root}/peer-a/bridge-state.json" 2>/dev/null || echo 0)"
        b_round="$(jq -r '.rounds_completed' "${artifact_root}/peer-b/bridge-state.json" 2>/dev/null || echo 0)"
        echo "Phase 6 progress: elapsed=${poll}s rounds(A/B)=${a_round}/${b_round} records(A/B)=${a_count}/${b_count}"
    fi
    if jq -e '([.updates[].kind.ModelUpdate?|select(.!=null)]|length==4) and ([.updates[].kind.PeerReview?|select(.!=null)]|length==4)' "${artifact_root}/peer-a/api-updates.tmp.json" >/dev/null 2>&1 && \
       jq -e '([.updates[].kind.ModelUpdate?|select(.!=null)]|length==4) and ([.updates[].kind.PeerReview?|select(.!=null)]|length==4)' "${artifact_root}/peer-b/api-updates.tmp.json" >/dev/null 2>&1; then
        history_ready=1
        break
    fi
    sleep 1
done
[[ "${history_ready}" == 1 ]] || { echo "Cross-node histories did not converge" >&2; exit 1; }

for peer_name in peer-a peer-b; do
    if [[ "${peer_name}" == peer-a ]]; then api_base="${a_base}"; else api_base="${b_base}"; fi
    mv "${artifact_root}/${peer_name}/api-updates.tmp.json" "${artifact_root}/${peer_name}/api-updates.json"
    curl --silent --fail "${api_base}/status" > "${artifact_root}/${peer_name}/api-status.json"
    curl --silent --fail "${api_base}/peers" > "${artifact_root}/${peer_name}/api-peers.json"
    curl --silent --fail "${api_base}/leaderboard" > "${artifact_root}/${peer_name}/api-leaderboard.json"
done

echo "=== Stop both peers after exactly two rounds ==="
stop_step "${b_step_pid}"; b_step_pid=""
stop_step "${a_step_pid}"; a_step_pid=""

echo "=== Restart Peer A on its original node ==="
start_peer_step "${a_host}" peer-a "0,2,4,6" "${a_ip}" none \
    recovery.toml 1 1000000000 "${artifact_root}/peer-a/recovery-rust-node.log"
recovery_step_pid="${started_step_pid}"
if ! wait_api "${recovery_step_pid}" "${a_base}/status" "${artifact_root}/peer-a/recovery-status.json"; then
    tail -n 180 "${artifact_root}/peer-a/recovery-rust-node.log" >&2
    exit 1
fi
curl --silent --fail "${a_base}/peers" > "${artifact_root}/peer-a/recovery-peers.json"
curl --silent --fail "${a_base}/updates" > "${artifact_root}/peer-a/recovery-updates.json"
stop_step "${recovery_step_pid}"; recovery_step_pid=""
trap - EXIT INT TERM

sleep 3
echo "=== Check both nodes for allocation-scoped process residue ==="
srun --nodes=1 --ntasks=1 --nodelist="${a_host}" --exclusive --exact --cpus-per-task=1 \
    python "${experiment_root}/src/phase6/node_probe.py" clean \
    --job-id "${SLURM_JOB_ID}" --output "${artifact_root}/peer-a/process-residue.json"
srun --nodes=1 --ntasks=1 --nodelist="${b_host}" --exclusive --exact --cpus-per-task=1 \
    python "${experiment_root}/src/phase6/node_probe.py" clean \
    --job-id "${SLURM_JOB_ID}" --output "${artifact_root}/peer-b/process-residue.json"

python "${experiment_root}/src/phase6/verify_phase6.py" \
    --artifact-root "${artifact_root}" --output "${artifact_root}/verification-summary.json"

if [[ -n "$(git -C Slakshna status --short)" ]]; then
    echo "Slakshna submodule became dirty" >&2
    git -C Slakshna status --short >&2
    exit 1
fi
{
    echo "revision=$(git -C Slakshna rev-parse HEAD)"
    echo "status=clean"
} > "${artifact_root}/source-integrity.txt"
printf '%s\n' "${artifact_root}" > "${experiment_root}/artifacts/phase6/latest-${SLURM_JOB_ID}.txt"
echo "Phase 6 artifact: ${artifact_root}"
