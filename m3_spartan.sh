#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
    echo "Usage: bash m3_spartan.sh prepare [g0-archive]|reset-run|run <m3-endpoint-id>|status|audit" >&2
    exit 2
fi
action="$1"
shift

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
export SLAKSHNA_CLUSTER=spartan
# shellcheck source=monash_exps/scripts/cluster/activate.sh
source monash_exps/scripts/cluster/activate.sh

experiment_root="${SLAKSHNA_EXPERIMENT_ROOT}"
run_id="phase8-m3-spartan-20260811-v1"
federation_id="slakshna-${run_id}"
cross_root="${experiment_root}/artifacts/phase8/${run_id}"
site_root="${cross_root}/site-b"
runtime_dir="${site_root}/runtime"
data_dir="${site_root}/node-data"
transfer_dir="${site_root}/g0-transfer-bundle"
endpoint_file="${site_root}/iroh-endpoint-id.txt"
node_identity_file="${site_root}/slakshna-node-identity.txt"
rust_log="${site_root}/rust-node.log"
site_audit="${cross_root}/site-b-audit.json"
rust_binary="${SLAKSHNA_RUNTIME_ROOT}/cargo-target/slakshna/release/iiitd"
p2p_port=39080
ws_port=39081
api_port=39082
epoch_duration=600
sync_deadline=570
public_seed_address="147.185.221.231:51716"

require_runtime() {
    [[ -x "${rust_binary}" ]] || { echo "Missing Slakshna release binary: ${rust_binary}" >&2; exit 1; }
    [[ "$(git -C Slakshna rev-parse HEAD)" == "f09eff9a73ae8f1080d4f0b43114b3a8aa5e99bb" ]] || {
        echo "Unexpected Slakshna revision" >&2; exit 1;
    }
}

require_gpu() {
    CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"Phase 8 site requires exactly one visible GPU, got {torch.cuda.device_count()}")
if "A100" not in torch.cuda.get_device_name(0).upper():
    raise SystemExit(f"Phase 8 requires an A100, got {torch.cuda.get_device_name(0)}")
print(f"Spartan Phase 8 GPU: {torch.cuda.get_device_name(0)}")
PY
}

cpu_limit() {
    python - <<'PY'
import os
print(max(4, min(len(os.sched_getaffinity(0)), 12)))
PY
}

stop_pid() {
    local pid="${1:-}"
    [[ -n "${pid}" ]] || return 0
    if kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${pid}" 2>/dev/null; then kill -KILL "${pid}" 2>/dev/null || true; fi
    fi
    wait "${pid}" 2>/dev/null || true
}

json_value() {
    local path="$1" key="$2"
    python - "${path}" "${key}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for part in sys.argv[2].split("."):
    value = value[part]
print(str(value).lower() if isinstance(value, bool) else value)
PY
}

wait_api() {
    local pid="$1" output="$2"
    for _ in $(seq 1 120); do
        kill -0 "${pid}" 2>/dev/null || return 1
        if curl --silent --fail --connect-timeout 2 \
            "http://127.0.0.1:${api_port}/status" > "${output}.tmp"; then
            mv "${output}.tmp" "${output}"
            endpoint="$(json_value "${output}" endpoint_id)"
            if [[ "${endpoint}" =~ ^[0-9a-f]{64}$ ]]; then return 0; fi
        fi
        sleep 1
    done
    return 1
}

start_node() {
    local log_path="$1" limit
    limit="$(cpu_limit)"
    (
        cd "${runtime_dir}"
        exec env \
            SLAKSHNA_EXPERIMENT_ROOT="${experiment_root}" \
            SLAKSHNA_PHASE8_RUN_ID="${run_id}" \
            SLAKSHNA_PHASE8_SITE=site-b \
            SLAKSHNA_PHASE8_SITE_ROOT="${site_root}" \
            SLAKSHNA_CPU_LIMIT="${limit}" \
            SLAKSHNA_RAY_CONTROL_CPUS=4 \
            CUDA_VISIBLE_DEVICES=0 \
            "${rust_binary}" --config node.toml
    ) > "${log_path}" 2>&1 &
    node_pid=$!
}

render_bootstrap_runtime() {
    python "${experiment_root}/src/phase8/prepare_runtime.py" \
        --template "${experiment_root}/configs/phase8/two_site.toml" \
        --bridge "${experiment_root}/src/phase8/ml_bridge.py" \
        --runtime-dir "${runtime_dir}" --data-dir "${data_dir}" \
        --site-root "${site_root}" --site site-b \
        --federation-id "${federation_id}" --gpu-id 0 \
        --p2p-port "${p2p_port}" --ws-port "${ws_port}" --api-port "${api_port}" \
        --epoch-duration 86400 --sync-deadline 86300
}

render_final_runtime() {
    local m3_endpoint="$1"
    python "${experiment_root}/src/phase8/prepare_runtime.py" \
        --template "${experiment_root}/configs/phase8/two_site.toml" \
        --bridge "${experiment_root}/src/phase8/ml_bridge.py" \
        --runtime-dir "${runtime_dir}" --data-dir "${data_dir}" \
        --site-root "${site_root}" --site site-b \
        --federation-id "${federation_id}" --gpu-id 0 \
        --p2p-port "${p2p_port}" --ws-port "${ws_port}" --api-port "${api_port}" \
        --seed-peer "${m3_endpoint}@${public_seed_address}" \
        --allowed-peer "${m3_endpoint}" \
        --epoch-duration "${epoch_duration}" --sync-deadline "${sync_deadline}" \
        --reuse-runtime
}

bootstrap_identity() {
    rm -f -- "${site_root}/identity-api-status.json.tmp"
    node_pid=""
    trap 'stop_pid "${node_pid:-}"' EXIT INT TERM
    start_node "${site_root}/identity-bootstrap.log"
    if ! wait_api "${node_pid}" "${site_root}/identity-api-status.json"; then
        tail -n 160 "${site_root}/identity-bootstrap.log" >&2
        exit 1
    fi
    endpoint="$(json_value "${site_root}/identity-api-status.json" endpoint_id)"
    printf '%s\n' "${endpoint}" > "${endpoint_file}"
    python - "${site_root}/identity-bootstrap.log" "${node_identity_file}" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
text = re.sub(r"\x1b\[[0-9;]*m", "", text)
match = re.search(r"Node Identity: ([a-z0-9]+)", text)
if not match:
    raise SystemExit("could not extract the persistent Slakshna node identity")
open(sys.argv[2], "w", encoding="utf-8").write(match.group(1) + "\n")
PY
    stop_pid "${node_pid}"; node_pid=""
    trap - EXIT INT TERM
}

reset_run_state() {
    [[ -d "${site_root}" && -f "${site_root}/global-0/g0-manifest.json" ]] || {
        echo "Spartan preparation is incomplete; run prepare first" >&2
        exit 1
    }
    if ss -ltnu 2>/dev/null | grep -Eq ":(${p2p_port}|${ws_port}|${api_port})[[:space:]]"; then
        echo "A Phase 8 Spartan port is still active; stop the existing run before reset-run" >&2
        exit 1
    fi
    find "${site_root}" -mindepth 1 -maxdepth 1 -type d \
        \( -name 'round-*' -o -name 'global-[1-9]*' \) -exec rm -rf -- {} +
    rm -rf -- "${data_dir}" "${runtime_dir}"
    rm -f -- \
        "${site_root}/bridge-state.json" \
        "${site_root}/finalization-audit.json" \
        "${site_root}/identity-api-status.json" \
        "${site_root}/identity-api-status.json.tmp" \
        "${site_root}/identity-bootstrap.log" \
        "${site_root}/api-status.initial.json" \
        "${site_root}/api-status.final.json" \
        "${site_root}/api-peers.connected.json" \
        "${site_root}/api-peers.final.json" \
        "${site_root}/api-peers.tmp.json" \
        "${site_root}/api-updates.final.json" \
        "${endpoint_file}" "${node_identity_file}" "${rust_log}" "${site_audit}"
    render_bootstrap_runtime
    bootstrap_identity
}

show_status() {
    echo "Spartan site root  : ${site_root}"
    if [[ -f "${endpoint_file}" ]]; then echo "Spartan EndpointId : $(< "${endpoint_file}")"; fi
    if [[ -f "${node_identity_file}" ]]; then echo "Spartan Node ID     : $(< "${node_identity_file}")"; fi
    if [[ -f "${site_root}/bridge-state.json" ]]; then python -m json.tool "${site_root}/bridge-state.json"; fi
    if [[ -f "${rust_log}" ]]; then tail -n 40 "${rust_log}"; fi
}

case "${action}" in
    prepare)
        [[ "$#" -le 1 ]] || { echo "Usage: bash m3_spartan.sh prepare [g0-archive]" >&2; exit 2; }
        archive="${1:-${script_dir}/phase8-g0-transfer.tar.gz}"
        archive="$(realpath "${archive}")"
        require_runtime
        require_gpu
        [[ -f "${archive}" ]] || { echo "Missing transferred G0 archive: ${archive}" >&2; exit 1; }
        [[ ! -e "${site_root}" ]] || {
            echo "Phase 8 Spartan site already exists; refusing to overwrite: ${site_root}" >&2; exit 1;
        }
        archive_members="$(tar -tzf "${archive}" | sort)"
        expected_members="$(printf '%s\n' creation-audit.json g0-manifest.json global_adapter.pth global_adapter.safetensors | sort)"
        [[ "${archive_members}" == "${expected_members}" ]] || {
            echo "Transferred G0 archive has unexpected members:" >&2
            printf '%s\n' "${archive_members}" >&2
            exit 1
        }
        mkdir -p "${cross_root}"
        limit="$(cpu_limit)"
        echo "=== Prepare private site-b Dolly shard and tokenized inputs ==="
        SLAKSHNA_CPU_LIMIT="${limit}" CUDA_VISIBLE_DEVICES=0 python \
            "${experiment_root}/src/phase8/prepare_site.py" \
            --site site-b --site-root "${site_root}" \
            2>&1 | tee "${cross_root}/site-b-prepare.log"
        mkdir -p "${transfer_dir}"
        tar -xzf "${archive}" --no-same-owner -C "${transfer_dir}"
        echo "=== Verify and install the M3-created G0 bundle ==="
        python "${experiment_root}/src/phase8/g0_bundle.py" install \
            --bundle-dir "${transfer_dir}" --site-root "${site_root}"
        echo "=== Generate and persist the Spartan Iroh identity ==="
        render_bootstrap_runtime
        bootstrap_identity
        echo "SPARTAN PHASE8 PREPARE PASSED"
        show_status
        ;;
    reset-run)
        [[ "$#" -eq 0 ]] || { echo "reset-run takes no arguments" >&2; exit 2; }
        require_runtime
        require_gpu
        echo "=== Reset failed-run state and issue a fresh Spartan identity ==="
        reset_run_state
        echo "SPARTAN PHASE8 RESET PASSED"
        show_status
        ;;
    run)
        [[ "$#" -eq 1 ]] || { echo "Usage: bash m3_spartan.sh run <m3-endpoint-id>" >&2; exit 2; }
        m3_endpoint="$1"
        [[ "${m3_endpoint}" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid M3 EndpointId" >&2; exit 2; }
        require_runtime
        require_gpu
        [[ -f "${endpoint_file}" && -f "${site_root}/global-0/g0-manifest.json" ]] || {
            echo "Run Spartan prepare first" >&2; exit 1;
        }
        [[ ! -e "${site_root}/bridge-state.json" ]] || {
            echo "Bridge state already exists; refusing an implicit resume" >&2; exit 1;
        }
        render_final_runtime "${m3_endpoint}"
        node_pid=""
        trap 'stop_pid "${node_pid:-}"' EXIT INT TERM
        start_node "${rust_log}"
        if ! wait_api "${node_pid}" "${site_root}/api-status.initial.json"; then
            tail -n 180 "${rust_log}" >&2; exit 1
        fi
        actual_endpoint="$(json_value "${site_root}/api-status.initial.json" endpoint_id)"
        [[ "${actual_endpoint}" == "$(< "${endpoint_file}")" ]] || {
            echo "Spartan Iroh identity changed after bootstrap" >&2; exit 1;
        }
        echo "SPARTAN PHASE8 READY: endpoint=${actual_endpoint}"
        echo "=== Require the allowlisted M3 peer through Playit ==="
        mesh_ready=0
        for _ in $(seq 1 600); do
            kill -0 "${node_pid}" 2>/dev/null || break
            curl --silent --fail --connect-timeout 2 \
                "http://127.0.0.1:${api_port}/peers" > "${site_root}/api-peers.tmp.json" || true
            if python - "${site_root}/api-peers.tmp.json" "${m3_endpoint}" <<'PY' 2>/dev/null
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value.get("connected") == [sys.argv[2]] else 1)
PY
            then mesh_ready=1; break; fi
            sleep 1
        done
        [[ "${mesh_ready}" == 1 ]] || { tail -n 200 "${rust_log}" >&2; exit 1; }
        mv "${site_root}/api-peers.tmp.json" "${site_root}/api-peers.connected.json"
        echo "M3-SPARTAN IROH MESH CONNECTED"
        echo "=== Wait for five local rounds and the aggregation-only finalizer ==="
        completed=0
        for poll in $(seq 1 600); do
            kill -0 "${node_pid}" 2>/dev/null || { tail -n 200 "${rust_log}" >&2; exit 1; }
            if grep -Eq "Python ML Engine failed|Failed to start Python process|panicked at" "${rust_log}"; then
                tail -n 240 "${rust_log}" >&2; exit 1
            fi
            if [[ -f "${site_root}/bridge-state.json" ]]; then
                rounds="$(json_value "${site_root}/bridge-state.json" training_rounds_completed)"
                finalized="$(json_value "${site_root}/bridge-state.json" finalized)"
            else
                rounds=0; finalized=false
            fi
            if (( poll % 6 == 0 )); then
                echo "Spartan Phase 8 progress: elapsed=$((poll * 10))s rounds=${rounds}/5 finalized=${finalized}"
            fi
            if [[ "${rounds}" == 5 && "${finalized}" == true ]]; then completed=1; break; fi
            sleep 10
        done
        [[ "${completed}" == 1 ]] || { echo "Spartan Phase 8 timed out" >&2; exit 1; }
        sleep 5
        curl --silent --fail "http://127.0.0.1:${api_port}/status" > "${site_root}/api-status.final.json"
        curl --silent --fail "http://127.0.0.1:${api_port}/peers" > "${site_root}/api-peers.final.json"
        curl --silent --fail "http://127.0.0.1:${api_port}/updates" > "${site_root}/api-updates.final.json"
        stop_pid "${node_pid}"; node_pid=""
        trap - EXIT INT TERM
        python "${experiment_root}/src/phase8/audit_site.py" \
            --site-root "${site_root}" --output "${site_audit}"
        printf '%s\n' "${cross_root}" > "${experiment_root}/artifacts/phase8/latest-cross-cluster-spartan.txt"
        echo "SPARTAN PHASE8 SITE RUN PASSED"
        echo "Audit: ${site_audit}"
        ;;
    status)
        [[ "$#" -eq 0 ]] || { echo "status takes no arguments" >&2; exit 2; }
        show_status
        ;;
    audit)
        [[ "$#" -eq 0 ]] || { echo "audit takes no arguments" >&2; exit 2; }
        python "${experiment_root}/src/phase8/audit_site.py" \
            --site-root "${site_root}" --output "${site_audit}"
        ;;
    *)
        echo "Unknown action: ${action}" >&2
        exit 2
        ;;
esac
