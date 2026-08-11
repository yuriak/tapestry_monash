#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
cd "${workspace_root}"

# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

local_port="${SLAKSHNA_PHASE8_PLAYIT_LOCAL_PORT:-38080}"
probe_token="${SLAKSHNA_PHASE8_PROBE_TOKEN:-slakshna-phase8-m3-spartan-v1}"
wait_seconds="${SLAKSHNA_PHASE8_PROBE_WAIT:-1800}"
public_host="${SLAKSHNA_PHASE8_PLAYIT_PUBLIC_HOST:-}"
public_port="${SLAKSHNA_PHASE8_PLAYIT_PUBLIC_PORT:-}"
if [[ ! "${local_port}" =~ ^[0-9]+$ ]] || (( local_port < 1024 || local_port > 65535 )); then
    echo "Invalid SLAKSHNA_PHASE8_PLAYIT_LOCAL_PORT=${local_port}" >&2
    exit 2
fi
if [[ ! "${wait_seconds}" =~ ^[0-9]+$ ]] || (( wait_seconds < 60 || wait_seconds > 3600 )); then
    echo "SLAKSHNA_PHASE8_PROBE_WAIT must be 60-3600 seconds" >&2
    exit 2
fi

run_id="$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID:-interactive}_m3_playit_probe"
artifact_root="${experiment_root}/artifacts/phase8/network-preflight/${run_id}"
mkdir -p "${artifact_root}"

cleanup() {
    bash "${experiment_root}/scripts/phase8/playit_agent.sh" stop >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "=== Start the claimed playit agent and load its assigned UDP tunnel ==="
bash "${experiment_root}/scripts/phase8/playit_agent.sh" start

echo
echo "=== Wait for the Spartan UDP probe ==="
if [[ -n "${public_host}" || -n "${public_port}" ]]; then
    echo "Public tunnel: ${public_host:-<unset>}:${public_port:-<unset>} (UDP)"
fi
echo "Local mapping : 127.0.0.1:${local_port}"
echo "Probe token   : ${probe_token}"
echo "Timeout       : ${wait_seconds} seconds"
python "${experiment_root}/src/phase8/udp_probe.py" server \
    --bind 127.0.0.1 --port "${local_port}" \
    --token "${probe_token}" --wait "${wait_seconds}" \
    --output "${artifact_root}/m3-server.json"

printf '%s\n' "${artifact_root}" > \
    "${experiment_root}/artifacts/phase8/latest-network-preflight-m3.txt"
echo "M3 network preflight evidence: ${artifact_root}/m3-server.json"
