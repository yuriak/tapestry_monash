#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "Usage: run_playit_probe_client.sh PUBLIC_HOST PUBLIC_UDP_PORT" >&2
    exit 2
fi
public_host="$1"
public_port="$2"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
cd "${workspace_root}"

# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

probe_token="${SLAKSHNA_PHASE8_PROBE_TOKEN:-slakshna-phase8-m3-spartan-v1}"
run_id="$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID:-interactive}_spartan_playit_probe"
artifact_root="${experiment_root}/artifacts/phase8/network-preflight/${run_id}"
mkdir -p "${artifact_root}"

echo "=== Probe M3's playit UDP tunnel from Spartan ==="
python "${experiment_root}/src/phase8/udp_probe.py" client \
    --host "${public_host}" --port "${public_port}" \
    --token "${probe_token}" --attempts 15 --timeout 2 --interval 1 \
    --output "${artifact_root}/spartan-client.json"

printf '%s\n' "${artifact_root}" > \
    "${experiment_root}/artifacts/phase8/latest-network-preflight-spartan.txt"
echo "Spartan network preflight evidence: ${artifact_root}/spartan-client.json"
