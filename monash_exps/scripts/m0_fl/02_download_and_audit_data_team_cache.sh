#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
runtime_root="${experiment_root}/.runtime"
python_bin="${runtime_root}/venvs/primary/bin/python"
folder_id="${M0_FL_DATA_TEAM_FOLDER_ID:-1B25qTaWb0oAkqjkNgZMuxEIDu4lzhcvW}"
authuser="${M0_FL_DRIVE_AUTHUSER:-2}"
cookie_file="${M0_FL_GDOWN_COOKIE_FILE:-${HOME}/.cache/gdown/cookies.txt}"
incoming_root="${workspace_root}/local_data/m0_incoming/data_team_tokenized"
reference_root="${runtime_root}/data/m0/tokenized/olmo2-7b-chatml-seq1024"
manifest_root="${runtime_root}/manifests/m0_fl"

cd "${workspace_root}"
[[ -x "${python_bin}" ]] || {
    echo "Primary environment is missing: ${python_bin}" >&2
    exit 1
}
[[ -s "${cookie_file}" ]] || {
    echo "Authenticated Drive cookie file is missing: ${cookie_file}" >&2
    exit 1
}
[[ -d "${reference_root}" ]] || {
    echo "Local reference token caches are missing: ${reference_root}" >&2
    exit 1
}
mkdir -p "${incoming_root}" "${manifest_root}"

echo "=== Download the authenticated data-team token cache ==="
"${python_bin}" -m monash_exps.src.m0_fl.download_data_team_folder \
    --folder-id "${folder_id}" \
    --output "${incoming_root}" \
    --cookie-file "${cookie_file}" \
    --authuser "${authuser}" \
    --include Oceania.jsonl \
    --include South_Asia.jsonl \
    --manifest "${manifest_root}/data-team-drive-download.json"

echo "=== Audit data-team token strings against the selected views ==="
"${python_bin}" -m monash_exps.src.m0_fl.audit_data_team_tokens \
    --incoming-root "${incoming_root}" \
    --prepared-root "${runtime_root}/data/m0/prepared" \
    --model "${runtime_root}/models/m0/OLMo-2-1124-7B-Instruct" \
    --output-json "${manifest_root}/data-team-token-cache-audit.json" \
    --output-markdown "${manifest_root}/data-team-token-cache-audit.md"

echo "Data-team download and compatibility audit complete."
