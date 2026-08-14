#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
activation_script="${slakshna_root}/Bhaskera/phase9-activate.sh"
source_root="${PHASE9_CULTURE_SOURCE_ROOT:-${workspace_root}/local_data/culture-instruct/data}"
output_root="${PHASE9_CULTURE_OUTPUT_ROOT:-${experiment_root}/.runtime/data/phase9/culture-instruct-alpaca-olmo1b-v1}"
expected_revision="9f93ec45ae0d3eb9c901aff3b50d4325b5050488"

[[ -f "${activation_script}" ]] || {
    echo "Missing Phase 9 environment activation helper: ${activation_script}" >&2
    echo "Run bash 1_setup_env.sh first." >&2
    exit 1
}
# shellcheck source=/dev/null
source "${activation_script}"

current_revision="$(git -C "${slakshna_root}" rev-parse HEAD)"
[[ "${current_revision}" == "${expected_revision}" ]] || {
    echo "Slakshna revision mismatch: ${current_revision}; expected ${expected_revision}" >&2
    exit 1
}
[[ -x "${SLAKSHNA_UV_ENVIRONMENT}/bin/python" ]] || {
    echo "Phase 9 Python is missing from ${SLAKSHNA_UV_ENVIRONMENT}" >&2
    exit 1
}
[[ -f "${source_root}/Australia.json" && -f "${source_root}/India.json" ]] || {
    echo "Expected Australia.json and India.json under ${source_root}" >&2
    exit 1
}

mkdir -p "${experiment_root}/.runtime/logs/phase9"
run_stamp="$(date +%Y%m%dT%H%M%S)"
log_path="${experiment_root}/.runtime/logs/phase9/prepare-data-${run_stamp}.log"

args=(
    --source-root "${source_root}"
    --output-root "${output_root}"
    --countries Australia India
    --model-id allenai/OLMo-1B-hf
    --model-revision aee7752d9c08ee4775e9b0091426d8410e8f6a89
    --seq-len 512
    --validation-fraction 0.05
    --smoke-train-rows 1000
    --smoke-validation-rows 128
)
if [[ "${PHASE9_DATA_OVERWRITE:-0}" == "1" ]]; then
    args+=(--overwrite)
fi

echo "Phase 9 cultureInstruct preparation"
echo "  source : ${source_root}"
echo "  output : ${output_root}"
echo "  log    : ${log_path}"
echo "  Python : ${SLAKSHNA_UV_ENVIRONMENT}/bin/python"

CUDA_VISIBLE_DEVICES="" "${SLAKSHNA_UV_ENVIRONMENT}/bin/python" \
    "${experiment_root}/src/phase9/prepare_culture_data.py" \
    "${args[@]}" 2>&1 | tee "${log_path}"

manifest="${output_root}/manifest.json"
[[ -s "${manifest}" ]] || {
    echo "Data preparation did not produce ${manifest}" >&2
    exit 1
}
echo "PHASE 9 DATA PREPARATION PASSED"
echo "Manifest: ${manifest}"
echo "Log     : ${log_path}"
