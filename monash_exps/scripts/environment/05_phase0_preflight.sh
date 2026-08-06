#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

mode="${1:-cpu}"
if [[ "${mode}" != "cpu" && "${mode}" != "gpu" ]]; then
    echo "Usage: bash scripts/environment/05_phase0_preflight.sh [cpu|gpu]" >&2
    exit 2
fi

phase0_require_layout
phase0_require_uv
phase0_require_locked_source_revision
phase0_stage_bhaskera
[[ -x "${SLAKSHNA_UV_ENVIRONMENT}/bin/python" ]] || {
    echo "Missing synced environment: ${SLAKSHNA_UV_ENVIRONMENT}" >&2
    echo "Run scripts/environment/03_sync_environment.sh first." >&2
    exit 1
}

run_stamp="$(date +%Y%m%d_%H%M%S)"
run_identity="${SLURM_JOB_ID:-interactive_$$}"
run_dir="${SLAKSHNA_EXPERIMENT_ROOT}/artifacts/phase0/${run_stamp}_${run_identity}_${mode}"
mkdir -p "${run_dir}"

phase0_print_layout | tee "${run_dir}/layout.txt"
"${SLAKSHNA_UV_BIN}" --version | tee "${run_dir}/uv-version.txt"
sha256sum "${SLAKSHNA_ENV_PROJECT}/uv.lock" | tee "${run_dir}/uv-lock.sha256"
"${SLAKSHNA_UV_BIN}" pip freeze \
    --python "${SLAKSHNA_UV_ENVIRONMENT}/bin/python" \
    > "${run_dir}/package-manifest.txt"

if type module >/dev/null 2>&1; then
    module -t list > "${run_dir}/modules.txt" 2>&1 || true
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "${run_dir}/nvidia-smi.txt" 2>&1 || true
fi
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version > "${run_dir}/cuda-toolkit.txt" 2>&1 || true
fi

export SLAKSHNA_PHASE0_RUN_DIR="${run_dir}"
export SLAKSHNA_PHASE0_MODE="${mode}"

"${SLAKSHNA_UV_ENVIRONMENT}/bin/python" \
    "${SLAKSHNA_EXPERIMENT_ROOT}/src/phase0/verify_environment.py" \
    --mode "${mode}" \
    --output-dir "${run_dir}" \
    --workspace-root "${SLAKSHNA_WORKSPACE_ROOT}" \
    --experiment-root "${SLAKSHNA_EXPERIMENT_ROOT}" \
    --config "${SLAKSHNA_EXPERIMENT_ROOT}/configs/phase0/preflight.yaml"

echo
echo "Phase 0 ${mode} preflight artifacts: ${run_dir}"
