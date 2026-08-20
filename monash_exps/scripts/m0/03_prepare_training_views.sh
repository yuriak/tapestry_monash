#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
runtime_root="${experiment_root}/.runtime"
venv="${M0_UV_ENVIRONMENT:-${slakshna_root}/Bhaskera/.venv-phase9}"
python_bin="${venv}/bin/python"

source_root="${M0_CONTINENT_SOURCE_ROOT:-${workspace_root}/local_data/m0_incoming/continent_splits}"
output_root="${M0_PREPARED_ROOT:-${runtime_root}/data/m0/prepared}"
log_root="${runtime_root}/logs/m0"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "${python_bin}" ]] || die "Missing M0 Python environment: ${python_bin}"
[[ -d "${source_root}" ]] || die "Missing continent-split source: ${source_root}"
[[ -f "${experiment_root}/src/m0/prepare_training_views.py" ]] || \
    die "Missing M0 data preparation implementation"

mkdir -p "${log_root}"
stamp="$(date +%Y%m%dT%H%M%S)"
log_path="${log_root}/prepare-training-views-${stamp}.log"
exec > >(tee -a "${log_path}") 2>&1

echo "M0 authoritative data preparation"
echo "  source : ${source_root}"
echo "  output : ${output_root}"
echo "  log    : ${log_path}"
echo "  policy : new continent splits only; no old-data comparison; no deduplication"

"${python_bin}" "${experiment_root}/src/m0/prepare_training_views.py" \
    --workspace-root "${workspace_root}" \
    --source-root "${source_root}" \
    --output-root "${output_root}"

echo
echo "M0 TRAINING VIEW PREPARATION PASSED"
echo "Manifest: ${output_root}/manifests/prepared-data.json"
