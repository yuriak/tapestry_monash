#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

phase0_require_layout
phase0_require_uv
phase0_print_layout

mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

echo
echo "Creating a clean tracked-file snapshot of Bhaskera..."
phase0_stage_bhaskera
source_revision="$(phase0_source_revision)"

echo
echo "Installing the project-managed Python ${SLAKSHNA_PYTHON_VERSION} interpreter..."
"${SLAKSHNA_UV_BIN}" python install "${SLAKSHNA_PYTHON_VERSION}"

python_path="$("${SLAKSHNA_UV_BIN}" python find "${SLAKSHNA_PYTHON_VERSION}")"
echo "Resolved Python: ${python_path}"

echo
echo "Resolving the CUDA 12.8 environment and writing uv.lock..."
"${SLAKSHNA_UV_BIN}" lock \
    --project "${SLAKSHNA_ENV_PROJECT}" \
    --python "${python_path}"

"${SLAKSHNA_UV_BIN}" lock --check --project "${SLAKSHNA_ENV_PROJECT}"
printf '%s\n' "${source_revision}" > "${SLAKSHNA_BHASKERA_REVISION_FILE}"

echo
sha256sum "${SLAKSHNA_ENV_PROJECT}/uv.lock"
sha256sum "${SLAKSHNA_BHASKERA_REVISION_FILE}"
echo "Lock complete. Review and version environment/uv.lock before using --frozen in jobs."
