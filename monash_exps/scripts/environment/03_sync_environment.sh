#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

profile="${1:-primary}"
if [[ ! "${profile}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Environment profile contains invalid characters: ${profile}" >&2
    exit 2
fi
SLAKSHNA_UV_ENVIRONMENT="${SLAKSHNA_RUNTIME_ROOT}/venvs/${profile}"
export SLAKSHNA_UV_ENVIRONMENT
export UV_PROJECT_ENVIRONMENT="${SLAKSHNA_UV_ENVIRONMENT}"

# A venv contains absolute interpreter links and cannot survive a repository
# move. Preserve a stale directory for inspection and let uv recreate the
# requested profile at its new absolute path.
if [[ -d "${SLAKSHNA_UV_ENVIRONMENT}" && ! -x "${SLAKSHNA_UV_ENVIRONMENT}/bin/python" ]]; then
    stale_environment="${SLAKSHNA_UV_ENVIRONMENT}.stale.$(date +%Y%m%d_%H%M%S)"
    mv "${SLAKSHNA_UV_ENVIRONMENT}" "${stale_environment}"
    echo "Moved non-relocatable environment to: ${stale_environment}"
fi

phase0_require_layout
phase0_require_uv
[[ -f "${SLAKSHNA_ENV_PROJECT}/uv.lock" ]] || {
    echo "Missing uv.lock. Run scripts/environment/02_lock_environment.sh first." >&2
    exit 1
}
phase0_require_locked_source_revision
phase0_stage_bhaskera

phase0_print_layout
mkdir -p "$(dirname "${SLAKSHNA_UV_ENVIRONMENT}")"

python_path="$("${SLAKSHNA_UV_BIN}" python find "${SLAKSHNA_PYTHON_VERSION}")"

echo
echo "Synchronizing profile '${profile}' from the frozen lockfile..."
"${SLAKSHNA_UV_BIN}" sync \
    --project "${SLAKSHNA_ENV_PROJECT}" \
    --python "${python_path}" \
    --frozen \
    --reinstall-package bhaskera

"${SLAKSHNA_UV_BIN}" pip check --python "${SLAKSHNA_UV_ENVIRONMENT}/bin/python"

manifest_dir="${SLAKSHNA_RUNTIME_ROOT}/manifests/${profile}"
mkdir -p "${manifest_dir}"
"${SLAKSHNA_UV_BIN}" pip freeze \
    --python "${SLAKSHNA_UV_ENVIRONMENT}/bin/python" \
    > "${manifest_dir}/packages.txt"
sha256sum "${SLAKSHNA_ENV_PROJECT}/uv.lock" > "${manifest_dir}/uv-lock.sha256"
cp "${SLAKSHNA_BHASKERA_REVISION_FILE}" "${manifest_dir}/bhaskera-source-revision.txt"

echo
"${SLAKSHNA_UV_ENVIRONMENT}/bin/python" --version
echo "Environment ready: ${SLAKSHNA_UV_ENVIRONMENT}"
echo "Activate it through: source scripts/cluster/activate.sh"
