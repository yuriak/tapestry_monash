#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "common.sh is a library and must be sourced by another script." >&2
    exit 2
fi

_phase0_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLAKSHNA_EXPERIMENT_ROOT="${SLAKSHNA_EXPERIMENT_ROOT:-$(cd "${_phase0_common_dir}/../.." && pwd)}"
SLAKSHNA_WORKSPACE_ROOT="${SLAKSHNA_WORKSPACE_ROOT:-$(cd "${SLAKSHNA_EXPERIMENT_ROOT}/.." && pwd)}"
SLAKSHNA_SOURCE_ROOT="${SLAKSHNA_SOURCE_ROOT:-${SLAKSHNA_WORKSPACE_ROOT}/Slakshna}"
SLAKSHNA_ENV_PROJECT="${SLAKSHNA_ENV_PROJECT:-${SLAKSHNA_EXPERIMENT_ROOT}/environment}"
SLAKSHNA_RUNTIME_ROOT="${SLAKSHNA_RUNTIME_ROOT:-${SLAKSHNA_EXPERIMENT_ROOT}/.runtime}"
SLAKSHNA_TOOL_ROOT="${SLAKSHNA_TOOL_ROOT:-${SLAKSHNA_RUNTIME_ROOT}/tools}"
SLAKSHNA_UV_BIN="${SLAKSHNA_UV_BIN:-${SLAKSHNA_TOOL_ROOT}/uv/bin/uv}"
SLAKSHNA_UV_CACHE_DIR="${SLAKSHNA_UV_CACHE_DIR:-${SLAKSHNA_RUNTIME_ROOT}/cache/uv}"
SLAKSHNA_UV_PYTHON_DIR="${SLAKSHNA_UV_PYTHON_DIR:-${SLAKSHNA_RUNTIME_ROOT}/python}"
SLAKSHNA_UV_PYTHON_BIN_DIR="${SLAKSHNA_UV_PYTHON_BIN_DIR:-${SLAKSHNA_TOOL_ROOT}/python-bin}"
SLAKSHNA_UV_ENVIRONMENT="${SLAKSHNA_UV_ENVIRONMENT:-${SLAKSHNA_RUNTIME_ROOT}/venvs/primary}"
SLAKSHNA_BHASKERA_SNAPSHOT="${SLAKSHNA_BHASKERA_SNAPSHOT:-${SLAKSHNA_RUNTIME_ROOT}/sources/Bhaskera}"
SLAKSHNA_BHASKERA_REVISION_FILE="${SLAKSHNA_BHASKERA_REVISION_FILE:-${SLAKSHNA_ENV_PROJECT}/bhaskera-source-revision.txt}"
SLAKSHNA_PYTHON_VERSION="${SLAKSHNA_PYTHON_VERSION:-3.11.13}"

export SLAKSHNA_EXPERIMENT_ROOT SLAKSHNA_WORKSPACE_ROOT SLAKSHNA_SOURCE_ROOT
export SLAKSHNA_ENV_PROJECT SLAKSHNA_RUNTIME_ROOT SLAKSHNA_TOOL_ROOT
export SLAKSHNA_UV_BIN SLAKSHNA_UV_CACHE_DIR SLAKSHNA_UV_PYTHON_DIR
export SLAKSHNA_UV_PYTHON_BIN_DIR
export SLAKSHNA_UV_ENVIRONMENT SLAKSHNA_BHASKERA_SNAPSHOT
export SLAKSHNA_BHASKERA_REVISION_FILE SLAKSHNA_PYTHON_VERSION
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLAKSHNA_UV_CACHE_DIR}}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${SLAKSHNA_UV_PYTHON_DIR}}"
export UV_PYTHON_BIN_DIR="${UV_PYTHON_BIN_DIR:-${SLAKSHNA_UV_PYTHON_BIN_DIR}}"

phase0_require_layout() {
    [[ -f "${SLAKSHNA_ENV_PROJECT}/pyproject.toml" ]] || {
        echo "Missing environment specification: ${SLAKSHNA_ENV_PROJECT}/pyproject.toml" >&2
        return 1
    }
    [[ -f "${SLAKSHNA_SOURCE_ROOT}/Bhaskera/pyproject.toml" ]] || {
        echo "Missing Bhaskera submodule source: ${SLAKSHNA_SOURCE_ROOT}/Bhaskera" >&2
        return 1
    }
}

phase0_require_uv() {
    [[ -x "${SLAKSHNA_UV_BIN}" ]] || {
        echo "Project-local uv is missing. Run scripts/environment/01_install_uv.sh first." >&2
        return 1
    }
}

phase0_source_revision() {
    command -v git >/dev/null 2>&1 || {
        echo "git is required to snapshot the Bhaskera source." >&2
        return 1
    }
    git -C "${SLAKSHNA_SOURCE_ROOT}" rev-parse HEAD
}

phase0_require_clean_submodule() {
    local status
    status="$(git -C "${SLAKSHNA_SOURCE_ROOT}" status --porcelain)"
    if [[ -n "${status}" ]]; then
        echo "Slakshna submodule is not clean:" >&2
        echo "${status}" >&2
        return 1
    fi
}

phase0_stage_bhaskera() {
    phase0_require_layout
    command -v tar >/dev/null 2>&1 || {
        echo "tar is required to stage the Bhaskera source." >&2
        return 1
    }
    phase0_require_clean_submodule

    local revision source_parent staging previous
    revision="$(phase0_source_revision)"

    source_parent="$(dirname "${SLAKSHNA_BHASKERA_SNAPSHOT}")"
    mkdir -p "${source_parent}"
    staging="$(mktemp -d "${source_parent}/.Bhaskera-stage.XXXXXX")"

    if ! git -C "${SLAKSHNA_SOURCE_ROOT}" archive HEAD:Bhaskera \
        | tar -x -C "${staging}"; then
        rm -rf -- "${staging}"
        echo "Failed to create a tracked Bhaskera source snapshot." >&2
        return 1
    fi
    printf '%s\n' "${revision}" > "${staging}/.slakshna-source-revision"

    previous=""
    if [[ -e "${SLAKSHNA_BHASKERA_SNAPSHOT}" ]]; then
        previous="${source_parent}/.Bhaskera-previous.$$"
        mv "${SLAKSHNA_BHASKERA_SNAPSHOT}" "${previous}"
    fi
    mv "${staging}" "${SLAKSHNA_BHASKERA_SNAPSHOT}"
    if [[ -n "${previous}" ]]; then
        rm -rf -- "${previous}"
    fi

    echo "Staged Bhaskera ${revision} at ${SLAKSHNA_BHASKERA_SNAPSHOT}"
}

phase0_require_locked_source_revision() {
    [[ -f "${SLAKSHNA_BHASKERA_REVISION_FILE}" ]] || {
        echo "Missing ${SLAKSHNA_BHASKERA_REVISION_FILE}. Re-run 02_lock_environment.sh." >&2
        return 1
    }
    local locked current
    locked="$(< "${SLAKSHNA_BHASKERA_REVISION_FILE}")"
    current="$(phase0_source_revision)"
    if [[ "${locked}" != "${current}" ]]; then
        echo "Bhaskera lock/source mismatch: lock=${locked}, submodule=${current}" >&2
        echo "Re-run 02_lock_environment.sh and review the updated lock." >&2
        return 1
    fi
}

phase0_print_layout() {
    echo "Phase 0 environment layout"
    echo "  workspace   : ${SLAKSHNA_WORKSPACE_ROOT}"
    echo "  experiments : ${SLAKSHNA_EXPERIMENT_ROOT}"
    echo "  submodule   : ${SLAKSHNA_SOURCE_ROOT}"
    echo "  uv project  : ${SLAKSHNA_ENV_PROJECT}"
    echo "  uv binary   : ${SLAKSHNA_UV_BIN}"
    echo "  uv cache    : ${UV_CACHE_DIR}"
    echo "  uv Python   : ${UV_PYTHON_INSTALL_DIR}"
    echo "  Python links: ${UV_PYTHON_BIN_DIR}"
    echo "  environment : ${SLAKSHNA_UV_ENVIRONMENT}"
    echo "  Bhaskera src: ${SLAKSHNA_BHASKERA_SNAPSHOT}"
}

unset _phase0_common_dir
