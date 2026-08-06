#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This library must be sourced by a cluster adapter." >&2
    exit 2
fi

_cluster_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../environment/common.sh
source "${_cluster_common_dir}/../environment/common.sh"

export UV_CACHE_DIR="${SLAKSHNA_UV_CACHE_DIR}"
export UV_PYTHON_INSTALL_DIR="${SLAKSHNA_UV_PYTHON_DIR}"
export UV_PYTHON_BIN_DIR="${SLAKSHNA_UV_PYTHON_BIN_DIR}"
export HF_HOME="${HF_HOME:-${SLAKSHNA_RUNTIME_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SLAKSHNA_RUNTIME_ROOT}/cache/huggingface-datasets}"
export TORCH_HOME="${TORCH_HOME:-${SLAKSHNA_RUNTIME_ROOT}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SLAKSHNA_RUNTIME_ROOT}/cache/xdg}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${HF_HOME}" \
    "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

if [[ -x "${SLAKSHNA_UV_ENVIRONMENT}/bin/python" ]]; then
    # Standard venv activation scripts embed the absolute creation path and
    # become stale when a checkout is renamed or moved. Activate from the
    # resolved experiment path instead; a frozen sync on each cluster remains
    # the supported way to create this directory.
    export VIRTUAL_ENV="${SLAKSHNA_UV_ENVIRONMENT}"
    case ":${PATH}:" in
        *":${VIRTUAL_ENV}/bin:"*) ;;
        *) export PATH="${VIRTUAL_ENV}/bin:${PATH}" ;;
    esac
    unset PYTHONHOME 2>/dev/null || true
    _cluster_python="${VIRTUAL_ENV}/bin/python"
else
    _cluster_python="not installed yet"
    if [[ -d "${SLAKSHNA_UV_ENVIRONMENT}" ]]; then
        echo "Phase 0 uv environment is stale or non-relocatable: ${SLAKSHNA_UV_ENVIRONMENT}" >&2
        echo "Re-run scripts/environment/03_sync_environment.sh for this checkout path." >&2
    else
        echo "Phase 0 uv environment does not exist yet: ${SLAKSHNA_UV_ENVIRONMENT}" >&2
    fi
fi

echo "Slakshna experiment environment active"
echo "  cluster              : ${SLAKSHNA_CLUSTER:-generic}"
echo "  CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "  CUDA_HOME            : ${CUDA_HOME:-unset}"
echo "  Python               : ${_cluster_python}"
echo "  environment          : ${SLAKSHNA_UV_ENVIRONMENT}"
echo "  experiment root      : ${SLAKSHNA_EXPERIMENT_ROOT}"
echo "  Slakshna source      : ${SLAKSHNA_SOURCE_ROOT}"
echo "  Hugging Face cache   : ${HF_HOME}"

unset _cluster_common_dir _cluster_python
