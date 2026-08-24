#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runtime_root="${M0_RUNTIME_ROOT:-${repo_root}/Slakshna/m0_runtime}"
import_root="${M0_SPARTAN_IMPORT_ROOT:-${runtime_root}/imported_spartan}"
python_bin="${M0_PYTHON:-${repo_root}/Slakshna/Bhaskera/.venv-phase9/bin/python}"

[[ -x "${python_bin}" ]] || {
    echo "ERROR: Phase-9 Python is unavailable: ${python_bin}" >&2
    exit 1
}
[[ -d "${runtime_root}" ]] || {
    echo "ERROR: M0 runtime root is unavailable: ${runtime_root}" >&2
    exit 1
}

echo "M0 runtime       : $(realpath "${runtime_root}")"
echo "Spartan imports  : ${import_root}"
echo "Python           : ${python_bin}"
echo "DCP SHA-256      : enabled (set M0_SKIP_DCP_HASH=1 to skip)"

args=(
    "${repo_root}/monash_exps/src/m0/collect_training_results.py"
    --runtime-root "${runtime_root}"
    --import-root "${import_root}"
)
if [[ "${M0_SKIP_DCP_HASH:-0}" == "1" ]]; then
    args+=(--skip-dcp-hash)
fi
if [[ -n "${M0_COLLECT_RUNS:-}" ]]; then
    read -r -a selected_runs <<<"${M0_COLLECT_RUNS}"
    args+=(--runs "${selected_runs[@]}")
fi

exec "${python_bin}" "${args[@]}"
