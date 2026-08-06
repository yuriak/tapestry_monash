#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This adapter must be sourced: source scripts/cluster/m3.sh" >&2
    exit 2
fi

_m3_adapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_m3_restore_nounset=0
case "$-" in
    *u*) _m3_restore_nounset=1; set +u ;;
esac

# M3 module labels can change independently of the experiment. Set these
# variables to the labels reported by `module avail` on M3. Leaving one empty
# means the submitting shell/allocation is responsible for preloading it.
if type module >/dev/null 2>&1; then
    [[ -z "${SLAKSHNA_M3_COMPILER_MODULE:-}" ]] || module load "${SLAKSHNA_M3_COMPILER_MODULE}"
    [[ -z "${SLAKSHNA_M3_CUDA_MODULE:-}" ]] || module load "${SLAKSHNA_M3_CUDA_MODULE}"
    if ! command -v git >/dev/null 2>&1 && [[ -n "${SLAKSHNA_M3_GIT_MODULE:-}" ]]; then
        module load "${SLAKSHNA_M3_GIT_MODULE}"
    fi
fi

if [[ "${_m3_restore_nounset}" -eq 1 ]]; then
    set -u
fi

export SLAKSHNA_CLUSTER="m3"
# shellcheck source=common.sh
source "${_m3_adapter_dir}/common.sh"

unset _m3_adapter_dir _m3_restore_nounset
