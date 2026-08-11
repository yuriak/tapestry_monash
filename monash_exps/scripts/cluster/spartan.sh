#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This adapter must be sourced: source scripts/cluster/spartan.sh" >&2
    exit 2
fi

_spartan_adapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_spartan_restore_nounset=0
case "$-" in
    *u*) _spartan_restore_nounset=1; set +u ;;
esac

# Spartan module labels evolve independently of the experiment. Existing
# interactive allocations may already provide a suitable toolchain, while
# these optional variables allow a clean allocation to load site-specific
# compiler, CUDA, LLVM, and Git modules without embedding volatile labels in
# the portable repository.
if type module >/dev/null 2>&1; then
    [[ -z "${SLAKSHNA_SPARTAN_COMPILER_MODULE:-}" ]] || module load "${SLAKSHNA_SPARTAN_COMPILER_MODULE}"
    [[ -z "${SLAKSHNA_SPARTAN_CUDA_MODULE:-}" ]] || module load "${SLAKSHNA_SPARTAN_CUDA_MODULE}"
    [[ -z "${SLAKSHNA_SPARTAN_LLVM_MODULE:-}" ]] || module load "${SLAKSHNA_SPARTAN_LLVM_MODULE}"
    if ! command -v git >/dev/null 2>&1 && [[ -n "${SLAKSHNA_SPARTAN_GIT_MODULE:-}" ]]; then
        module load "${SLAKSHNA_SPARTAN_GIT_MODULE}"
    fi
fi

if [[ "${_spartan_restore_nounset}" -eq 1 ]]; then
    set -u
fi

export SLAKSHNA_CLUSTER="spartan"
# shellcheck source=common.sh
source "${_spartan_adapter_dir}/common.sh"

unset _spartan_adapter_dir _spartan_restore_nounset
