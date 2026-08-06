#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This adapter must be sourced: source scripts/cluster/fit.sh" >&2
    exit 2
fi

_fit_adapter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_fit_restore_nounset=0
case "$-" in
    *u*) _fit_restore_nounset=1; set +u ;;
esac

if type module >/dev/null 2>&1; then
    module load gcc/10.2.0
    module load cuda/12.8.1-none-none-6xn5wck
    if ! command -v git >/dev/null 2>&1; then
        module load git/2.42.0-gcc-8.5.0-2kerkdj
    fi
fi

if [[ "${_fit_restore_nounset}" -eq 1 ]]; then
    set -u
fi

export SLAKSHNA_CLUSTER="fit"
# shellcheck source=common.sh
source "${_fit_adapter_dir}/common.sh"

unset _fit_adapter_dir _fit_restore_nounset
