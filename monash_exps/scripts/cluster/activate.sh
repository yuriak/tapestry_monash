#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This adapter must be sourced: source scripts/cluster/activate.sh" >&2
    exit 2
fi

_activate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_requested_cluster="${SLAKSHNA_CLUSTER:-}"

if [[ -z "${_requested_cluster}" ]]; then
    case "${SLURM_CLUSTER_NAME:-}" in
        *[Ff][Ii][Tt]*) _requested_cluster="fit" ;;
        *[Mm]3*) _requested_cluster="m3" ;;
    esac
fi

# FIT currently exposes this shared Spack tree. This fallback keeps existing
# interactive allocations convenient without coupling experiment code to FIT.
if [[ -z "${_requested_cluster}" && -d /cm/shared/apps/spack ]]; then
    _requested_cluster="fit"
fi

case "${_requested_cluster:-generic}" in
    fit|m3)
        # shellcheck source=/dev/null
        source "${_activate_dir}/${_requested_cluster}.sh"
        ;;
    generic)
        export SLAKSHNA_CLUSTER="generic"
        # shellcheck source=common.sh
        source "${_activate_dir}/common.sh"
        ;;
    *)
        echo "Unsupported SLAKSHNA_CLUSTER=${_requested_cluster}; expected fit, m3, or generic." >&2
        return 2
        ;;
esac

unset _activate_dir _requested_cluster
