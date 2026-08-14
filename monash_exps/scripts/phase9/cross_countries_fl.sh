#!/usr/bin/env bash
set -euo pipefail

action="${1:-local}"
site="${2:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
activation_script="${slakshna_root}/Bhaskera/phase9-activate.sh"
default_config="${slakshna_root}/configs/phase9/cross_countries_fl.yaml"
tracked_config="${experiment_root}/configs/phase9/cross_countries_fl.yaml"
config_path="${PHASE9_CROSS_COUNTRIES_CONFIG:-${default_config}}"
python_driver="${experiment_root}/src/phase9/run_cross_countries.py"

if [[ -z "${PHASE9_CROSS_COUNTRIES_CONFIG:-}" && ! -f "${default_config}" ]]; then
    mkdir -p "$(dirname "${default_config}")"
    install -m 0644 "${tracked_config}" "${default_config}"
    echo "Installed the editable deployment config at ${default_config}"
fi

[[ -f "${activation_script}" ]] || {
    echo "Missing Phase 9 environment. Run bash 1_setup_env.sh first." >&2
    exit 1
}
# shellcheck source=/dev/null
source "${activation_script}"
python_bin="${SLAKSHNA_UV_ENVIRONMENT}/bin/python"

# The Rust binary needs the system libstdc++ rather than the older GCC module
# runtime loaded for building flash-attn.  Compute this before every action so
# the identity-only workflow uses the same known-good runtime as training.
module_gcc_runtime="$(dirname "$(realpath "$(gcc -print-file-name=libstdc++.so.6)")")"
rust_ld_library_path=""
IFS=':' read -r -a ld_entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${ld_entries[@]}"; do
    if [[ -z "${entry}" || "${entry}" == "${module_gcc_runtime}" ]]; then continue; fi
    [[ -z "${rust_ld_library_path}" ]] || rust_ld_library_path+=":"
    rust_ld_library_path+="${entry}"
done
export PHASE9_RUST_LD_LIBRARY_PATH="${rust_ld_library_path}"

case "${action}" in
    claim-agent)
        PHASE9_PLAYIT_REQUIRED_TUNNELS="${PHASE9_PLAYIT_REQUIRED_TUNNELS:-1}" \
            bash "${script_dir}/playit_agent.sh" claim
        exit 0
        ;;
    agent-status)
        bash "${script_dir}/playit_agent.sh" status
        exit 0
        ;;
    stop-agent)
        bash "${script_dir}/playit_agent.sh" stop
        exit 0
        ;;
    identity)
        [[ "${site}" == "australia" || "${site}" == "india" ]] || {
            echo "Usage: $0 identity australia|india" >&2
            exit 2
        }
        exec "${python_bin}" "${python_driver}" identity \
            --config "${config_path}" --site "${site}"
        ;;
    local)
        required_tunnels=2
        ;;
    site)
        [[ "${site}" == "australia" || "${site}" == "india" ]] || {
            echo "Usage: $0 site australia|india" >&2
            exit 2
        }
        required_tunnels=1
        ;;
    *)
        echo "Usage: $0 local | identity SITE | site SITE | claim-agent | agent-status | stop-agent" >&2
        exit 2
        ;;
esac

"${python_bin}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Cross-countries FL requires a visible GPU")
print(f"Visible GPU count: {torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"  GPU {index}: {torch.cuda.get_device_name(index)}")
PY

allocated_cpus="${SLURM_CPUS_ON_NODE:-$(nproc)}"
allocated_cpus="${allocated_cpus%%(*}"
if [[ ! "${allocated_cpus}" =~ ^[0-9]+$ ]] || (( allocated_cpus < 16 )); then
    echo "Cross-countries FL requires at least 16 allocated CPUs, got ${allocated_cpus}" >&2
    exit 1
fi

cleanup() {
    bash "${script_dir}/playit_agent.sh" stop >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM
PHASE9_PLAYIT_REQUIRED_TUNNELS="${required_tunnels}" \
    bash "${script_dir}/playit_agent.sh" start

if [[ "${action}" == "local" ]]; then
    "${python_bin}" "${python_driver}" local --config "${config_path}"
else
    "${python_bin}" "${python_driver}" site --config "${config_path}" --site "${site}"
fi
