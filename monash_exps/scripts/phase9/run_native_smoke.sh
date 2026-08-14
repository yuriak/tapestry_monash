#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
activation_script="${slakshna_root}/Bhaskera/phase9-activate.sh"
native_manifest="${experiment_root}/.runtime/manifests/phase9/prepare-native.json"
rust_binary="${experiment_root}/.runtime/cargo-target/phase9-stock/release/iiitd"

[[ -f "${activation_script}" ]] || {
    echo "Missing Phase 9 environment. Run bash 1_setup_env.sh first." >&2
    exit 1
}
# shellcheck source=/dev/null
source "${activation_script}"

[[ -s "${native_manifest}" && -x "${rust_binary}" ]] || {
    echo "Missing native preparation outputs. Run bash 3_prepare_native.sh first." >&2
    exit 1
}

python_bin="${SLAKSHNA_UV_ENVIRONMENT}/bin/python"
"${python_bin}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Phase 9 native smoke requires a visible GPU")
if "A100" not in torch.cuda.get_device_name(0).upper():
    raise SystemExit(f"Expected an A100, got {torch.cuda.get_device_name(0)}")
print(f"Phase 9 smoke GPU: {torch.cuda.get_device_name(0)}")
PY

allocated_cpus="${SLURM_CPUS_ON_NODE:-$(nproc)}"
allocated_cpus="${allocated_cpus%%(*}"
if [[ ! "${allocated_cpus}" =~ ^[0-9]+$ ]] || (( allocated_cpus < 8 )); then
    echo "Phase 9 native smoke requires at least 8 allocated CPUs, got ${allocated_cpus}" >&2
    exit 1
fi

run_id="${PHASE9_SMOKE_RUN_ID:-$(date '+%Y%m%dT%H%M%S')_${SLURM_JOB_ID:-interactive}_native-smoke}"
if [[ ! "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid PHASE9_SMOKE_RUN_ID: ${run_id}" >&2
    exit 1
fi
run_root="${experiment_root}/artifacts/phase9/${run_id}"
epoch_duration="${PHASE9_SMOKE_EPOCH_DURATION:-120}"
timeout_seconds="${PHASE9_SMOKE_TIMEOUT:-1800}"

module_gcc_runtime="$(dirname "$(realpath "$(gcc -print-file-name=libstdc++.so.6)")")"
rust_ld_library_path=""
IFS=':' read -r -a ld_entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${ld_entries[@]}"; do
    if [[ -z "${entry}" || "${entry}" == "${module_gcc_runtime}" ]]; then continue; fi
    [[ -z "${rust_ld_library_path}" ]] || rust_ld_library_path+=":"
    rust_ld_library_path+="${entry}"
done
export PHASE9_RUST_LD_LIBRARY_PATH="${rust_ld_library_path}"

echo "Phase 9 stock native smoke"
echo "  run id         : ${run_id}"
echo "  artifact root  : ${run_root}"
echo "  epoch duration : ${epoch_duration}s"
echo "  timeout        : ${timeout_seconds}s"

"${python_bin}" "${experiment_root}/src/phase9/run_native_smoke.py" \
    --slakshna-root "${slakshna_root}" \
    --rust-binary "${rust_binary}" \
    --native-manifest "${native_manifest}" \
    --run-root "${run_root}" \
    --epoch-duration "${epoch_duration}" \
    --timeout "${timeout_seconds}"
