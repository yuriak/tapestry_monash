#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
activation_script="${slakshna_root}/Bhaskera/phase9-activate.sh"
expected_revision="9f93ec45ae0d3eb9c901aff3b50d4325b5050488"
data_root="${PHASE9_CULTURE_OUTPUT_ROOT:-${experiment_root}/.runtime/data/phase9/culture-instruct-alpaca-olmo1b-v1}"
model_root="${PHASE9_MODEL_ROOT:-${experiment_root}/.runtime/models/phase9/OLMo-1B-hf-aee7752d9c08}"
tokenized_root="${PHASE9_TOKENIZED_ROOT:-${experiment_root}/.runtime/data/phase9/tokenized-olmo1b}"
manifest_root="${experiment_root}/.runtime/manifests/phase9"
manifest="${manifest_root}/prepare-native.json"
cross_countries_source="${experiment_root}/configs/phase9/cross_countries_fl.yaml"
cross_countries_config="${slakshna_root}/configs/phase9/cross_countries_fl.yaml"

[[ -f "${activation_script}" ]] || {
    echo "Missing Phase 9 environment. Run bash monash_exps/scripts/phase9/setup_env.sh first." >&2
    exit 1
}
# shellcheck source=/dev/null
source "${activation_script}"

current_revision="$(git -C "${slakshna_root}" rev-parse HEAD)"
[[ "${current_revision}" == "${expected_revision}" ]] || {
    echo "Slakshna revision mismatch: ${current_revision}; expected ${expected_revision}" >&2
    exit 1
}
[[ -s "${data_root}/manifest.json" ]] || {
    echo "Missing prepared Phase 9 data. Run bash monash_exps/scripts/phase9/prepare_data.sh first." >&2
    exit 1
}

python_bin="${SLAKSHNA_UV_ENVIRONMENT}/bin/python"
[[ -x "${python_bin}" ]] || {
    echo "Missing Phase 9 Python: ${python_bin}" >&2
    exit 1
}
"${python_bin}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Phase 9 native preparation requires a visible GPU")
if "A100" not in torch.cuda.get_device_name(0).upper():
    raise SystemExit(f"Expected an A100, got {torch.cuda.get_device_name(0)}")
print(f"Phase 9 preparation GPU: {torch.cuda.get_device_name(0)}")
PY

allocated_cpus="${SLURM_CPUS_ON_NODE:-$(nproc)}"
allocated_cpus="${allocated_cpus%%(*}"
if [[ ! "${allocated_cpus}" =~ ^[0-9]+$ ]]; then allocated_cpus=8; fi
tokenizer_workers="${PHASE9_TOKENIZER_WORKERS:-$(( allocated_cpus < 8 ? allocated_cpus : 8 ))}"
if (( tokenizer_workers < 1 )); then tokenizer_workers=1; fi

mkdir -p "${manifest_root}" "${experiment_root}/.runtime/logs/phase9"
mkdir -p "$(dirname "${cross_countries_config}")"
install -m 0644 "${cross_countries_source}" "${cross_countries_config}"
run_stamp="$(date +%Y%m%dT%H%M%S)"
log_path="${experiment_root}/.runtime/logs/phase9/prepare-native-${run_stamp}.log"

echo "Phase 9 native preparation"
echo "  model root      : ${model_root}"
echo "  data root       : ${data_root}"
echo "  tokenized root  : ${tokenized_root}"
echo "  tokenizer workers: ${tokenizer_workers}"
echo "  log             : ${log_path}"

"${python_bin}" "${experiment_root}/src/phase9/prepare_native.py" \
    --slakshna-root "${slakshna_root}" \
    --data-root "${data_root}" \
    --model-root "${model_root}" \
    --tokenized-root "${tokenized_root}" \
    --manifest "${manifest}" \
    --workers "${tokenizer_workers}" \
    2>&1 | tee "${log_path}"

echo
echo "=== Build stock Slakshna release binary ==="
cargo_home="${experiment_root}/.runtime/cargo"
rustup_home="${experiment_root}/.runtime/rustup"
if [[ -x "${cargo_home}/bin/cargo" ]]; then
    export CARGO_HOME="${cargo_home}"
    export RUSTUP_HOME="${rustup_home}"
    export PATH="${CARGO_HOME}/bin:${PATH}"
fi
command -v cargo >/dev/null 2>&1 || {
    echo "cargo is unavailable; install the workspace-local Rust toolchain first" >&2
    exit 1
}
export CARGO_TARGET_DIR="${experiment_root}/.runtime/cargo-target/phase9-stock"
mkdir -p "${CARGO_TARGET_DIR}"

libclang_path="${PHASE9_LIBCLANG_PATH:-/usr/lib64/llvm21/lib64}"
[[ -d "${libclang_path}" ]] || {
    echo "libclang directory is unavailable: ${libclang_path}" >&2
    exit 1
}
module_gcc_runtime="$(dirname "$(realpath "$(gcc -print-file-name=libstdc++.so.6)")")"
rust_ld_library_path=""
IFS=':' read -r -a ld_entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${ld_entries[@]}"; do
    if [[ -z "${entry}" || "${entry}" == "${module_gcc_runtime}" ]]; then continue; fi
    [[ -z "${rust_ld_library_path}" ]] || rust_ld_library_path+=":"
    rust_ld_library_path+="${entry}"
done

env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u LIBRARY_PATH -u CPATH \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ LIBCLANG_PATH="${libclang_path}" \
    LD_LIBRARY_PATH="${rust_ld_library_path}" \
    cargo build --locked --release --manifest-path "${slakshna_root}/Cargo.toml" \
    2>&1 | tee -a "${log_path}"

rust_binary="${CARGO_TARGET_DIR}/release/iiitd"
[[ -x "${rust_binary}" ]] || {
    echo "Rust build did not produce ${rust_binary}" >&2
    exit 1
}
rust_manifest="${manifest_root}/rust-build.txt"
{
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "slakshna_revision=${current_revision}"
    echo "cargo_lock_sha256=$(sha256sum "${slakshna_root}/Cargo.lock" | awk '{print $1}')"
    echo "binary=${rust_binary}"
    echo "binary_sha256=$(sha256sum "${rust_binary}" | awk '{print $1}')"
    rustc --version
    cargo --version
} > "${rust_manifest}"

echo
echo "PHASE 9 NATIVE PREPARATION PASSED"
echo "Active config : ${slakshna_root}/node_template.yaml"
echo "Cross-country : ${cross_countries_config}"
echo "Data manifest : ${manifest}"
echo "Rust binary   : ${rust_binary}"
echo "Rust manifest : ${rust_manifest}"
echo "Log           : ${log_path}"
