#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
cd "${workspace_root}"

# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Phase 3 requires a CUDA GPU")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"Phase 3 requires exactly one visible GPU, got {torch.cuda.device_count()}")
print(f"Phase 3 GPU: {torch.cuda.get_device_name(0)}")
PY

cargo_home="${experiment_root}/.runtime/cargo"
rustup_home="${experiment_root}/.runtime/rustup"
if [[ -x "${cargo_home}/bin/cargo" ]]; then
    export CARGO_HOME="${cargo_home}"
    export RUSTUP_HOME="${rustup_home}"
    export PATH="${CARGO_HOME}/bin:${PATH}"
fi
if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo is unavailable. Run monash_exps/scripts/environment/06_install_rust.sh first." >&2
    exit 2
fi

run_id="${1:-$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID:-interactive}_phase3}"
artifact_root="${experiment_root}/artifacts/phase3/${run_id}"
runtime_dir="${artifact_root}/runtime"
data_dir="${artifact_root}/node-data"
mkdir -p "${artifact_root}"

export CARGO_TARGET_DIR="${experiment_root}/.runtime/cargo-target/slakshna"
mkdir -p "${CARGO_TARGET_DIR}"

# M3's CUDA-compatible GCC 10 module prepends an older libstdc++ which cannot
# load the system LLVM 21 libclang used by bindgen. Compile and run the Rust
# boundary with the host GCC 11 runtime while retaining CUDA's library paths
# for the inherited Python training child.
module_gcc_runtime="$(dirname "$(realpath "$(gcc -print-file-name=libstdc++.so.6)")")"
rust_ld_library_path=""
IFS=':' read -r -a ld_entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${ld_entries[@]}"; do
    if [[ -z "${entry}" || "${entry}" == "${module_gcc_runtime}" ]]; then
        continue
    fi
    if [[ -n "${rust_ld_library_path}" ]]; then
        rust_ld_library_path+=":"
    fi
    rust_ld_library_path+="${entry}"
done
libclang_path="/usr/lib64/llvm21/lib64"
test -f "${libclang_path}/libclang.so.21.1.8"

{
    rustc --version
    cargo --version
    echo "cargo_target_dir=${CARGO_TARGET_DIR}"
    echo "slakshna_revision=$(git -C Slakshna rev-parse HEAD)"
    echo "cargo_lock_sha256=$(sha256sum Slakshna/Cargo.lock | awk '{print $1}')"
    echo "rust_cc=/usr/bin/gcc"
    echo "rust_cxx=/usr/bin/g++"
    echo "libclang_path=${libclang_path}"
    echo "excluded_module_gcc_runtime=${module_gcc_runtime}"
} > "${artifact_root}/rust-toolchain.txt"

echo "=== Build pinned Slakshna Rust binary ==="
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u LIBRARY_PATH -u CPATH \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    LIBCLANG_PATH="${libclang_path}" \
    LD_LIBRARY_PATH="${rust_ld_library_path}" \
    cargo build --locked --release --manifest-path Slakshna/Cargo.toml \
    2>&1 | tee "${artifact_root}/cargo-build.log"
rust_binary="${CARGO_TARGET_DIR}/release/iiitd"
test -x "${rust_binary}"

read -r p2p_port ws_port api_port < <(python - <<'PY'
import socket

sockets = []
ports = []
for _ in range(3):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sockets.append(sock)
    ports.append(sock.getsockname()[1])
print(*ports)
for sock in sockets:
    sock.close()
PY
)

python "${experiment_root}/src/phase3/prepare_runtime.py" \
    --template "${experiment_root}/configs/phase3/single_node.toml" \
    --bridge "${experiment_root}/src/phase3/ml_bridge.py" \
    --runtime-dir "${runtime_dir}" \
    --data-dir "${data_dir}" \
    --p2p-port "${p2p_port}" \
    --ws-port "${ws_port}" \
    --api-port "${api_port}"

export SLAKSHNA_PHASE3_ARTIFACT_ROOT="${artifact_root}"
export SLAKSHNA_EXPERIMENT_ROOT="${experiment_root}"
export SLAKSHNA_PHASE3_RUN_ID="${run_id}"
export CUDA_VISIBLE_DEVICES=0
export SLAKSHNA_EXPECTED_GPUS=1

rust_log="${artifact_root}/rust-node.log"
rust_pid=""
cleanup() {
    if [[ -n "${rust_pid}" ]] && kill -0 "${rust_pid}" 2>/dev/null; then
        kill -TERM "${rust_pid}" 2>/dev/null || true
        wait "${rust_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "=== Start real single-node Slakshna lifecycle ==="
(
    cd "${runtime_dir}"
    exec env LD_LIBRARY_PATH="${rust_ld_library_path}" \
        "${rust_binary}" --config node.toml
) > "${rust_log}" 2>&1 &
rust_pid=$!
echo "rust_pid=${rust_pid} api_port=${api_port}"

api_base="http://127.0.0.1:${api_port}"
api_ready=0
for _ in $(seq 1 120); do
    if ! kill -0 "${rust_pid}" 2>/dev/null; then
        echo "Slakshna exited before its API became ready" >&2
        tail -n 120 "${rust_log}" >&2
        exit 1
    fi
    if curl --silent --fail "${api_base}/status" > "${artifact_root}/api-status.json.tmp"; then
        mv "${artifact_root}/api-status.json.tmp" "${artifact_root}/api-status.json"
        api_ready=1
        break
    fi
    sleep 1
done
if [[ "${api_ready}" != 1 ]]; then
    echo "Slakshna API did not become ready" >&2
    exit 1
fi

update_ready=0
for _ in $(seq 1 1800); do
    if ! kill -0 "${rust_pid}" 2>/dev/null; then
        echo "Slakshna exited before recording its update" >&2
        tail -n 160 "${rust_log}" >&2
        exit 1
    fi
    if curl --silent --fail "${api_base}/updates" > "${artifact_root}/api-updates.json.tmp"; then
        if jq -e '.success == true and (.updates | length) > 0' \
            "${artifact_root}/api-updates.json.tmp" >/dev/null; then
            mv "${artifact_root}/api-updates.json.tmp" "${artifact_root}/api-updates.json"
            update_ready=1
            break
        fi
    fi
    sleep 1
done
if [[ "${update_ready}" != 1 ]]; then
    echo "Slakshna did not record an update within 30 minutes" >&2
    exit 1
fi

# Capture status after the completed lifecycle, then stop the perpetual node.
curl --silent --fail "${api_base}/status" > "${artifact_root}/api-status.json"
cleanup
rust_pid=""
trap - EXIT INT TERM

python "${experiment_root}/src/phase3/verify_phase3.py" \
    --artifact-root "${artifact_root}" \
    --updates "${artifact_root}/api-updates.json" \
    --status "${artifact_root}/api-status.json" \
    --rust-log "${rust_log}" \
    --output "${artifact_root}/verification-summary.json"

if [[ -n "$(git -C Slakshna status --short)" ]]; then
    echo "Slakshna submodule became dirty" >&2
    git -C Slakshna status --short >&2
    exit 1
fi
{
    echo "revision=$(git -C Slakshna rev-parse HEAD)"
    echo "status=clean"
} > "${artifact_root}/source-integrity.txt"

echo "Phase 3 artifact: ${artifact_root}"
