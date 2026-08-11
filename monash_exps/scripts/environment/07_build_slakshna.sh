#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
cd "${workspace_root}"

# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

cargo_home="${experiment_root}/.runtime/cargo"
rustup_home="${experiment_root}/.runtime/rustup"
export CARGO_HOME="${cargo_home}"
export RUSTUP_HOME="${rustup_home}"
export PATH="${CARGO_HOME}/bin:${PATH}"
export CARGO_TARGET_DIR="${experiment_root}/.runtime/cargo-target/slakshna"

command -v cargo >/dev/null 2>&1 || {
    echo "Workspace-local cargo is unavailable; run 06_install_rust.sh first." >&2
    exit 1
}
command -v gcc >/dev/null 2>&1 || {
    echo "A C compiler is required to build Slakshna." >&2
    exit 1
}

mkdir -p "${CARGO_TARGET_DIR}" "${experiment_root}/.runtime/manifests/primary"

# CUDA-compatible compiler modules can prepend an older libstdc++ that cannot
# load rustc. Retain every LD path except the active GCC module runtime while
# Rust and native dependencies build.
module_gcc_runtime="$(dirname "$(realpath "$(gcc -print-file-name=libstdc++.so.6)")")"
rust_ld_library_path=""
IFS=':' read -r -a ld_entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${ld_entries[@]}"; do
    if [[ -z "${entry}" || "${entry}" == "${module_gcc_runtime}" ]]; then
        continue
    fi
    [[ -z "${rust_ld_library_path}" ]] || rust_ld_library_path+=":"
    rust_ld_library_path+="${entry}"
done

libclang_path="${SLAKSHNA_LIBCLANG_PATH:-}"
if [[ -z "${libclang_path}" ]] && command -v llvm-config >/dev/null 2>&1; then
    candidate="$(llvm-config --libdir 2>/dev/null || true)"
    if compgen -G "${candidate}/libclang.so*" >/dev/null; then
        libclang_path="${candidate}"
    fi
fi
if [[ -z "${libclang_path}" ]]; then
    shopt -s nullglob
    for candidate in /usr/lib64/llvm*/lib64 /usr/lib64/llvm*/lib /usr/lib/llvm-*/lib; do
        if compgen -G "${candidate}/libclang.so*" >/dev/null; then
            libclang_path="${candidate}"
            break
        fi
    done
    shopt -u nullglob
fi
if [[ -z "${libclang_path}" ]]; then
    echo "libclang was not found; set SLAKSHNA_LIBCLANG_PATH to its library directory." >&2
    exit 1
fi

cc="${SLAKSHNA_BUILD_CC:-}"
cxx="${SLAKSHNA_BUILD_CXX:-}"
if [[ -z "${cc}" ]]; then
    [[ -x /usr/bin/gcc ]] && cc=/usr/bin/gcc || cc="$(command -v gcc)"
fi
if [[ -z "${cxx}" ]]; then
    [[ -x /usr/bin/g++ ]] && cxx=/usr/bin/g++ || cxx="$(command -v g++)"
fi

echo "Building Slakshna $(git -C Slakshna rev-parse HEAD) with the locked Cargo graph..."
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u LIBRARY_PATH -u CPATH \
    CC="${cc}" CXX="${cxx}" LIBCLANG_PATH="${libclang_path}" \
    LD_LIBRARY_PATH="${rust_ld_library_path}" \
    cargo build --locked --release --manifest-path Slakshna/Cargo.toml

rust_binary="${CARGO_TARGET_DIR}/release/iiitd"
test -x "${rust_binary}"
manifest="${experiment_root}/.runtime/manifests/primary/slakshna-build.txt"
{
    echo "revision=$(git -C Slakshna rev-parse HEAD)"
    echo "cargo_lock_sha256=$(sha256sum Slakshna/Cargo.lock | awk '{print $1}')"
    echo "binary=${rust_binary}"
    echo "binary_sha256=$(sha256sum "${rust_binary}" | awk '{print $1}')"
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
    echo "cc=${cc}"
    echo "cxx=${cxx}"
    echo "libclang_path=${libclang_path}"
    echo "runtime_ld_library_path=${rust_ld_library_path}"
} > "${manifest}"

echo "Slakshna release binary ready: ${rust_binary}"
echo "Build manifest: ${manifest}"
