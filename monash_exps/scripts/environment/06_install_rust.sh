#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
cargo_home="${experiment_root}/.runtime/cargo"
rustup_home="${experiment_root}/.runtime/rustup"
installer="${experiment_root}/.runtime/downloads/rustup-init"

mkdir -p "$(dirname "${installer}")" "${cargo_home}" "${rustup_home}"

if [[ ! -x "${cargo_home}/bin/cargo" ]]; then
    echo "Downloading the official rustup installer..."
    curl --proto '=https' --tlsv1.2 --fail --location \
        https://sh.rustup.rs -o "${installer}"
    chmod 700 "${installer}"
    CARGO_HOME="${cargo_home}" RUSTUP_HOME="${rustup_home}" \
        "${installer}" -y --no-modify-path --profile minimal --default-toolchain stable
fi

export CARGO_HOME="${cargo_home}"
export RUSTUP_HOME="${rustup_home}"
export PATH="${CARGO_HOME}/bin:${PATH}"

rustc --version
cargo --version
rustup show active-toolchain
