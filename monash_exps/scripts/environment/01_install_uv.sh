#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

phase0_require_layout
phase0_print_layout

if [[ -x "${SLAKSHNA_UV_BIN}" ]]; then
    echo
    echo "Project-local uv is already installed."
    "${SLAKSHNA_UV_BIN}" --version
    exit 0
fi

command -v curl >/dev/null 2>&1 || {
    echo "curl is required to install uv." >&2
    exit 1
}

uv_install_dir="$(dirname "${SLAKSHNA_UV_BIN}")"
mkdir -p "${uv_install_dir}" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"
installer_dir="$(mktemp -d /tmp/slakshna-uv-install.XXXXXX)"
trap 'rm -rf "${installer_dir}"' EXIT

echo
echo "Downloading the official uv installer..."
curl --proto '=https' --tlsv1.2 --fail --show-error --location \
    https://astral.sh/uv/install.sh \
    --output "${installer_dir}/install.sh"

UV_INSTALL_DIR="${uv_install_dir}" UV_NO_MODIFY_PATH=1 \
    sh "${installer_dir}/install.sh"

if [[ ! -x "${SLAKSHNA_UV_BIN}" ]]; then
    echo "uv installer completed but ${SLAKSHNA_UV_BIN} is missing." >&2
    exit 1
fi

echo
"${SLAKSHNA_UV_BIN}" --version
echo "uv installation complete; no shell profile or Conda environment was modified."
