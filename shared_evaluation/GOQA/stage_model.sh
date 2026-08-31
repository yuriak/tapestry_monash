#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 SOURCE_MODEL_DIR NODE_LOCAL_MODEL_DIR" >&2
    exit 2
fi

source_dir="$(cd "$1" && pwd -P)"
destination="$2"
marker="${destination}/.goqa-staged-source"

[[ -f "${source_dir}/config.json" ]] || {
    echo "Missing model config: ${source_dir}/config.json" >&2
    exit 1
}

if [[ -f "${marker}" ]] && [[ "$(<"${marker}")" == "${source_dir}" ]]; then
    echo "Reusing node-local GOQA model: ${destination}"
    exit 0
fi
if [[ -e "${destination}" ]]; then
    echo "Refusing to overwrite an incomplete or mismatched staging directory: ${destination}" >&2
    exit 1
fi

mkdir -p "$(dirname "${destination}")"
temporary="$(mktemp -d "${destination}.partial.XXXXXX")"
cleanup() {
    rm -rf -- "${temporary}"
}
trap cleanup EXIT

echo "Staging model files from ${source_dir} to node-local storage..."
cp -aL "${source_dir}/." "${temporary}/"
printf '%s\n' "${source_dir}" > "${temporary}/.goqa-staged-source"
mv "${temporary}" "${destination}"
trap - EXIT
echo "Node-local GOQA model ready: ${destination}"
