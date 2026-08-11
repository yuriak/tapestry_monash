#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

# Version and digests are pinned from the upstream GitHub v1.0.10 release.
playit_version="1.0.10"
case "$(uname -m)" in
    x86_64|amd64)
        asset_arch="amd64"
        daemon_sha256="2df7d9f10227ab312b1ad341853db4e8a8243df5cfcdbae58713a4271711c339"
        cli_sha256="6fd54d147ae1d3232b22c1c1f4aa3d13cf16d889e840ca2d3f90b4f50a2e7301"
        ;;
    aarch64|arm64)
        asset_arch="aarch64"
        daemon_sha256="4c0db3e7b3a8158e249441c2f0b73f54e83429395890c7b1ca45fd7a6303d763"
        cli_sha256="b126b4164c03838598c8f33f209d76f6acf1c257d07900c0af2d461b9647099f"
        ;;
    *)
        echo "Unsupported playit architecture: $(uname -m)" >&2
        exit 2
        ;;
esac

command -v curl >/dev/null 2>&1 || {
    echo "curl is required to install playit." >&2
    exit 1
}
command -v sha256sum >/dev/null 2>&1 || {
    echo "sha256sum is required to verify playit." >&2
    exit 1
}

download_root="${SLAKSHNA_RUNTIME_ROOT}/downloads/playit/${playit_version}"
install_root="${SLAKSHNA_PLAYIT_ROOT}"
mkdir -p "${download_root}" "${install_root}/bin"

install_asset() {
    local asset_name="$1"
    local expected_sha256="$2"
    local destination="$3"
    local download="${download_root}/${asset_name}"
    local url="https://github.com/playit-cloud/playit-agent/releases/download/v${playit_version}/${asset_name}"
    local temporary=""

    if [[ ! -f "${download}" ]] || \
       ! printf '%s  %s\n' "${expected_sha256}" "${download}" | sha256sum --check --status; then
        temporary="$(mktemp "${download_root}/.${asset_name}.XXXXXX")"
        trap 'rm -f -- "${temporary:-}"' RETURN
        echo "Downloading ${asset_name} from the official playit release..."
        curl --proto '=https' --tlsv1.2 --fail --location \
            --retry 3 --output "${temporary}" "${url}"
        printf '%s  %s\n' "${expected_sha256}" "${temporary}" | sha256sum --check --status || {
            echo "Checksum verification failed for ${asset_name}" >&2
            exit 1
        }
        mv "${temporary}" "${download}"
        trap - RETURN
    fi

    install -m 0700 "${download}" "${destination}"
    printf '%s  %s\n' "${expected_sha256}" "${destination}" | sha256sum --check --status
}

install_asset "playit-linux-${asset_arch}" "${daemon_sha256}" "${SLAKSHNA_PLAYIT_DAEMON}"
install_asset "playit-cli-linux-${asset_arch}" "${cli_sha256}" "${SLAKSHNA_PLAYIT_BIN}"

manifest="${install_root}/install-manifest.txt"
{
    echo "version=${playit_version}"
    echo "architecture=${asset_arch}"
    echo "daemon=${SLAKSHNA_PLAYIT_DAEMON}"
    echo "daemon_sha256=${daemon_sha256}"
    echo "cli=${SLAKSHNA_PLAYIT_BIN}"
    echo "cli_sha256=${cli_sha256}"
} > "${manifest}"

echo "playit ${playit_version} installed without root privileges."
echo "  daemon: ${SLAKSHNA_PLAYIT_DAEMON}"
echo "  CLI   : ${SLAKSHNA_PLAYIT_BIN}"
echo "Account claiming and tunnel credentials are intentionally not managed here."
