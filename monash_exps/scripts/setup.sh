#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
cd "${workspace_root}"

cluster="${SLAKSHNA_CLUSTER:-}"
install_playit=1
run_preflight=1

usage() {
    cat <<'EOF'
Usage: bash monash_exps/scripts/setup.sh [options]

Prepare one clone for Slakshna experiments using the committed environment lock.

Options:
  --cluster NAME       Select m3, spartan, fit, or generic (default: auto-detection)
  --skip-playit        Do not install the pinned unprivileged playit binaries
  --skip-preflight     Do not run the CPU/API environment preflight
  -h, --help           Show this help
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --cluster)
            [[ $# -ge 2 ]] || { echo "--cluster requires a value" >&2; exit 2; }
            cluster="$2"
            shift 2
            ;;
        --skip-playit)
            install_playit=0
            shift
            ;;
        --skip-preflight)
            run_preflight=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown setup option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "${cluster}" && ! "${cluster}" =~ ^(m3|spartan|fit|generic)$ ]]; then
    echo "Unsupported cluster '${cluster}'; expected m3, spartan, fit, or generic." >&2
    exit 2
fi
if [[ -n "${cluster}" ]]; then
    export SLAKSHNA_CLUSTER="${cluster}"
fi

command -v git >/dev/null 2>&1 || {
    echo "git is required before setup can initialize the Slakshna submodule." >&2
    exit 1
}
[[ -f .gitmodules && -d monash_exps ]] || {
    echo "setup.sh must run from a complete tapestry_monash clone." >&2
    exit 1
}

echo "=== Initialize the pinned Slakshna submodule ==="
git submodule update --init --recursive -- Slakshna
expected_revision="$(git ls-files --stage -- Slakshna | awk '$1 == 160000 {print $2}')"
current_revision="$(git -C Slakshna rev-parse HEAD)"
[[ -n "${expected_revision}" && "${current_revision}" == "${expected_revision}" ]] || {
    echo "Slakshna submodule mismatch: parent=${expected_revision:-missing}, checkout=${current_revision}" >&2
    exit 1
}
if [[ -n "$(git -C Slakshna status --porcelain)" ]]; then
    echo "Slakshna submodule must be clean before setup." >&2
    git -C Slakshna status --short >&2
    exit 1
fi

echo
echo "=== Activate the cluster adapter ==="
# shellcheck source=cluster/activate.sh
source "${script_dir}/cluster/activate.sh"

echo
echo "=== Install project-local uv when absent ==="
if [[ ! -x "${SLAKSHNA_UV_BIN}" ]]; then
    bash "${script_dir}/environment/01_install_uv.sh"
fi

echo
echo "=== Synchronize the primary environment from the frozen lock ==="
bash "${script_dir}/environment/03_sync_environment.sh" primary
# The first activation occurred before the venv existed or was upgraded.
source "${script_dir}/cluster/activate.sh"

echo
echo "=== Install the project-local Rust toolchain when absent ==="
if [[ ! -x "${experiment_root}/.runtime/cargo/bin/cargo" ]]; then
    bash "${script_dir}/environment/06_install_rust.sh"
fi

echo
echo "=== Build the pinned Slakshna release binary ==="
bash "${script_dir}/environment/07_build_slakshna.sh"

if (( install_playit )); then
    echo
    echo "=== Install checksum-pinned playit binaries ==="
    bash "${script_dir}/environment/04_install_playit.sh"
fi

if (( run_preflight )); then
    echo
    echo "=== Run the CPU/API deployment preflight ==="
    bash "${script_dir}/environment/05_phase0_preflight.sh" cpu
fi

setup_manifest="${experiment_root}/.runtime/manifests/primary/setup.txt"
mkdir -p "$(dirname "${setup_manifest}")"
{
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "cluster=${SLAKSHNA_CLUSTER}"
    echo "workspace=${workspace_root}"
    echo "slakshna_revision=${current_revision}"
    echo "uv_lock_sha256=$(sha256sum "${experiment_root}/environment/uv.lock" | awk '{print $1}')"
    echo "environment=${SLAKSHNA_UV_ENVIRONMENT}"
    echo "rust_binary=${experiment_root}/.runtime/cargo-target/slakshna/release/iiitd"
    if (( install_playit )); then
        echo "playit_cli=${SLAKSHNA_PLAYIT_BIN}"
        echo "playit_daemon=${SLAKSHNA_PLAYIT_DAEMON}"
    fi
} > "${setup_manifest}"

echo
echo "SETUP PASSED"
echo "Cluster            : ${SLAKSHNA_CLUSTER}"
echo "Slakshna revision  : ${current_revision}"
echo "Primary environment: ${SLAKSHNA_UV_ENVIRONMENT}"
echo "Manifest           : ${setup_manifest}"
if (( install_playit )); then
    echo "playit is installed but remains unclaimed; account authorization is a separate manual step."
fi
