#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
bhaskera_root="${slakshna_root}/Bhaskera"

# v0.1.1-alpha is the stock Slakshna release selected for Phase 9.
release_tag="${PHASE9_SLAKSHNA_TAG:-v0.1.1-alpha}"
release_revision="${PHASE9_SLAKSHNA_REVISION:-9f93ec45ae0d3eb9c901aff3b50d4325b5050488}"
phase9_venv="${PHASE9_UV_ENVIRONMENT:-${bhaskera_root}/.venv-phase9}"
python_version="${PHASE9_PYTHON_VERSION:-3.11.13}"
torch_version="${PHASE9_TORCH_VERSION:-2.9.0}"
torchvision_version="${PHASE9_TORCHVISION_VERSION:-0.24.0}"
torchaudio_version="${PHASE9_TORCHAUDIO_VERSION:-2.9.0}"
torch_index="${PHASE9_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
flash_attn_spec="${PHASE9_FLASH_ATTN_SPEC:-flash-attn>=2.7,<3}"
flash_attn_wheel_name="flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl"
flash_attn_wheel_url="${PHASE9_FLASH_ATTN_WHEEL_URL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl}"
flash_attn_wheel_sha256="${PHASE9_FLASH_ATTN_WHEEL_SHA256:-d4c842693349ef3e8b3b1396056003ecb9d21e94493dd1ea9ad6a2b3c6ce5c73}"
cuda_arch_list="${PHASE9_CUDA_ARCH_LIST:-8.0}"
max_jobs="${MAX_JOBS:-4}"

runtime_root="${experiment_root}/.runtime"
manifest_dir="${runtime_root}/manifests/phase9"
log_dir="${runtime_root}/logs/phase9"
uv_cache_dir="${runtime_root}/cache/uv-phase9"
hf_cache_dir="${runtime_root}/cache/huggingface"
xdg_cache_dir="${runtime_root}/cache/xdg-phase9"
download_dir="${runtime_root}/downloads/phase9"
activation_script="${bhaskera_root}/phase9-activate.sh"

usage() {
    cat <<'EOF'
Usage: bash monash_exps/scripts/phase9/setup_env.sh [options]

Create the dedicated Phase 9 uv environment for the stock Slakshna release.

Options:
  --skip-release-checkout  Require the selected release to be checked out already
  --skip-flash-attn        Skip FlashAttention installation (diagnostic fallback only)
  --source-flash-attn      Compile FlashAttention instead of using the pinned wheel
  --reinstall-flash-attn   Reinstall FlashAttention even when it is already importable
  -h, --help               Show this help

Environment overrides:
  PHASE9_UV_ENVIRONMENT      Virtual environment location
  PHASE9_PYTHON_VERSION      uv-managed Python version (default: 3.11.13)
  PHASE9_TORCH_VERSION       PyTorch version (default: 2.9.0)
  PHASE9_TORCH_INDEX         PyTorch wheel index (default: cu128)
  PHASE9_FLASH_ATTN_SPEC     FlashAttention requirement (default: >=2.7,<3)
  PHASE9_FLASH_ATTN_WHEEL_URL       Override the pinned prebuilt wheel URL
  PHASE9_FLASH_ATTN_WHEEL_SHA256    Override the pinned wheel digest
  PHASE9_CUDA_ARCH_LIST      CUDA architectures to compile (default: 8.0 for A100)
  MAX_JOBS                   Maximum parallel FlashAttention build jobs (default: 4)
EOF
}

checkout_release=1
install_flash_attn=1
source_flash_attn=0
reinstall_flash_attn=0
while (( $# > 0 )); do
    case "$1" in
        --skip-release-checkout)
            checkout_release=0
            shift
            ;;
        --skip-flash-attn)
            install_flash_attn=0
            shift
            ;;
        --source-flash-attn)
            source_flash_attn=1
            shift
            ;;
        --reinstall-flash-attn)
            reinstall_flash_attn=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p "${manifest_dir}" "${log_dir}" "${uv_cache_dir}" \
    "${hf_cache_dir}" "${xdg_cache_dir}" "${download_dir}"

timestamp="$(date +%Y%m%dT%H%M%S)"
setup_log="${log_dir}/setup-env-${timestamp}.log"
exec > >(tee -a "${setup_log}") 2>&1

die() {
    echo "ERROR: $*" >&2
    exit 1
}

section() {
    echo
    echo "=== $* ==="
}

[[ -d "${slakshna_root}/.git" || -f "${slakshna_root}/.git" ]] || \
    die "Missing Slakshna checkout at ${slakshna_root}"
command -v git >/dev/null 2>&1 || die "git is required"

section "Select the stock Slakshna ${release_tag} release"
current_revision="$(git -C "${slakshna_root}" rev-parse HEAD)"
if [[ "${current_revision}" != "${release_revision}" ]]; then
    if (( ! checkout_release )); then
        die "Slakshna is ${current_revision}; expected ${release_revision}"
    fi

    tracked_changes="$(git -C "${slakshna_root}" status --porcelain --untracked-files=no)"
    if [[ -n "${tracked_changes}" ]]; then
        echo "Tracked changes in Slakshna must be reviewed before changing releases:" >&2
        echo "${tracked_changes}" >&2
        exit 1
    fi

    git -C "${slakshna_root}" fetch --force origin \
        "refs/tags/${release_tag}:refs/tags/${release_tag}"
    fetched_revision="$(git -C "${slakshna_root}" rev-list -n 1 "refs/tags/${release_tag}")"
    [[ "${fetched_revision}" == "${release_revision}" ]] || \
        die "${release_tag} resolved to ${fetched_revision}, expected ${release_revision}"
    git -C "${slakshna_root}" checkout --detach "${release_revision}"
fi

current_revision="$(git -C "${slakshna_root}" rev-parse HEAD)"
[[ "${current_revision}" == "${release_revision}" ]] || \
    die "Failed to select Slakshna ${release_revision}"
[[ -f "${bhaskera_root}/pyproject.toml" ]] || \
    die "The release does not contain Bhaskera/pyproject.toml"
echo "Slakshna revision: ${current_revision}"

section "Load the M3 compiler and CUDA toolchain"
export SLAKSHNA_CLUSTER="${SLAKSHNA_CLUSTER:-m3}"
export SLAKSHNA_M3_COMPILER_MODULE="${SLAKSHNA_M3_COMPILER_MODULE:-gcc/10.2.0}"
export SLAKSHNA_M3_CUDA_MODULE="${SLAKSHNA_M3_CUDA_MODULE:-cuda/12.8}"
export SLAKSHNA_UV_ENVIRONMENT="${phase9_venv}"

# Reuse the repository's cluster adapter for caches and module loading, while
# pointing it at the dedicated Phase 9 environment rather than Phase 8's venv.
# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

if command -v nvcc >/dev/null 2>&1; then
    nvcc_path="$(command -v nvcc)"
    export CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "${nvcc_path}")/.." && pwd)}"
else
    die "nvcc is unavailable after loading ${SLAKSHNA_M3_CUDA_MODULE}"
fi
echo "CUDA_HOME: ${CUDA_HOME}"
nvcc --version | tail -n 4

section "Locate the project-local uv"
uv_bin="${SLAKSHNA_UV_BIN:-${runtime_root}/tools/uv/bin/uv}"
if [[ ! -x "${uv_bin}" ]]; then
    bash "${experiment_root}/scripts/environment/01_install_uv.sh"
fi
[[ -x "${uv_bin}" ]] || die "uv installation did not create ${uv_bin}"
"${uv_bin}" --version

export UV_CACHE_DIR="${uv_cache_dir}"
export UV_PYTHON_INSTALL_DIR="${SLAKSHNA_UV_PYTHON_DIR:-${runtime_root}/python}"
export UV_PYTHON_BIN_DIR="${SLAKSHNA_UV_PYTHON_BIN_DIR:-${runtime_root}/tools/python-bin}"
export HF_HOME="${HF_HOME:-${hf_cache_dir}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${hf_cache_dir}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${xdg_cache_dir}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" \
    "${UV_PYTHON_BIN_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" \
    "${XDG_CACHE_HOME}"

section "Create the dedicated Python ${python_version} uv environment"
if [[ ! -x "${phase9_venv}/bin/python" ]]; then
    "${uv_bin}" python install "${python_version}"
    "${uv_bin}" venv "${phase9_venv}" --python "${python_version}"
fi
phase9_python="${phase9_venv}/bin/python"
actual_python="$(${phase9_python} -c 'import platform; print(platform.python_version())')"
[[ "${actual_python}" == 3.11.* ]] || \
    die "Existing Phase 9 environment uses Python ${actual_python}; expected Python 3.11"
echo "Python: ${actual_python}"
echo "Environment: ${phase9_venv}"

section "Install the CUDA 12.8 PyTorch stack first"
"${uv_bin}" pip install --python "${phase9_python}" \
    --index-url "${torch_index}" \
    "torch==${torch_version}" \
    "torchvision==${torchvision_version}" \
    "torchaudio==${torchaudio_version}"

section "Install the release-native Bhaskera dependency set"
"${uv_bin}" pip install --python "${phase9_python}" \
    setuptools packaging ninja wheel
"${uv_bin}" pip install --python "${phase9_python}" \
    --editable "${bhaskera_root}[wandb,mlflow,inference,dev]"

# These packages are requested by the top-level Slakshna setup or documented
# in Bhaskera's complete requirements snapshot but are not all declared in the
# Bhaskera core metadata.
"${uv_bin}" pip install --python "${phase9_python}" \
    "bitsandbytes>=0.43" \
    "einops>=0.8" \
    "opacus>=1.4" \
    "pyarrow>=15" \
    "scipy>=1.10" \
    "opt-einsum>=3.3"

flash_attn_install_mode="skipped"
if (( install_flash_attn && source_flash_attn )); then
    section "Build and install FlashAttention for A100"
    flash_attn_install_mode="source"
    export MAX_JOBS="${max_jobs}"
    export NVCC_THREADS="${NVCC_THREADS:-2}"
    export TORCH_CUDA_ARCH_LIST="${cuda_arch_list}"
    flash_args=(
        pip install
        --python "${phase9_python}"
        --no-build-isolation
    )
    if (( reinstall_flash_attn )); then
        flash_args+=(--reinstall)
    fi
    flash_args+=("${flash_attn_spec}")
    "${uv_bin}" "${flash_args[@]}"
elif (( install_flash_attn )); then
    section "Install the pinned Phase 9 FlashAttention wheel"
    flash_attn_install_mode="prebuilt"

    compatibility="$(${phase9_python} - <<'PY'
import platform
import sys
import torch

print("compatible=" + str(
    sys.version_info[:2] == (3, 11)
    and platform.system() == "Linux"
    and platform.machine() == "x86_64"
    and torch.__version__ == "2.9.0+cu128"
    and torch.version.cuda == "12.8"
).lower())
print(f"python={sys.version_info.major}.{sys.version_info.minor}")
print(f"platform={platform.system()}-{platform.machine()}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
PY
)"
    echo "${compatibility}"
    if ! grep -qx 'compatible=true' <<< "${compatibility}"; then
        die "The pinned FlashAttention wheel is incompatible. Use --source-flash-attn for this environment."
    fi

    wheel_path="${download_dir}/${flash_attn_wheel_name}"
    if [[ -f "${wheel_path}" ]]; then
        cached_sha256="$(sha256sum "${wheel_path}" | awk '{print $1}')"
        if [[ "${cached_sha256}" != "${flash_attn_wheel_sha256}" ]]; then
            rejected_path="${wheel_path}.rejected-${timestamp}"
            mv "${wheel_path}" "${rejected_path}"
            echo "Rejected cached wheel with SHA256 ${cached_sha256}: ${rejected_path}"
        fi
    fi

    if [[ ! -f "${wheel_path}" ]]; then
        wheel_tmp="${wheel_path}.partial-${timestamp}"
        curl --fail --location --retry 3 --output "${wheel_tmp}" \
            "${flash_attn_wheel_url}"
        downloaded_sha256="$(sha256sum "${wheel_tmp}" | awk '{print $1}')"
        if [[ "${downloaded_sha256}" != "${flash_attn_wheel_sha256}" ]]; then
            die "FlashAttention wheel SHA256 mismatch: got ${downloaded_sha256}, expected ${flash_attn_wheel_sha256}; partial file: ${wheel_tmp}"
        fi
        mv "${wheel_tmp}" "${wheel_path}"
    fi

    verified_sha256="$(sha256sum "${wheel_path}" | awk '{print $1}')"
    [[ "${verified_sha256}" == "${flash_attn_wheel_sha256}" ]] || \
        die "Cached FlashAttention wheel failed its pinned SHA256 check"

    wheel_args=(pip install --python "${phase9_python}" --no-deps)
    if (( reinstall_flash_attn )); then
        wheel_args+=(--reinstall)
    fi
    wheel_args+=("${wheel_path}")
    "${uv_bin}" "${wheel_args[@]}"
else
    echo "WARNING: FlashAttention was skipped by explicit request."
fi

section "Validate the installed environment"
"${uv_bin}" pip check --python "${phase9_python}"
PHASE9_EXPECT_FLASH="${install_flash_attn}" \
PHASE9_SLAKSHNA_ROOT="${slakshna_root}" \
"${phase9_python}" - <<'PY'
import importlib
import os
import pathlib
import sys

import torch
import transformers
from transformers import OlmoConfig, OlmoForCausalLM

required = [
    "accelerate",
    "bitsandbytes",
    "datasets",
    "liger_kernel",
    "mlflow",
    "opacus",
    "peft",
    "pyarrow",
    "ray",
    "safetensors",
    "wandb",
]
modules = {name: importlib.import_module(name) for name in required}
if os.environ["PHASE9_EXPECT_FLASH"] == "1":
    modules["flash_attn"] = importlib.import_module("flash_attn")

bhaskera = importlib.import_module("bhaskera")
expected_root = (
    pathlib.Path(os.environ["PHASE9_SLAKSHNA_ROOT"]) / "Bhaskera" / "src"
).resolve()
expected_package = expected_root / "bhaskera"
bhaskera_paths = [pathlib.Path(path).resolve() for path in bhaskera.__path__]
if expected_package not in bhaskera_paths:
    raise RuntimeError(
        f"bhaskera namespace paths are {bhaskera_paths}, "
        f"expected editable source at {expected_package}"
    )

# This proves that the selected Transformers build contains native OLMo
# support without downloading the model during the environment step.
tiny_olmo = OlmoForCausalLM(
    OlmoConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
)
assert hasattr(tiny_olmo.model.layers[0].self_attn, "q_proj")
assert hasattr(tiny_olmo.model.layers[0].self_attn, "v_proj")

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print("bhaskera=" + ",".join(str(path) for path in bhaskera_paths))
for name, module in sorted(modules.items()):
    print(f"{name}={getattr(module, '__version__', 'installed')}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(x @ x, torch.mm(x, x))
    print(f"gpu={torch.cuda.get_device_name(0)}")

    if os.environ["PHASE9_EXPECT_FLASH"] == "1":
        from flash_attn import flash_attn_func

        q = torch.randn(1, 32, 4, 64, device=device, dtype=torch.float16)
        y = flash_attn_func(q, q, q, causal=True)
        assert y.shape == q.shape
        print("flash_attn_cuda_forward=passed")
else:
    print("gpu=not visible; CUDA kernel validation deferred to the GPU smoke test")
PY

section "Write the Phase 9 activation helper and environment manifest"
cat > "${activation_script}" <<EOF
#!/usr/bin/env bash
# Generated by monash_exps/scripts/phase9/setup_env.sh.
export SLAKSHNA_CLUSTER="m3"
export SLAKSHNA_M3_COMPILER_MODULE="${SLAKSHNA_M3_COMPILER_MODULE}"
export SLAKSHNA_M3_CUDA_MODULE="${SLAKSHNA_M3_CUDA_MODULE}"
export SLAKSHNA_UV_ENVIRONMENT="${phase9_venv}"
export UV_CACHE_DIR="${uv_cache_dir}"
export HF_HOME="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME}"
source "${experiment_root}/scripts/cluster/activate.sh"
EOF
chmod +x "${activation_script}"

freeze_file="${manifest_dir}/environment-freeze.txt"
manifest_file="${manifest_dir}/setup-env.txt"
"${uv_bin}" pip freeze --python "${phase9_python}" > "${freeze_file}"
{
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "slakshna_tag=${release_tag}"
    echo "slakshna_revision=${current_revision}"
    echo "python=${actual_python}"
    echo "environment=${phase9_venv}"
    echo "torch_requested=${torch_version}"
    echo "torch_index=${torch_index}"
    echo "flash_attn_requested=$([[ ${install_flash_attn} -eq 1 ]] && echo yes || echo no)"
    echo "flash_attn_install_mode=${flash_attn_install_mode}"
    if [[ "${flash_attn_install_mode}" == "prebuilt" ]]; then
        echo "flash_attn_wheel_url=${flash_attn_wheel_url}"
        echo "flash_attn_wheel_sha256=${flash_attn_wheel_sha256}"
    fi
    echo "cuda_arch_list=${cuda_arch_list}"
    echo "activation_script=${activation_script}"
    echo "freeze=${freeze_file}"
    echo "setup_log=${setup_log}"
} > "${manifest_file}"

echo
echo "PHASE 9 ENVIRONMENT SETUP PASSED"
echo "Slakshna release : ${release_tag} (${current_revision})"
echo "Environment      : ${phase9_venv}"
echo "Activation       : ${activation_script}"
echo "Manifest         : ${manifest_file}"
echo "Freeze           : ${freeze_file}"
echo "Log              : ${setup_log}"
