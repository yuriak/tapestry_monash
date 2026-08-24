#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
bhaskera_root="${slakshna_root}/Bhaskera"
runtime_root="${experiment_root}/.runtime"
environment_root="${runtime_root}/venvs/primary"
uv_bin="${runtime_root}/tools/uv/bin/uv"
python_bin="${environment_root}/bin/python"
download_root="${runtime_root}/downloads/m0_fl"
manifest_root="${runtime_root}/manifests/m0_fl"
activation_root="${runtime_root}/activation"

expected_slakshna_revision="${M0_FL_SLAKSHNA_REVISION:-b1317dc97093a31476976d55ff76f29f2a04d4b3}"
expected_bhaskera_revision="${M0_FL_BHASKERA_REVISION:-d737ced2ca2bf9c27521bc515d4c98b27551f6be}"
flash_wheel_name="flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl"
flash_wheel_url="${M0_FL_FLASH_ATTN_WHEEL_URL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl}"
flash_wheel_sha256="${M0_FL_FLASH_ATTN_WHEEL_SHA256:-d4c842693349ef3e8b3b1396056003ecb9d21e94493dd1ea9ad6a2b3c6ce5c73}"
flash_wheel_path="${download_root}/${flash_wheel_name}"

skip_rust=0
skip_gpu_check=0

usage() {
    cat <<'EOF'
Usage: 01_upgrade_environment.sh [--skip-rust] [--skip-gpu-check]

Upgrade the existing primary uv environment in place for the local FL run.
The script never creates a second virtual environment.

Options:
  --skip-rust       Do not rebuild the Slakshna release binary.
  --skip-gpu-check  Validate imports only; skip the CUDA/FlashAttention kernel test.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --skip-rust)
            skip_rust=1
            shift
            ;;
        --skip-gpu-check)
            skip_gpu_check=1
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

section() {
    printf '\n=== %s ===\n' "$1"
}

require_clean_repo() {
    local repo="$1"
    local label="$2"
    local status
    status="$(git -C "${repo}" status --porcelain)"
    if [[ -n "${status}" ]]; then
        echo "${label} must be clean before the environment is upgraded:" >&2
        echo "${status}" >&2
        exit 1
    fi
}

cd "${workspace_root}"

section "Preflight"
[[ -x "${uv_bin}" ]] || {
    echo "Project-local uv is missing: ${uv_bin}" >&2
    exit 1
}
[[ -x "${python_bin}" ]] || {
    echo "The existing primary environment is missing: ${environment_root}" >&2
    echo "This workflow intentionally does not create a replacement environment." >&2
    exit 1
}
[[ -f "${bhaskera_root}/pyproject.toml" ]] || {
    echo "Bhaskera submodule is incomplete: ${bhaskera_root}" >&2
    exit 1
}

actual_slakshna_revision="$(git -C "${slakshna_root}" rev-parse HEAD)"
actual_bhaskera_revision="$(git -C "${bhaskera_root}" rev-parse HEAD)"
[[ "${actual_slakshna_revision}" == "${expected_slakshna_revision}" ]] || {
    echo "Unexpected Slakshna revision: ${actual_slakshna_revision}" >&2
    echo "Expected: ${expected_slakshna_revision}" >&2
    exit 1
}
[[ "${actual_bhaskera_revision}" == "${expected_bhaskera_revision}" ]] || {
    echo "Unexpected Bhaskera revision: ${actual_bhaskera_revision}" >&2
    echo "Expected: ${expected_bhaskera_revision}" >&2
    exit 1
}
require_clean_repo "${slakshna_root}" "Slakshna"
require_clean_repo "${bhaskera_root}" "Bhaskera"

# Load the cluster compiler/CUDA adapter without allowing it to select another
# Python environment. The uv environment path is always the existing primary.
export SLAKSHNA_UV_ENVIRONMENT="${environment_root}"
# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"
export UV_CACHE_DIR="${runtime_root}/cache/uv-m0-fl"
mkdir -p "${UV_CACHE_DIR}" "${download_root}" "${manifest_root}" "${activation_root}"

echo "Cluster       : ${SLAKSHNA_CLUSTER}"
echo "Environment   : ${environment_root}"
echo "Python        : $("${python_bin}" --version)"
echo "Slakshna      : ${actual_slakshna_revision}"
echo "Bhaskera      : ${actual_bhaskera_revision}"

section "Install the current Bhaskera source and training dependencies"
"${uv_bin}" pip install --python "${python_bin}" \
    "setuptools>=69" "packaging>=23" "wheel>=0.43" "ninja>=1.11"
"${uv_bin}" pip install --python "${python_bin}" --editable \
    "${bhaskera_root}[wandb,mlflow,inference,dev]"
"${uv_bin}" pip install --python "${python_bin}" \
    "bitsandbytes>=0.43" \
    "einops>=0.8" \
    "gdown>=5" \
    "opacus>=1.4" \
    "opt-einsum>=3.3" \
    "scipy>=1.10" \
    "toml>=0.10" \
    "setproctitle>=1.3"

section "Install the pinned FlashAttention wheel"
compatibility="$("${python_bin}" - <<'PY'
import platform
import sys
import torch

print(f"python={sys.version_info.major}.{sys.version_info.minor}")
print(f"machine={platform.machine()}")
print(f"torch={torch.__version__}")
assert sys.version_info[:2] == (3, 11)
assert platform.system() == "Linux" and platform.machine() == "x86_64"
assert torch.__version__.startswith("2.9.0+cu128")
PY
)"
echo "${compatibility}"

if [[ ! -f "${flash_wheel_path}" ]] || \
   [[ "$(sha256sum "${flash_wheel_path}" | awk '{print $1}')" != "${flash_wheel_sha256}" ]]; then
    rm -f -- "${flash_wheel_path}"
    curl --fail --location --retry 3 --output "${flash_wheel_path}" "${flash_wheel_url}"
fi
echo "${flash_wheel_sha256}  ${flash_wheel_path}" | sha256sum --check
"${uv_bin}" pip install --python "${python_bin}" --no-deps --reinstall "${flash_wheel_path}"

section "Validate the upgraded Python environment"
"${uv_bin}" pip check --python "${python_bin}"
EXPECTED_BHASKERA_ROOT="${bhaskera_root}" \
M0_FL_SKIP_GPU_CHECK="${skip_gpu_check}" \
"${python_bin}" - <<'PY'
from pathlib import Path
import os

import bhaskera
import bhaskera.launcher.train
import datasets
import flash_attn
import liger_kernel
import peft
import ray
import torch
import transformers
from flash_attn import flash_attn_func
from transformers import Olmo2Config, Olmo2ForCausalLM

expected = Path(os.environ["EXPECTED_BHASKERA_ROOT"]).resolve()
# Bhaskera intentionally uses a PEP 420 namespace package, so its top-level
# module has no __file__. Verify a concrete module and every namespace path.
actual = Path(bhaskera.launcher.train.__file__).resolve()
namespace_paths = [Path(path).resolve() for path in bhaskera.__path__]
if expected not in actual.parents or not any(expected in path.parents for path in namespace_paths):
    raise RuntimeError(
        f"Bhaskera resolves to module={actual}, namespace={namespace_paths}, "
        f"not the current source {expected}"
    )

config = Olmo2Config(
    vocab_size=128,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
)
model = Olmo2ForCausalLM(config)
output = model(input_ids=torch.randint(0, 128, (1, 16)))
assert output.logits.shape == (1, 16, 128)

print(f"bhaskera_module={actual}")
print(f"bhaskera_namespace={namespace_paths}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"ray={ray.__version__}")
print(f"datasets={datasets.__version__}")
print(f"peft={peft.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"liger_kernel={Path(liger_kernel.__file__).resolve()}")

if os.environ["M0_FL_SKIP_GPU_CHECK"] == "1":
    print("CUDA/FlashAttention kernel check skipped by request.")
else:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; rerun on an A100 or use --skip-gpu-check")
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 0):
        raise RuntimeError(f"Expected an A100 (compute capability 8.0), found {capability}")
    q = torch.randn(2, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    result = flash_attn_func(q, q, q, causal=True)
    assert result.shape == q.shape and torch.isfinite(result).all()
    print(f"gpu={torch.cuda.get_device_name(0)}; flash_forward=PASS")
PY

# Editable installs may create this generated, ignored metadata directory.
rm -rf -- "${bhaskera_root}/src/bhaskera.egg-info"
require_clean_repo "${slakshna_root}" "Slakshna"
require_clean_repo "${bhaskera_root}" "Bhaskera"

if (( ! skip_rust )); then
    section "Build the current Slakshna Rust binary"
    SLAKSHNA_UV_ENVIRONMENT="${environment_root}" \
        bash "${experiment_root}/scripts/environment/07_build_slakshna.sh"
fi

section "Record the environment"
"${uv_bin}" pip freeze --python "${python_bin}" > "${manifest_root}/python-freeze.txt"
{
    echo "created_at=$(date --iso-8601=seconds)"
    echo "cluster=${SLAKSHNA_CLUSTER}"
    echo "environment=${environment_root}"
    echo "python=$("${python_bin}" --version 2>&1)"
    echo "slakshna_revision=${actual_slakshna_revision}"
    echo "bhaskera_revision=${actual_bhaskera_revision}"
    echo "flash_attn_wheel=${flash_wheel_name}"
    echo "flash_attn_sha256=${flash_wheel_sha256}"
    if (( ! skip_rust )); then
        echo "rust_binary=${runtime_root}/cargo-target/slakshna/release/iiitd"
        echo "rust_binary_sha256=$(sha256sum "${runtime_root}/cargo-target/slakshna/release/iiitd" | awk '{print $1}')"
    else
        echo "rust_binary=not_built"
    fi
} > "${manifest_root}/environment.txt"

cat > "${activation_root}/m0_fl.sh" <<EOF
#!/usr/bin/env bash
export SLAKSHNA_UV_ENVIRONMENT="${environment_root}"
export PATH="${environment_root}/bin:\${PATH}"
EOF
chmod 755 "${activation_root}/m0_fl.sh"

echo "Environment upgrade complete."
echo "Manifest: ${manifest_root}/environment.txt"
echo "Activate : source ${activation_root}/m0_fl.sh"
