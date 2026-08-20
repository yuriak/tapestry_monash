#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
bhaskera_root="${slakshna_root}/Bhaskera"
runtime_root="${experiment_root}/.runtime"

expected_revision="${M0_SLAKSHNA_REVISION:-036e8fbe570bd2dabfcf4b65d44569b2fba11876}"
venv="${M0_UV_ENVIRONMENT:-${bhaskera_root}/.venv-phase9}"
uv_bin="${SLAKSHNA_UV_BIN:-${runtime_root}/tools/uv/bin/uv}"
manifest_root="${runtime_root}/manifests/m0"
log_root="${runtime_root}/logs/m0"
activation_script="${runtime_root}/activation/m0.sh"
cargo_target="${M0_CARGO_TARGET_DIR:-${runtime_root}/cargo-target/m0}"

skip_rust=0
skip_gpu_check=0
install_vllm=0

usage() {
    cat <<'EOF'
Usage: bash monash_exps/scripts/m0/01_upgrade_stack.sh [options]

Upgrade the existing Phase 9 uv environment in place for M0 and build the
latest Slakshna Rust binary. The environment remains at
Slakshna/Bhaskera/.venv-phase9.

Options:
  --skip-rust       Do not build the Rust binary
  --skip-gpu-check  Permit running without a visible GPU
  --install-vllm    Install vLLM now (deferred by default to avoid changing torch)
  -h, --help        Show this help

Required/recommended environment:
  SLAKSHNA_CLUSTER=m3|spartan
  M0_SLAKSHNA_REVISION=<expected commit>
  M0_LIBCLANG_PATH=<directory containing libclang>
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --skip-rust) skip_rust=1 ;;
        --skip-gpu-check) skip_gpu_check=1 ;;
        --install-vllm) install_vllm=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

die() { echo "ERROR: $*" >&2; exit 1; }
section() { echo; echo "=== $* ==="; }

[[ -d "${slakshna_root}" ]] || die "Missing Slakshna checkout: ${slakshna_root}"
[[ -f "${bhaskera_root}/pyproject.toml" ]] || die "Missing Bhaskera source"
current_revision="$(git -C "${slakshna_root}" rev-parse HEAD)"
[[ "${current_revision}" == "${expected_revision}" ]] || \
    die "Slakshna is ${current_revision}; expected ${expected_revision}"
[[ -z "$(git -C "${slakshna_root}" status --porcelain --untracked-files=no)" ]] || \
    die "Slakshna has tracked modifications; review them before environment setup"

mkdir -p "${manifest_root}" "${log_root}" "${cargo_target}" \
    "$(dirname "${activation_script}")"
stamp="$(date +%Y%m%dT%H%M%S)"
log_path="${log_root}/upgrade-stack-${stamp}.log"
exec > >(tee -a "${log_path}") 2>&1

section "Activate cluster adapter without changing the environment path"
export SLAKSHNA_UV_ENVIRONMENT="${venv}"
if [[ -z "${SLAKSHNA_CLUSTER:-}" ]]; then
    case "${workspace_root}" in
        /fs*) export SLAKSHNA_CLUSTER=m3 ;;
        /home/*) export SLAKSHNA_CLUSTER=spartan ;;
        *) export SLAKSHNA_CLUSTER=generic ;;
    esac
fi
if [[ "${SLAKSHNA_CLUSTER}" == "m3" ]]; then
    export SLAKSHNA_M3_COMPILER_MODULE="${SLAKSHNA_M3_COMPILER_MODULE:-gcc/10.2.0}"
    export SLAKSHNA_M3_CUDA_MODULE="${SLAKSHNA_M3_CUDA_MODULE:-cuda/12.8}"
fi
# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"
echo "Slakshna revision : ${current_revision}"
echo "Cluster           : ${SLAKSHNA_CLUSTER}"
echo "Environment       : ${venv}"

section "Locate uv and the existing Phase 9 environment"
if [[ ! -x "${uv_bin}" ]]; then
    bash "${experiment_root}/scripts/environment/01_install_uv.sh"
fi
[[ -x "${uv_bin}" ]] || die "uv is unavailable: ${uv_bin}"
[[ -x "${venv}/bin/python" ]] || \
    die "Existing environment is missing: ${venv}; run the Phase 9 setup first"
python_bin="${venv}/bin/python"
"${uv_bin}" --version
"${python_bin}" -c 'import sys; print("python=" + sys.version.split()[0])'

section "Upgrade the editable Bhaskera package and required dependencies in place"
"${uv_bin}" pip install --python "${python_bin}" \
    setuptools packaging ninja wheel
"${uv_bin}" pip install --python "${python_bin}" \
    --editable "${bhaskera_root}[wandb,mlflow,inference,dev]"
"${uv_bin}" pip install --python "${python_bin}" \
    "bitsandbytes>=0.43" \
    "einops>=0.8" \
    "gdown>=5" \
    "opacus>=1.4" \
    "pyarrow>=15" \
    "scipy>=1.10" \
    "opt-einsum>=3.3"

# uv's editable setuptools build may leave this generated metadata beside the
# source tree. It is not Slakshna source and should not make the submodule dirty.
generated_egg_info="${bhaskera_root}/src/bhaskera.egg-info"
if [[ -d "${generated_egg_info}" ]]; then
    rm -rf -- "${generated_egg_info}"
fi

if (( install_vllm )); then
    section "Install vLLM by explicit request"
    echo "WARNING: review the resulting torch version before accepting this environment."
    "${uv_bin}" pip install --python "${python_bin}" vllm
fi

section "Validate Python dependencies and updated source imports"
"${uv_bin}" pip check --python "${python_bin}"
M0_BHASKERA_ROOT="${bhaskera_root}" \
M0_SKIP_GPU_CHECK="${skip_gpu_check}" \
"${python_bin}" - <<'PY'
import importlib
import os
from pathlib import Path

import torch
import transformers
from transformers import Olmo2Config, Olmo2ForCausalLM

required = [
    "accelerate", "bitsandbytes", "datasets", "einops", "flash_attn", "gdown",
    "liger_kernel", "mlflow", "opacus", "peft", "pyarrow", "ray",
    "safetensors", "wandb",
]
loaded = {name: importlib.import_module(name) for name in required}
bhaskera = importlib.import_module("bhaskera")
expected = (Path(os.environ["M0_BHASKERA_ROOT"]) / "src" / "bhaskera").resolve()
actual = [Path(path).resolve() for path in bhaskera.__path__]
if expected not in actual:
    raise RuntimeError(f"Bhaskera editable source mismatch: {actual}; expected {expected}")

# Exercise native OLMo 2 construction without downloading the 7B weights.
cfg = Olmo2Config(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=4,
)
model = Olmo2ForCausalLM(cfg)
attention = model.model.layers[0].self_attn
assert hasattr(attention, "q_proj") and hasattr(attention, "v_proj")

# Import the new upstream plugins explicitly; registration is tested here.
importlib.import_module("bhaskera.plugins.optimizers.muon")
importlib.import_module("bhaskera.plugins.optimizers.galore_muon")

print(f"torch={torch.__version__} cuda_build={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print("bhaskera=" + ",".join(str(path) for path in actual))
for name, module in sorted(loaded.items()):
    print(f"{name}={getattr(module, '__version__', 'installed')}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    value = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(value @ value, torch.mm(value, value))
    from flash_attn import flash_attn_func
    query = torch.randn(1, 32, 4, 64, device=device, dtype=torch.float16)
    output = flash_attn_func(query, query, query, causal=True)
    assert output.shape == query.shape
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print("flash_attn_cuda_forward=passed")
elif os.environ["M0_SKIP_GPU_CHECK"] != "1":
    raise RuntimeError("No GPU is visible; rerun on a GPU allocation or use --skip-gpu-check")
else:
    print("gpu=not visible; GPU checks explicitly deferred")
PY

rust_binary=""
if (( ! skip_rust )); then
    section "Build the latest Slakshna Rust binary"
    cargo_home="${runtime_root}/cargo"
    rustup_home="${runtime_root}/rustup"
    if [[ -x "${cargo_home}/bin/cargo" ]]; then
        export CARGO_HOME="${cargo_home}"
        export RUSTUP_HOME="${rustup_home}"
        export PATH="${CARGO_HOME}/bin:${PATH}"
    fi
    command -v cargo >/dev/null 2>&1 || die "cargo is unavailable"
    command -v rustc >/dev/null 2>&1 || die "rustc is unavailable"

    libclang_path="${M0_LIBCLANG_PATH:-${LIBCLANG_PATH:-}}"
    if [[ -z "${libclang_path}" ]]; then
        for candidate in \
            /usr/lib64/llvm21/lib64 \
            /usr/lib/llvm-18/lib \
            /usr/lib/llvm-17/lib \
            /usr/lib64; do
            if compgen -G "${candidate}/libclang.so*" >/dev/null; then
                libclang_path="${candidate}"
                break
            fi
        done
    fi
    [[ -n "${libclang_path}" && -d "${libclang_path}" ]] || \
        die "Could not locate libclang; set M0_LIBCLANG_PATH"

    # The M3 CUDA/compiler adapter intentionally loads GCC 10 for Python CUDA
    # extensions. LLVM 21's libclang, however, needs a newer libstdc++ than
    # that module provides. Keep the adapter active for Python, but isolate the
    # Rust build from the module's runtime and use the system C/C++ compilers.
    [[ -x /usr/bin/gcc && -x /usr/bin/g++ ]] || \
        die "System gcc/g++ are required for the Rust build"
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

    export CARGO_TARGET_DIR="${cargo_target}"
    env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS -u LIBRARY_PATH -u CPATH \
        CC=/usr/bin/gcc CXX=/usr/bin/g++ \
        LIBCLANG_PATH="${libclang_path}" \
        LD_LIBRARY_PATH="${rust_ld_library_path}" \
        cargo build --locked --release --manifest-path "${slakshna_root}/Cargo.toml"
    rust_binary="${cargo_target}/release/iiitd"
    [[ -x "${rust_binary}" ]] || die "Rust build did not produce ${rust_binary}"
    rust_ldd="$(env LD_LIBRARY_PATH="${rust_ld_library_path}" ldd "${rust_binary}")"
    ! grep -q 'not found' <<< "${rust_ldd}" || \
        die "Rust binary has unresolved runtime libraries"
    grep -q '/lib64/libstdc++.so.6' <<< "${rust_ldd}" || \
        die "Rust binary is not resolving the system libstdc++ runtime"
fi

section "Write M0 activation helper and manifests"
cat > "${activation_script}" <<EOF
#!/usr/bin/env bash
# Generated by monash_exps/scripts/m0/01_upgrade_stack.sh.
export SLAKSHNA_CLUSTER="${SLAKSHNA_CLUSTER}"
export SLAKSHNA_UV_ENVIRONMENT="${venv}"
export SLAKSHNA_M3_COMPILER_MODULE="${SLAKSHNA_M3_COMPILER_MODULE:-}"
export SLAKSHNA_M3_CUDA_MODULE="${SLAKSHNA_M3_CUDA_MODULE:-}"
export SLAKSHNA_SPARTAN_COMPILER_MODULE="${SLAKSHNA_SPARTAN_COMPILER_MODULE:-}"
export SLAKSHNA_SPARTAN_CUDA_MODULE="${SLAKSHNA_SPARTAN_CUDA_MODULE:-}"
export SLAKSHNA_SPARTAN_LLVM_MODULE="${SLAKSHNA_SPARTAN_LLVM_MODULE:-}"
source "${experiment_root}/scripts/cluster/activate.sh"
export M0_RUST_LD_LIBRARY_PATH="${rust_ld_library_path:-}"
EOF
chmod +x "${activation_script}"

freeze_path="${manifest_root}/environment-freeze.txt"
manifest_path="${manifest_root}/stack.json"
"${uv_bin}" pip freeze --python "${python_bin}" > "${freeze_path}"
M0_MANIFEST_PATH="${manifest_path}" \
M0_REVISION="${current_revision}" \
M0_ENVIRONMENT="${venv}" \
M0_FREEZE="${freeze_path}" \
M0_RUST_BINARY="${rust_binary}" \
M0_LOG_PATH="${log_path}" \
"${python_bin}" - <<'PY'
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

rust = Path(os.environ["M0_RUST_BINARY"]) if os.environ["M0_RUST_BINARY"] else None
payload = {
    "schema_version": 1,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "slakshna_revision": os.environ["M0_REVISION"],
    "environment": os.environ["M0_ENVIRONMENT"],
    "python": platform.python_version(),
    "freeze": os.environ["M0_FREEZE"],
    "freeze_sha256": sha256(Path(os.environ["M0_FREEZE"])),
    "rust_binary": str(rust) if rust else None,
    "rust_binary_sha256": sha256(rust) if rust else None,
    "log": os.environ["M0_LOG_PATH"],
}
path = Path(os.environ["M0_MANIFEST_PATH"])
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

echo
echo "M0 STACK UPGRADE PASSED"
echo "Environment : ${venv}"
echo "Activation  : ${activation_script}"
echo "Manifest    : ${manifest_path}"
echo "Log         : ${log_path}"
if [[ -n "${rust_binary}" ]]; then echo "Rust binary : ${rust_binary}"; fi
