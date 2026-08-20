#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
runtime_root="${experiment_root}/.runtime"
venv="${M0_UV_ENVIRONMENT:-${slakshna_root}/Bhaskera/.venv-phase9}"
python_bin="${venv}/bin/python"

template="${experiment_root}/configs/m0/smoke_olmo2_7b_single_a100.yaml"
run_root="${M0_SMOKE_RUN_ROOT:-${runtime_root}/runs/m0/smoke-olmo2-7b-single-a100}"
resolved_config="${run_root}/resolved-config.yaml"
preparation_manifest="${run_root}/preparation.json"
training_log="${run_root}/training.log"
gpu_csv="${run_root}/gpu-telemetry.csv"
result_manifest="${runtime_root}/manifests/m0/single-gpu-7b-smoke.json"
g0_root="${runtime_root}/artifacts/m0/g0"
g0_path="${g0_root}/olmo2-7b-r16-qv-seed20260820.pth"
g0_manifest="${g0_root}/olmo2-7b-r16-qv-seed20260820.json"

die() { echo "ERROR: $*" >&2; exit 1; }
section() { echo; echo "=== $* ==="; }

[[ -x "${python_bin}" ]] || die "Missing M0 Python environment: ${python_bin}"
[[ -f "${template}" ]] || die "Missing smoke template: ${template}"
[[ ! -f "${result_manifest}" ]] || die \
    "A passed smoke manifest already exists: ${result_manifest}"

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

export PATH="${venv}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR="${M0_SMOKE_RAY_TMPDIR:-/tmp/m0s$(id -u)}"
export TMPDIR="${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}" "$(dirname "${run_root}")" "${g0_root}"

if [[ -e "${run_root}" ]]; then
    failed_root="${runtime_root}/runs/m0/failed"
    mkdir -p "${failed_root}"
    archived="${failed_root}/smoke-olmo2-7b-single-a100-$(date +%Y%m%dT%H%M%S)"
    mv -- "${run_root}" "${archived}"
    echo "Preserved incomplete prior smoke at: ${archived}"
fi
mkdir -p "${run_root}"

workflow_started_epoch="$(date +%s)"

section "Validate the allocated GPU and fused-kernel path"
"${python_bin}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in this allocation")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory
if "A100" not in name or memory < 75_000 * 1024 * 1024:
    raise RuntimeError(f"expected one 80 GB A100, found {name} with {memory} bytes")
from flash_attn import flash_attn_func
query = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
output = flash_attn_func(query, query, query, causal=True)
assert output.shape == query.shape and torch.isfinite(output).all()
print(f"GPU: {name}; memory={memory / (1024 ** 3):.2f} GiB")
print("FlashAttention BF16 forward: PASS")
PY

section "Resolve the smoke configuration and prepare deterministic G0"
"${python_bin}" "${experiment_root}/src/m0/prepare_single_gpu_smoke.py" \
    --workspace-root "${workspace_root}" \
    --template "${template}" \
    --resolved-config "${resolved_config}" \
    --run-root "${run_root}" \
    --g0 "${g0_path}" \
    --g0-manifest "${g0_manifest}" \
    --preparation-manifest "${preparation_manifest}"

section "Run 12 native Bhaskera optimizer steps on one A100"
nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits \
    --loop=1 > "${gpu_csv}" 2> "${run_root}/gpu-monitor.stderr" &
monitor_pid=$!

stop_monitor() {
    if kill -0 "${monitor_pid}" 2>/dev/null; then
        kill "${monitor_pid}" 2>/dev/null || true
    fi
    wait "${monitor_pid}" 2>/dev/null || true
}
trap stop_monitor EXIT INT TERM

set +e
"${python_bin}" -m bhaskera.launcher.train \
    --config "${resolved_config}" \
    --num-workers 1 \
    --max-failures 0 \
    --storage-path "${run_root}/ray-results" \
    --no-dashboard 2>&1 | tee "${training_log}"
training_status=${PIPESTATUS[0]}
set -e
stop_monitor
trap - EXIT INT TERM
(( training_status == 0 )) || die \
    "Bhaskera smoke failed with status ${training_status}; retained ${run_root}"

workflow_completed_epoch="$(date +%s)"

section "Audit training, checkpoint, telemetry, and current FL wire payload"
"${python_bin}" "${experiment_root}/src/m0/audit_single_gpu_smoke.py" \
    --workspace-root "${workspace_root}" \
    --config "${resolved_config}" \
    --g0 "${g0_path}" \
    --g0-manifest "${g0_manifest}" \
    --preparation-manifest "${preparation_manifest}" \
    --training-log "${training_log}" \
    --gpu-csv "${gpu_csv}" \
    --output "${result_manifest}" \
    --started-epoch "${workflow_started_epoch}" \
    --completed-epoch "${workflow_completed_epoch}"

echo
echo "M0 SINGLE-GPU 7B SMOKE WORKFLOW PASSED"
echo "Result manifest: ${result_manifest}"
