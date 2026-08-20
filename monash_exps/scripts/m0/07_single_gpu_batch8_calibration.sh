#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
runtime_root="${experiment_root}/.runtime"
venv="${M0_UV_ENVIRONMENT:-${slakshna_root}/Bhaskera/.venv-phase9}"
python_bin="${venv}/bin/python"

config_template="${experiment_root}/configs/m0/calibrate_olmo2_7b_single_a100_batch8.yaml"
run_root="${M0_BATCH8_RUN_ROOT:-${runtime_root}/runs/m0/calibrate-olmo2-7b-single-a100-batch8}"
resolved_config="${run_root}/resolved-config.yaml"
training_log="${run_root}/training.log"
gpu_csv="${run_root}/gpu-telemetry.csv"
model_path="${runtime_root}/models/m0/OLMo-2-1124-7B-Instruct"
tokenized_path="${runtime_root}/data/m0/tokenized/olmo2-7b-chatml-seq1024/australia_nz/local_train_c4253f2a8d6a7d19"
g0_path="${runtime_root}/artifacts/m0/g0/olmo2-7b-r16-qv-seed20260820.pth"
g0_manifest="${runtime_root}/artifacts/m0/g0/olmo2-7b-r16-qv-seed20260820.json"
token_manifest="${runtime_root}/manifests/m0/tokenized-formal-views.json"

die() { echo "ERROR: $*" >&2; exit 1; }
section() { echo; echo "=== $* ==="; }

[[ -x "${python_bin}" ]] || die "Missing M0 Python environment: ${python_bin}"
[[ -f "${config_template}" ]] || die "Missing batch-8 calibration config template"
[[ -d "${model_path}" ]] || die "Missing local OLMo 2 model: ${model_path}"
[[ -d "${tokenized_path}" ]] || die "Missing Australia/New Zealand token cache: ${tokenized_path}"
[[ -f "${g0_path}" && -f "${g0_manifest}" ]] || die "Missing frozen G0"
[[ -f "${token_manifest}" ]] || die "Missing formal token-cache manifest"

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
export RAY_TMPDIR="${M0_BATCH8_RAY_TMPDIR:-/tmp/m0b8$(id -u)}"
export TMPDIR="${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}" "$(dirname "${run_root}")"

if [[ -e "${run_root}" ]]; then
    archive_root="${runtime_root}/runs/m0/calibration-archive"
    mkdir -p "${archive_root}"
    archived="${archive_root}/batch8-$(date +%Y%m%dT%H%M%S)"
    mv -- "${run_root}" "${archived}"
    echo "Preserved previous batch-8 calibration at: ${archived}"
fi
mkdir -p "${run_root}"

# Ray workers do not inherit the driver's working directory. Materialize a
# run-local config with absolute paths so Transformers never interprets a
# relative filesystem path as a Hugging Face repository identifier.
sed \
    -e "s|__M0_MODEL_PATH__|${model_path}|g" \
    -e "s|__M0_TOKENIZED_PATH__|${tokenized_path}|g" \
    -e "s|__M0_G0_PATH__|${g0_path}|g" \
    -e "s|__M0_CHECKPOINT_PATH__|${run_root}/checkpoints-disabled|g" \
    "${config_template}" > "${resolved_config}"
if grep -q '__M0_[A-Z_]*__' "${resolved_config}"; then
    die "Unresolved path token remains in ${resolved_config}"
fi

section "Validate the resolved calibration contract and allocated GPU"
cd "${workspace_root}"
M0_BATCH8_CONFIG="${resolved_config}" \
M0_G0_PATH="${g0_path}" \
M0_G0_MANIFEST="${g0_manifest}" \
M0_TOKEN_MANIFEST="${token_manifest}" \
"${python_bin}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch
from bhaskera.config import load_config

cfg = load_config(os.environ["M0_BATCH8_CONFIG"])
expected = {
    "attn_impl": "flash_attention_2",
    "seq_len": 1024,
    "batch_size": 8,
    "grad_accum": 1,
    "max_steps": 5,
    "strategy": "ddp",
    "checkpoint_enabled": False,
}
actual = {
    "attn_impl": cfg.model.attn_impl,
    "seq_len": cfg.data.seq_len,
    "batch_size": cfg.training.batch_size,
    "grad_accum": cfg.training.grad_accum,
    "max_steps": cfg.training.max_steps,
    "strategy": cfg.training.distributed.strategy,
    "checkpoint_enabled": cfg.checkpoint.enabled,
}
if actual != expected:
    raise RuntimeError(f"batch-8 calibration config mismatch: {actual}")

for label, value in {
    "model": cfg.model.name,
    "tokenized data": cfg.data.tokenized_path,
    "G0": cfg.lora.resume_path,
    "checkpoint directory": cfg.checkpoint.save_dir,
}.items():
    if not Path(value).is_absolute():
        raise RuntimeError(f"{label} path is not absolute: {value}")

g0_path = Path(os.environ["M0_G0_PATH"])
g0_manifest = json.loads(Path(os.environ["M0_G0_MANIFEST"]).read_text())
digest = hashlib.sha256(g0_path.read_bytes()).hexdigest()
if digest != g0_manifest["state_file_sha256"]:
    raise RuntimeError("G0 hash mismatch")
token_manifest = json.loads(Path(os.environ["M0_TOKEN_MANIFEST"]).read_text())
if token_manifest.get("seq_len") != 1024 or len(token_manifest.get("views", {})) != 7:
    raise RuntimeError("formal token cache is incomplete")

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError(f"expected one visible GPU, found {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory
if "A100" not in name or memory < 75_000 * 1024 * 1024:
    raise RuntimeError(f"expected one 80 GB A100, found {name}")
print(f"Static config: {actual}")
print(f"G0 SHA256: {digest}")
print(f"GPU: {name}; memory={memory / (1024 ** 3):.2f} GiB")
PY

section "Run five native Bhaskera steps at per-device batch 8"
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
    "Batch-8 calibration failed with status ${training_status}; retained ${run_root}"

section "Print review summary"
loss_count="$(grep -Eo '\[epoch [0-9]+\]\[step [0-9]+\] loss=[0-9.eE+-]+' "${training_log}" | sort -u | awk 'END{print NR}')"
[[ "${loss_count}" == "5" ]] || die "Expected five loss records, found ${loss_count}"
grep -E '\[epoch [0-9]+\]\[step [0-9]+\] loss=' "${training_log}" | sed -E 's/^.*(\[epoch [0-9]+\]\[step [0-9]+\] loss=.*)$/\1/' | sort -u

peak_memory="$(awk -F',' 'BEGIN{m=0} NF==7 {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5); if (($5+0)>m) m=$5+0} END{printf "%.0f",m}' "${gpu_csv}")"
peak_util="$(awk -F',' 'BEGIN{m=0} NF==7 {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); if (($4+0)>m) m=$4+0} END{printf "%.0f",m}' "${gpu_csv}")"
echo "Peak monitored GPU memory: ${peak_memory} MiB"
echo "Peak monitored GPU utilization: ${peak_util}%"
echo "Training log: ${training_log}"
echo "GPU telemetry: ${gpu_csv}"
echo "Resolved config: ${resolved_config}"
echo
echo "M0 BATCH-8 FIVE-STEP CALIBRATION COMPLETED; REVIEW REQUIRED"
