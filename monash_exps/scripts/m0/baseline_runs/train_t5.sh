#!/usr/bin/env bash
set -euo pipefail

run_name="${1:-${M0_RUN:-}}"
[[ -n "${run_name}" ]] || {
    echo "Usage: bash monash_exps/scripts/m0/baseline_runs/train_t5.sh <run-name>" >&2
    exit 2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
slakshna_root="${M0_SLAKSHNA_ROOT:-${repo_root}/Slakshna}"
config_template="${repo_root}/monash_exps/configs/m0/baseline_runs/${run_name}.yaml"
venv="${M0_UV_ENVIRONMENT:-${slakshna_root}/Bhaskera/.venv-phase9}"
python_bin="${venv}/bin/python"

case "${run_name}" in
    local_south_asia) cache_view="south_asia"; max_steps=1916 ;;
    local_variant_1) cache_view="australia_nz"; max_steps=1166 ;;
    local_variant_2) cache_view="australia_nz_western_europe"; max_steps=5630 ;;
    local_variant_3) cache_view="australia_nz_us_canada_uk"; max_steps=11238 ;;
    central_variant_1) cache_view="centralized_variant_1"; max_steps=3082 ;;
    central_variant_2) cache_view="centralized_variant_2"; max_steps=7546 ;;
    central_variant_3) cache_view="centralized_variant_3"; max_steps=13154 ;;
    *) echo "Unknown M0 T5 run: ${run_name}" >&2; exit 2 ;;
esac

if [[ -n "${M0_ASSET_ROOT:-}" ]]; then
    asset_root="${M0_ASSET_ROOT}"
elif [[ -d "${repo_root}/monash_exps/.runtime" ]]; then
    asset_root="${repo_root}/monash_exps/.runtime"
else
    asset_root="${slakshna_root}/m0_runtime/assets"
fi
output_root="${M0_OUTPUT_ROOT:-${slakshna_root}/m0_runtime}"

[[ -f "${config_template}" ]] || { echo "Missing config: ${config_template}" >&2; exit 1; }
[[ -x "${python_bin}" ]] || { echo "Missing Bhaskera environment: ${python_bin}" >&2; exit 1; }
[[ -d "${output_root}" ]] || {
    echo "Missing output root: ${output_root}" >&2
    echo "Create the shared m0_runtime link before launching." >&2
    exit 1
}

model_path="${asset_root}/models/m0/OLMo-2-1124-7B-Instruct"
g0_path="${asset_root}/artifacts/m0/g0/olmo2-7b-r16-qv-seed20260820.pth"
tokenized_path="${asset_root}/data/m0/tokenized/olmo2-7b-chatml-seq1024/${cache_view}/local_train_c4253f2a8d6a7d19"
[[ -d "${model_path}" ]] || { echo "Missing model: ${model_path}" >&2; exit 1; }
[[ -f "${g0_path}" ]] || { echo "Missing frozen G0: ${g0_path}" >&2; exit 1; }
[[ -d "${tokenized_path}" ]] || { echo "Missing token cache: ${tokenized_path}" >&2; exit 1; }
[[ -f "$(dirname "${tokenized_path}")/m0-cache-manifest.json" ]] || {
    echo "Missing audited cache manifest beside ${tokenized_path}" >&2
    exit 1
}
g0_sha256="$(sha256sum "${g0_path}" | awk '{print $1}')"
[[ "${g0_sha256}" == "0e87f53ad240ca04a4aaadc93079643e3b7cc1d0b38b7574e8b87e559361918c" ]] || {
    echo "Frozen G0 checksum mismatch: ${g0_path}" >&2
    exit 1
}

model_path="$(realpath "${model_path}")"
g0_path="$(realpath "${g0_path}")"
tokenized_path="$(realpath "${tokenized_path}")"
output_root="$(realpath "${output_root}")"
run_root="${output_root}/${run_name}"
dcp_dir="${run_root}/checkpoints"
adapter_dir="${run_root}/adapter_history"
resolved_config="${run_root}/resolved-config.yaml"
mkdir -p "${run_root}" "${output_root}/cache"

if [[ -f "${run_root}/COMPLETED" ]]; then
    echo "Run already completed: ${run_root}"
    exit 0
fi

escape_sed() {
    sed 's/[\\&|]/\\&/g' <<<"$1"
}

tmp_config="$(mktemp "${run_root}/.resolved-config.XXXXXX")"
sed \
    -e "s|__M0_MODEL_PATH__|$(escape_sed "${model_path}")|g" \
    -e "s|__M0_TOKENIZED_PATH__|$(escape_sed "${tokenized_path}")|g" \
    -e "s|__M0_G0_PATH__|$(escape_sed "${g0_path}")|g" \
    -e "s|__M0_DCP_DIR__|$(escape_sed "${dcp_dir}")|g" \
    -e "s|__M0_ADAPTER_DIR__|$(escape_sed "${adapter_dir}")|g" \
    "${config_template}" > "${tmp_config}"
if grep -q '__M0_[A-Z_]*__' "${tmp_config}"; then
    echo "Unresolved token in ${tmp_config}" >&2
    exit 1
fi
if [[ -f "${resolved_config}" ]]; then
    if ! cmp -s "${tmp_config}" "${resolved_config}"; then
        echo "Resolved config changed for an existing run: ${resolved_config}" >&2
        echo "Use a new output root or explicitly archive the existing run." >&2
        exit 1
    fi
    rm -f -- "${tmp_config}"
else
    mv -- "${tmp_config}" "${resolved_config}"
fi

if type module >/dev/null 2>&1; then
    restore_nounset=0
    case "$-" in *u*) restore_nounset=1; set +u ;; esac
    [[ -z "${M0_COMPILER_MODULE:-}" ]] || module load "${M0_COMPILER_MODULE}"
    [[ -z "${M0_CUDA_MODULE:-}" ]] || module load "${M0_CUDA_MODULE}"
    if [[ "${SLURM_CLUSTER_NAME:-}" == *m3* ]]; then
        module load "${M0_M3_COMPILER_MODULE:-gcc/10.2.0}"
        module load "${M0_M3_CUDA_MODULE:-cuda/12.8}"
    fi
    (( restore_nounset == 0 )) || set -u
fi

export VIRTUAL_ENV="${venv}"
export PATH="${venv}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${M0_HF_HOME:-${output_root}/cache/huggingface}"
export XDG_CACHE_HOME="${M0_XDG_CACHE_HOME:-${output_root}/cache/xdg}"
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR="${M0_RAY_TMPDIR:-/tmp/m0t5$(id -u)-${SLURM_JOB_ID:-manual}}"
export TMPDIR="${RAY_TMPDIR}"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${RAY_TMPDIR}"

if [[ "${M0_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    "${python_bin}" -c \
        'import os,sys; from bhaskera.config import load_config; c=load_config(sys.argv[1]); paths=[c.model.name,c.data.tokenized_path,c.lora.resume_path,c.checkpoint.save_dir,c.checkpoint.adapter_save_dir]; assert all(os.path.isabs(p) for p in paths); assert (c.training.batch_size,c.training.grad_accum,c.training.distributed.strategy)==(2,4,"ddp"); print({"max_steps":c.training.max_steps,"warmup_steps":c.training.warmup_steps,"adapter_interval":c.checkpoint.adapter_save_interval,"paths":paths})' \
        "${resolved_config}"
    echo "M0 preflight passed: ${run_name}"
    echo "Resolved config: ${resolved_config}"
    exit 0
fi

gpu_count="$(${python_bin} -c 'import torch; print(torch.cuda.device_count())')"
[[ "${gpu_count}" == "2" ]] || {
    echo "Expected exactly two visible GPUs, found ${gpu_count}" >&2
    exit 1
}
"${python_bin}" -c \
    'import torch; bad=[(i,torch.cuda.get_device_name(i),torch.cuda.get_device_properties(i).total_memory) for i in range(2) if "A100" not in torch.cuda.get_device_name(i) or torch.cuda.get_device_properties(i).total_memory < 75_000*1024*1024]; assert not bad, f"expected two 80 GB A100 GPUs, invalid devices: {bad}"'

telemetry="${run_root}/gpu-${SLURM_JOB_ID:-manual}.csv"
training_log="${run_root}/training.log"
nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits --loop=2 >> "${telemetry}" 2> "${run_root}/gpu-monitor.stderr" &
monitor_pid=$!

stop_monitor() {
    if kill -0 "${monitor_pid}" 2>/dev/null; then
        kill "${monitor_pid}" 2>/dev/null || true
    fi
    wait "${monitor_pid}" 2>/dev/null || true
}
trap stop_monitor EXIT INT TERM

cd "${slakshna_root}"
{
    echo "=== M0 T5 ${run_name} ==="
    echo "Started: $(date --iso-8601=seconds)"
    echo "Host: $(hostname)"
    echo "Slurm job: ${SLURM_JOB_ID:-manual}"
    echo "Config: ${resolved_config}"
    echo "Output: ${run_root}"
} | tee -a "${training_log}"

set +e
"${python_bin}" -m bhaskera.launcher.train \
    --config "${resolved_config}" \
    --num-workers 2 \
    --max-failures 0 \
    --storage-path "${run_root}/ray-results" \
    --no-dashboard 2>&1 | tee -a "${training_log}"
training_status=${PIPESTATUS[0]}
set -e
stop_monitor
trap - EXIT INT TERM
(( training_status == 0 )) || exit "${training_status}"

printf -v final_step 'step_%07d' "${max_steps}"
[[ -f "${adapter_dir}/${final_step}/.complete" ]] || {
    echo "Missing final adapter snapshot: ${adapter_dir}/${final_step}" >&2
    exit 1
}
[[ -f "${dcp_dir}/${final_step}/.complete" ]] || {
    echo "Missing final recoverable checkpoint: ${dcp_dir}/${final_step}" >&2
    exit 1
}

{
    echo "status=completed"
    echo "run=${run_name}"
    echo "step=${max_steps}"
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "config_sha256=$(sha256sum "${resolved_config}" | awk '{print $1}')"
    echo "adapter_sha256=$(sha256sum "${adapter_dir}/${final_step}/adapter_model.safetensors" | awk '{print $1}')"
} > "${run_root}/COMPLETED"

echo "M0 T5 RUN COMPLETED: ${run_name}"
echo "Completion record: ${run_root}/COMPLETED"
