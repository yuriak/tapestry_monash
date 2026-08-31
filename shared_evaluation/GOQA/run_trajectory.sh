#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 MODEL OUTPUT_DIR base [NAME=LORA_ADAPTER ...]" >&2
    echo "       $0 MODEL OUTPUT_DIR NAME=LORA_ADAPTER [NAME=LORA_ADAPTER ...]" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
python_bin="${GOQA_PYTHON:-python3}"
source_model="$(cd "$1" && pwd -P)"
output_dir="$(mkdir -p "$2" && cd "$2" && pwd -P)"
shift 2

model="${source_model}"
if [[ -n "${GOQA_STAGE_MODEL_DIR:-}" ]]; then
    bash "${script_dir}/stage_model.sh" "${source_model}" "${GOQA_STAGE_MODEL_DIR}"
    model="$(cd "${GOQA_STAGE_MODEL_DIR}" && pwd -P)"
fi
tokenizer="${GOQA_TOKENIZER:-${model}}"

trajectory_args=(
    --model "${model}"
    --tokenizer "${tokenizer}"
    --dataset "${script_dir}/data/goqa_au_nz_india.jsonl"
    --output-dir "${output_dir}"
    --request-batch-size "${GOQA_REQUEST_BATCH_SIZE:-8192}"
    --tensor-parallel-size "${GOQA_TENSOR_PARALLEL_SIZE:-1}"
    --gpu-memory-utilization "${GOQA_GPU_MEMORY_UTILIZATION:-0.90}"
    --max-model-len "${GOQA_MAX_MODEL_LEN:-512}"
    --max-num-batched-tokens "${GOQA_MAX_NUM_BATCHED_TOKENS:-32768}"
    --max-num-seqs "${GOQA_MAX_NUM_SEQS:-512}"
    --max-lora-rank "${GOQA_MAX_LORA_RANK:-64}"
    --dtype "${GOQA_DTYPE:-bfloat16}"
)

names=()
for specification in "$@"; do
    if [[ "${specification}" == "base" ]]; then
        trajectory_args+=(--include-base)
        names+=(base)
    elif [[ "${specification}" == *=* ]]; then
        name="${specification%%=*}"
        trajectory_args+=(--adapter "${specification}")
        names+=("${name}")
    else
        echo "Invalid trajectory member: ${specification}" >&2
        exit 2
    fi
done

if [[ "${GOQA_TRUST_REMOTE_CODE:-0}" == "1" ]]; then
    trajectory_args+=(--trust-remote-code)
fi
if [[ "${GOQA_ENFORCE_EAGER:-0}" == "1" ]]; then
    trajectory_args+=(--enforce-eager)
fi

"${python_bin}" "${script_dir}/validate_package.py"
"${python_bin}" "${script_dir}/run_trajectory.py" "${trajectory_args[@]}"

for name in "${names[@]}"; do
    predictions="${output_dir}/${name}/predictions.jsonl"
    "${python_bin}" "${script_dir}/validate_package.py" --predictions "${predictions}"
    "${python_bin}" "${script_dir}/score_predictions.py" \
        --dataset "${script_dir}/data/goqa_au_nz_india.jsonl" \
        --predictions "${predictions}" \
        --output-dir "${output_dir}/${name}/scores"
done

echo "GOQA TRAJECTORY EVALUATION PASSED"
echo "Results: ${output_dir}"
