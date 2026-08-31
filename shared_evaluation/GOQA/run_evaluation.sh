#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 MODEL OUTPUT_DIR [LORA_ADAPTER]" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
python_bin="${GOQA_PYTHON:-python3}"
model="$1"
output_dir="$(mkdir -p "$2" && cd "$2" && pwd -P)"
adapter="${3:-}"

"${python_bin}" "${script_dir}/validate_package.py"

inference_args=(
    --model "${model}"
    --dataset "${script_dir}/data/goqa_au_nz_india.jsonl"
    --output "${output_dir}/predictions.jsonl"
    --run-name "${GOQA_RUN_NAME:-model}"
    --request-batch-size "${GOQA_REQUEST_BATCH_SIZE:-8192}"
    --tensor-parallel-size "${GOQA_TENSOR_PARALLEL_SIZE:-1}"
    --gpu-memory-utilization "${GOQA_GPU_MEMORY_UTILIZATION:-0.90}"
    --max-model-len "${GOQA_MAX_MODEL_LEN:-512}"
    --max-num-batched-tokens "${GOQA_MAX_NUM_BATCHED_TOKENS:-32768}"
    --max-num-seqs "${GOQA_MAX_NUM_SEQS:-512}"
    --max-lora-rank "${GOQA_MAX_LORA_RANK:-64}"
    --dtype "${GOQA_DTYPE:-bfloat16}"
)
if [[ -n "${adapter}" ]]; then
    inference_args+=(--adapter "${adapter}")
fi
if [[ "${GOQA_TRUST_REMOTE_CODE:-0}" == "1" ]]; then
    inference_args+=(--trust-remote-code)
fi
if [[ "${GOQA_ENFORCE_EAGER:-0}" == "1" ]]; then
    inference_args+=(--enforce-eager)
fi

"${python_bin}" "${script_dir}/run_inference.py" "${inference_args[@]}"
"${python_bin}" "${script_dir}/validate_package.py" \
    --predictions "${output_dir}/predictions.jsonl"
"${python_bin}" "${script_dir}/score_predictions.py" \
    --dataset "${script_dir}/data/goqa_au_nz_india.jsonl" \
    --predictions "${output_dir}/predictions.jsonl" \
    --output-dir "${output_dir}/scores"

echo "GOQA AU/NZ/INDIA EVALUATION PASSED"
echo "Predictions: ${output_dir}/predictions.jsonl"
echo "Scores: ${output_dir}/scores"
