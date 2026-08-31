#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
run_root="${M0_FL_RUN_ROOT:?Set M0_FL_RUN_ROOT to the completed local-FL run}"
training_python="${repo_root}/monash_exps/.runtime/venvs/primary/bin/python"
vllm_python="${M0_VLLM_PYTHON:-/fs04/da33/minghanw/env/reason/bin/python}"
model="${M0_MODEL_DIR:-${repo_root}/monash_exps/.runtime/models/m0/OLMo-2-1124-7B-Instruct}"
benchmark_root="${repo_root}/monash_exps/.runtime/data/m0/benchmarks/raw"
output_root="${M0_FL_EVAL_OUTPUT_ROOT:-${run_root}/evaluation/full-grid}"
adapter_manifest="${run_root}/evaluation_grid/manifest.json"

[[ -f "${run_root}/COMPLETED.json" ]] || {
    echo "Completed FL run is missing: ${run_root}" >&2
    exit 1
}
[[ -x "${training_python}" && -x "${vllm_python}" ]] || {
    echo "Required Python environment is missing" >&2
    exit 1
}
[[ -f "${model}/config.json" ]] || {
    echo "Base model is missing: ${model}" >&2
    exit 1
}
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ]] || {
    echo "This evaluation requires two visible GPUs" >&2
    exit 1
}

"${training_python}" -m monash_exps.src.m0_fl.finalize_round10 \
    --run-root "${run_root}"
"${training_python}" -m monash_exps.src.m0_fl.build_evaluation_grid \
    --run-root "${run_root}"

mkdir -p "${output_root}/logs"
common=(
    --model "${model}"
    --runtime-root "$(readlink -f "${repo_root}/Slakshna/m0_runtime")"
    --import-root "$(readlink -f "${repo_root}/Slakshna/m0_runtime")/imported_spartan"
    --adapter-manifest "${adapter_manifest}"
    --prepare-only
)
"${vllm_python}" "${repo_root}/monash_exps/src/m0/evaluate_culturalbench_grid_vllm.py" \
    "${common[@]}" \
    --easy-dataset "${benchmark_root}/CulturalBench/CulturalBench-Easy.csv" \
    --hard-dataset "${benchmark_root}/CulturalBench/CulturalBench-Hard.csv" \
    --output-dir "${output_root}/culturalbench" \
    --request-batch-size "${M0_CULTURAL_REQUEST_BATCH_SIZE:-8192}"
"${vllm_python}" "${repo_root}/monash_exps/src/m0/evaluate_global_opinion_qa_five_prompt_vllm.py" \
    "${common[@]}" \
    --dataset "${benchmark_root}/GlobalOpinionQA/data/global_opinions.csv" \
    --output-dir "${output_root}/global-opinion-qa-five-prompt" \
    --request-batch-size "${M0_GOQA_REQUEST_BATCH_SIZE:-16384}"

if [[ "${M0_FL_PREPARE_ONLY:-0}" == "1" ]]; then
    echo "M0 LOCAL FL EVALUATION PREPARATION PASSED"
    echo "Prepared results root: ${output_root}"
    exit 0
fi

cultural_log="${output_root}/logs/culturalbench.log"
goqa_log="${output_root}/logs/global-opinion-qa.log"
pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM EXIT

CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
"${vllm_python}" "${repo_root}/monash_exps/src/m0/evaluate_culturalbench_grid_vllm.py" \
    --model "${model}" \
    --easy-dataset "${benchmark_root}/CulturalBench/CulturalBench-Easy.csv" \
    --hard-dataset "${benchmark_root}/CulturalBench/CulturalBench-Hard.csv" \
    --runtime-root "$(readlink -f "${repo_root}/Slakshna/m0_runtime")" \
    --import-root "$(readlink -f "${repo_root}/Slakshna/m0_runtime")/imported_spartan" \
    --adapter-manifest "${adapter_manifest}" \
    --output-dir "${output_root}/culturalbench" \
    --request-batch-size "${M0_CULTURAL_REQUEST_BATCH_SIZE:-8192}" \
    --gpu-memory-utilization "${M0_VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
    >"${cultural_log}" 2>&1 &
pids+=("$!")
cultural_pid="$!"

CUDA_VISIBLE_DEVICES=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
"${vllm_python}" "${repo_root}/monash_exps/src/m0/evaluate_global_opinion_qa_five_prompt_vllm.py" \
    --model "${model}" \
    --dataset "${benchmark_root}/GlobalOpinionQA/data/global_opinions.csv" \
    --runtime-root "$(readlink -f "${repo_root}/Slakshna/m0_runtime")" \
    --import-root "$(readlink -f "${repo_root}/Slakshna/m0_runtime")/imported_spartan" \
    --adapter-manifest "${adapter_manifest}" \
    --output-dir "${output_root}/global-opinion-qa-five-prompt" \
    --request-batch-size "${M0_GOQA_REQUEST_BATCH_SIZE:-16384}" \
    --gpu-memory-utilization "${M0_VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
    >"${goqa_log}" 2>&1 &
pids+=("$!")
goqa_pid="$!"

echo "CulturalBench PID=${cultural_pid} GPU=0 log=${cultural_log}"
echo "GlobalOpinionQA PID=${goqa_pid} GPU=1 log=${goqa_log}"

status=0
wait "${cultural_pid}" || status=1
wait "${goqa_pid}" || status=1
trap - INT TERM EXIT
if [[ "${status}" -ne 0 ]]; then
    echo "One or more FL evaluations failed; inspect ${output_root}/logs" >&2
    exit "${status}"
fi

test -f "${output_root}/culturalbench/summary.csv"
test -f "${output_root}/global-opinion-qa-five-prompt/summary.csv"
echo "M0 LOCAL FL FULL EVALUATION PASSED"
echo "Results: ${output_root}"
