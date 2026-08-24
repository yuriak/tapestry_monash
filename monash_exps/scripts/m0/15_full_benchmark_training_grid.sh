#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runtime_root="${M0_RUNTIME_ROOT:-${repo_root}/Slakshna/m0_runtime}"
python_bin="${M0_VLLM_PYTHON:-$(command -v python3)}"
summary_python="${M0_SUMMARY_PYTHON:-${python_bin}}"
output_root="${M0_GRID_OUTPUT_ROOT:-${runtime_root}/evaluation/full-benchmark-training-grid}"
model="${repo_root}/monash_exps/.runtime/models/m0/OLMo-2-1124-7B-Instruct"
cultural_dir="${output_root}/culturalbench"
goqa_dir="${output_root}/global-opinion-qa-five-prompt"
summary_dir="${output_root}/combined"
mpl_config_dir="${output_root}/.matplotlib"

[[ -x "${python_bin}" ]] || {
    echo "ERROR: vLLM Python is unavailable: ${python_bin}" >&2
    exit 1
}
[[ -d "${runtime_root}" ]] || {
    echo "ERROR: M0 runtime root is unavailable: ${runtime_root}" >&2
    exit 1
}
[[ -x "${summary_python}" ]] || {
    echo "ERROR: summary Python is unavailable: ${summary_python}" >&2
    exit 1
}

mkdir -p "${output_root}" "${mpl_config_dir}"
printf 'Runtime: %s\nOutput root: %s\nvLLM Python: %s\nSummary Python: %s\n' \
    "$(realpath "${runtime_root}")" "${output_root}" "${python_bin}" "${summary_python}"

prepare_args=()
if [[ "${M0_PREPARE_ONLY:-0}" == "1" ]]; then
    prepare_args+=(--prepare-only)
fi

echo "=== 1/3 CulturalBench Easy + Hard: base and 35 retained adapters ==="
"${python_bin}" \
    "${repo_root}/monash_exps/src/m0/evaluate_culturalbench_grid_vllm.py" \
    --model "${model}" \
    --easy-dataset "${repo_root}/monash_exps/.runtime/data/m0/benchmarks/raw/CulturalBench/CulturalBench-Easy.csv" \
    --hard-dataset "${repo_root}/monash_exps/.runtime/data/m0/benchmarks/raw/CulturalBench/CulturalBench-Hard.csv" \
    --runtime-root "${runtime_root}" \
    --import-root "${runtime_root}/imported_spartan" \
    --output-dir "${cultural_dir}" \
    --request-batch-size "${M0_CULTURALBENCH_REQUEST_BATCH_SIZE:-8192}" \
    --gpu-memory-utilization "${M0_VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
    "${prepare_args[@]}"

echo "=== 2/3 GlobalOpinionQA: all countries, five prompts, same checkpoint grid ==="
"${python_bin}" \
    "${repo_root}/monash_exps/src/m0/evaluate_global_opinion_qa_five_prompt_vllm.py" \
    --model "${model}" \
    --dataset "${repo_root}/monash_exps/.runtime/data/m0/benchmarks/raw/GlobalOpinionQA/data/global_opinions.csv" \
    --runtime-root "${runtime_root}" \
    --import-root "${runtime_root}/imported_spartan" \
    --output-dir "${goqa_dir}" \
    --all-checkpoints \
    --request-batch-size "${M0_GOQA_REQUEST_BATCH_SIZE:-16384}" \
    --gpu-memory-utilization "${M0_VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
    "${prepare_args[@]}"

if [[ "${M0_PREPARE_ONLY:-0}" == "1" ]]; then
    echo "M0 FULL BENCHMARK GRID PREPARATION PASSED"
    exit 0
fi

echo "=== 3/3 Combined tables and trajectory figure ==="
env MPLCONFIGDIR="${mpl_config_dir}" "${summary_python}" \
    "${repo_root}/monash_exps/src/m0/summarize_benchmark_grid.py" \
    --cultural-dir "${cultural_dir}" \
    --goqa-dir "${goqa_dir}" \
    --output-dir "${summary_dir}"

echo "M0 FULL BENCHMARK TRAINING GRID PASSED"
echo "Results: ${output_root}"
