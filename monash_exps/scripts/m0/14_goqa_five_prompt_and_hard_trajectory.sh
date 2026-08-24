#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runtime_root="${M0_RUNTIME_ROOT:-${repo_root}/Slakshna/m0_runtime}"
python_bin="${M0_VLLM_PYTHON:-$(command -v python3)}"
output_root="${M0_DIAGNOSTIC_OUTPUT_ROOT:-${runtime_root}/evaluation/paper-checks-goqa5-hard-trajectory}"
model="${repo_root}/monash_exps/.runtime/models/m0/OLMo-2-1124-7B-Instruct"

[[ -x "${python_bin}" ]] || {
    echo "ERROR: vLLM Python is unavailable: ${python_bin}" >&2
    exit 1
}
[[ -d "${runtime_root}" ]] || {
    echo "ERROR: M0 runtime root is unavailable: ${runtime_root}" >&2
    exit 1
}

mkdir -p "${output_root}"
printf 'Runtime: %s\nOutput root: %s\nPython: %s\n' \
    "$(realpath "${runtime_root}")" "${output_root}" "${python_bin}"

echo "=== 1/2 GlobalOpinionQA five-prompt evaluation ==="
"${python_bin}" \
    "${repo_root}/monash_exps/src/m0/evaluate_global_opinion_qa_five_prompt_vllm.py" \
    --model "${model}" \
    --dataset "${repo_root}/monash_exps/.runtime/data/m0/benchmarks/raw/GlobalOpinionQA/data/global_opinions.csv" \
    --runtime-root "${runtime_root}" \
    --import-root "${runtime_root}/imported_spartan" \
    --output-dir "${output_root}/global-opinion-qa-five-prompt" \
    --runs base local_south_asia local_variant_1 local_variant_2 local_variant_3 \
        central_variant_1 central_variant_2 central_variant_3 \
    --batch-size "${M0_GOQA_BATCH_SIZE:-128}" \
    --gpu-memory-utilization "${M0_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

echo "=== 2/2 CulturalBench-Hard checkpoint trajectories ==="
"${python_bin}" \
    "${repo_root}/monash_exps/src/m0/evaluate_culturalbench_hard_trajectory_vllm.py" \
    --model "${model}" \
    --dataset "${repo_root}/monash_exps/.runtime/data/m0/benchmarks/raw/CulturalBench/CulturalBench-Hard.csv" \
    --runtime-root "${runtime_root}" \
    --import-root "${runtime_root}/imported_spartan" \
    --output-dir "${output_root}/culturalbench-hard-trajectory" \
    --batch-size "${M0_HARD_BATCH_SIZE:-256}" \
    --gpu-memory-utilization "${M0_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

echo "M0 PAPER-COMPARISON DIAGNOSTICS PASSED"
echo "Results: ${output_root}"
