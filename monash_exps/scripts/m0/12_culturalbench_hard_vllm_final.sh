#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runtime_root="${M0_RUNTIME_ROOT:-${repo_root}/Slakshna/m0_runtime}"
python_bin="${M0_VLLM_PYTHON:-$(command -v python3)}"
timestamp="$(date +%Y%m%dT%H%M%S%z)"
exec "${python_bin}" \
    "${repo_root}/monash_exps/src/m0/evaluate_culturalbench_hard_vllm.py" \
    --model "${repo_root}/monash_exps/.runtime/models/m0/OLMo-2-1124-7B-Instruct" \
    --dataset "${repo_root}/monash_exps/.runtime/data/m0/benchmarks/raw/CulturalBench/CulturalBench-Hard.csv" \
    --runtime-root "${runtime_root}" --import-root "${runtime_root}/imported_spartan" \
    --output-dir "${M0_EVAL_OUTPUT_DIR:-${runtime_root}/evaluation/culturalbench-hard-final-${timestamp}}" \
    --runs base local_south_asia local_variant_1 local_variant_2 local_variant_3 central_variant_1 central_variant_2 central_variant_3 \
    --batch-size "${M0_EVAL_BATCH_SIZE:-256}" \
    --gpu-memory-utilization "${M0_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
