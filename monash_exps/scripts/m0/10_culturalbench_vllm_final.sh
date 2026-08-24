#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runtime_root="${M0_RUNTIME_ROOT:-${repo_root}/Slakshna/m0_runtime}"
import_root="${M0_SPARTAN_IMPORT_ROOT:-${runtime_root}/imported_spartan}"
python_bin="${M0_VLLM_PYTHON:-$(command -v python3)}"
model_dir="${M0_MODEL_DIR:-${repo_root}/monash_exps/.runtime/models/m0/OLMo-2-1124-7B-Instruct}"
dataset_path="${M0_CULTURALBENCH_CSV:-${repo_root}/monash_exps/.runtime/data/m0/benchmarks/raw/CulturalBench/CulturalBench-Easy.csv}"
timestamp="$(date +%Y%m%dT%H%M%S%z)"
output_dir="${M0_EVAL_OUTPUT_DIR:-${runtime_root}/evaluation/culturalbench-easy-final-${timestamp}}"

exec "${python_bin}" "${repo_root}/monash_exps/src/m0/evaluate_culturalbench_vllm.py" \
    --model "${model_dir}" \
    --dataset "${dataset_path}" \
    --runtime-root "${runtime_root}" \
    --import-root "${import_root}" \
    --output-dir "${output_dir}" \
    --runs \
        local_south_asia \
        local_variant_1 \
        local_variant_2 \
        local_variant_3 \
        central_variant_1 \
        central_variant_2 \
        central_variant_3 \
    --batch-size "${M0_EVAL_BATCH_SIZE:-256}" \
    --gpu-memory-utilization "${M0_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
