#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
primary_python="${repo_root}/monash_exps/.runtime/venvs/primary/bin/python"
vllm_python="${JOINT_GOQA_VLLM_PYTHON:-/fs04/da33/minghanw/env/reason/bin/python}"
base_model="${JOINT_GOQA_BASE_MODEL:-${repo_root}/Slakshna/hf_cache/hub/models--allenai--OLMo-2-1124-7B/snapshots/7df9a82518afdecae4e8c026b27adccc8c1f0032}"
tokenizer="${JOINT_GOQA_TOKENIZER:-${repo_root}/monash_exps/.runtime/models/m0/OLMo-2-1124-7B-Instruct}"
node_id="slakshna1vxwxlxaznw253ucsx4djr422y9wp2ej5kuavu0"
checkpoint_dir="${JOINT_GOQA_CHECKPOINT_DIR:-${repo_root}/Slakshna/ml_models/sync_ckpt_${node_id}}"
output_root="${JOINT_GOQA_OUTPUT_ROOT:-/fs04/scratch2/da33/minghanw/tapestry_runtime/cross_country_fl/evaluation/goqa_through_peer_offline_correct_base}"
goqa_dir="${repo_root}/shared_evaluation/GOQA"
request_batch_size="${JOINT_GOQA_REQUEST_BATCH_SIZE:-8192}"
gpu_memory_utilization="${JOINT_GOQA_GPU_MEMORY_UTILIZATION:-0.90}"
max_num_batched_tokens="${JOINT_GOQA_MAX_NUM_BATCHED_TOKENS:-32768}"
max_num_seqs="${JOINT_GOQA_MAX_NUM_SEQS:-512}"

[[ -x "${primary_python}" && -x "${vllm_python}" ]] || {
    echo "Required Python environment is missing" >&2
    exit 1
}
[[ -f "${base_model}/config.json" ]] || {
    echo "Base model is missing: ${base_model}" >&2
    exit 1
}
[[ -f "${tokenizer}/tokenizer_config.json" ]] || {
    echo "Tokenizer is missing: ${tokenizer}" >&2
    exit 1
}
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ]] || {
    echo "Two visible GPUs are required" >&2
    exit 1
}
if pgrep -f '[m]l_engine.py|/iiitd( |$)' >/dev/null; then
    echo "An FL training process is still using this node; stop it before evaluation." >&2
    exit 1
fi

mkdir -p "${output_root}/logs"
# vLLM uses TMPDIR for Unix-domain sockets, whose path is limited to 107
# characters on Linux. Keep this path deliberately short even though durable
# outputs belong under output_root.
tmp_root="$(mktemp -d /tmp/jgqa.XXXXXX)"
export TMPDIR="${tmp_root}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

"${primary_python}" "${repo_root}/monash_exps/src/phase9/prepare_joint_goqa_trajectory.py" \
    --checkpoint-dir "${checkpoint_dir}" \
    --base-model "${base_model}" \
    --joint-log "${repo_root}/Slakshna/monash_joint_20260826_retry.log" \
    --runtime-log "${repo_root}/Slakshna/logs/runtime_comm.log" \
    --output-dir "${output_root}" \
    --last-round 21

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    rm -rf -- "${tmp_root}"
}
trap cleanup INT TERM EXIT

for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES="${gpu}" \
    TORCHINDUCTOR_CACHE_DIR="${tmp_root}/inductor_gpu${gpu}" \
    "${vllm_python}" "${repo_root}/monash_exps/src/phase9/evaluate_joint_goqa_trajectory_vllm.py" \
        --manifest "${output_root}/adapter_manifest_gpu${gpu}.json" \
        --goqa-dir "${goqa_dir}" \
        --output-dir "${output_root}/results" \
        --tokenizer "${tokenizer}" \
        --request-batch-size "${request_batch_size}" \
        --gpu-memory-utilization "${gpu_memory_utilization}" \
        --max-num-batched-tokens "${max_num_batched_tokens}" \
        --max-num-seqs "${max_num_seqs}" \
        >"${output_root}/logs/gpu${gpu}_inference.log" 2>&1 &
    pids+=("$!")
    echo "GPU ${gpu}: PID=$! log=${output_root}/logs/gpu${gpu}_inference.log"
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
trap - INT TERM EXIT
rm -rf -- "${tmp_root}"
if [[ "${status}" -ne 0 ]]; then
    echo "One or more inference shards failed; inspect ${output_root}/logs" >&2
    exit 1
fi

for result_dir in "${output_root}"/results/*; do
    [[ -d "${result_dir}" && -f "${result_dir}/predictions.jsonl" ]] || continue
    "${primary_python}" "${goqa_dir}/validate_package.py" --predictions "${result_dir}/predictions.jsonl"
    "${primary_python}" "${goqa_dir}/score_predictions.py" \
        --dataset "${goqa_dir}/data/goqa_au_nz_india.jsonl" \
        --predictions "${result_dir}/predictions.jsonl" \
        --output-dir "${result_dir}/scores"
done

MPLCONFIGDIR="${output_root}/matplotlib" \
"${primary_python}" "${repo_root}/monash_exps/src/phase9/summarize_joint_goqa_trajectory.py" \
    --manifest "${output_root}/adapter_manifest.json" \
    --evaluation-dir "${output_root}/results"

echo "JOINT GOQA TRAJECTORY EVALUATION PASSED"
echo "Results: ${output_root}/results"
