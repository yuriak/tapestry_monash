#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
runtime_root="${experiment_root}/.runtime"
venv="${M0_UV_ENVIRONMENT:-${slakshna_root}/Bhaskera/.venv-phase9}"
python_bin="${venv}/bin/python"

prepared_root="${M0_PREPARED_ROOT:-${runtime_root}/data/m0/prepared}"
model_root="${M0_MODEL_ROOT:-${runtime_root}/models/m0}"
manifest_root="${runtime_root}/manifests/m0"
log_root="${runtime_root}/logs/m0"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "${python_bin}" ]] || die "Missing M0 Python environment: ${python_bin}"
[[ -f "${prepared_root}/manifests/prepared-data.json" ]] || \
    die "Missing prepared data manifest"
[[ -f "${model_root}/OLMo-2-1124-7B-Instruct/tokenizer.json" ]] || \
    die "Missing OLMo 2 7B tokenizer"
[[ -f "${model_root}/OLMo-2-0425-1B-Instruct-metadata/tokenizer.json" ]] || \
    die "Missing OLMo 2 1B tokenizer metadata"

mkdir -p "${manifest_root}" "${log_root}"
stamp="$(date +%Y%m%dT%H%M%S)"
log_path="${log_root}/tokenizer-length-audit-${stamp}.log"
exec > >(tee -a "${log_path}") 2>&1

echo "M0 tokenizer and sequence-length audit"
echo "  prepared : ${prepared_root}"
echo "  models   : ${model_root}"
echo "  log      : ${log_path}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
"${python_bin}" "${experiment_root}/src/m0/audit_tokenizer_lengths.py" \
    --workspace-root "${workspace_root}" \
    --prepared-root "${prepared_root}" \
    --model-7b "${model_root}/OLMo-2-1124-7B-Instruct" \
    --model-1b "${model_root}/OLMo-2-0425-1B-Instruct-metadata" \
    --output-json "${manifest_root}/tokenizer-length-audit.json" \
    --output-markdown "${manifest_root}/tokenizer-length-audit.md" \
    --batch-size "${M0_TOKENIZER_BATCH_SIZE:-256}" \
    --candidates 512 1024 2048 4096

echo
echo "M0 TOKENIZER AUDIT WORKFLOW PASSED"
echo "JSON report    : ${manifest_root}/tokenizer-length-audit.json"
echo "Markdown report: ${manifest_root}/tokenizer-length-audit.md"
