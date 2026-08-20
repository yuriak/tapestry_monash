#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
runtime_root="${experiment_root}/.runtime"
venv="${M0_UV_ENVIRONMENT:-${slakshna_root}/Bhaskera/.venv-phase9}"
python_bin="${venv}/bin/python"

prepared_manifest="${M0_PREPARED_MANIFEST:-${runtime_root}/data/m0/prepared/manifests/prepared-data.json}"
decision="${M0_SEQUENCE_DECISION:-${runtime_root}/manifests/m0/sequence-length-decision.json}"
config="${M0_TOKENIZE_CONFIG:-${experiment_root}/configs/m0/tokenize_olmo2_7b_chatml.yaml}"
tokenized_root="${M0_TOKENIZED_ROOT:-${runtime_root}/data/m0/tokenized/olmo2-7b-chatml-seq1024}"
manifest="${runtime_root}/manifests/m0/tokenized-formal-views.json"
log_root="${runtime_root}/logs/m0"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "${python_bin}" ]] || die "Missing M0 Python environment: ${python_bin}"
[[ -f "${prepared_manifest}" ]] || die "Missing prepared-data manifest"
[[ -f "${decision}" ]] || die "Missing sequence-length decision"
[[ -f "${config}" ]] || die "Missing native tokenization config"

allocated_cpus="${SLURM_CPUS_ON_NODE:-$(nproc)}"
allocated_cpus="${allocated_cpus%%(*}"
if [[ ! "${allocated_cpus}" =~ ^[0-9]+$ ]]; then allocated_cpus=8; fi
workers="${M0_TOKENIZER_WORKERS:-$(( allocated_cpus < 8 ? allocated_cpus : 8 ))}"
if (( workers < 1 )); then workers=1; fi

mkdir -p "${log_root}"
stamp="$(date +%Y%m%dT%H%M%S)"
log_path="${log_root}/tokenize-formal-views-${stamp}.log"
exec > >(tee -a "${log_path}") 2>&1

echo "M0 native Bhaskera offline tokenization"
echo "  prepared : ${prepared_manifest}"
echo "  decision : ${decision}"
echo "  config   : ${config}"
echo "  output   : ${tokenized_root}"
echo "  workers  : ${workers}"
echo "  log      : ${log_path}"

cd "${workspace_root}"
"${python_bin}" "${experiment_root}/src/m0/tokenize_formal_views.py" \
    --workspace-root "${workspace_root}" \
    --prepared-manifest "${prepared_manifest}" \
    --sequence-decision "${decision}" \
    --config "${config}" \
    --tokenized-root "${tokenized_root}" \
    --manifest "${manifest}" \
    --workers "${workers}"

echo
echo "M0 OFFLINE TOKENIZATION WORKFLOW PASSED"
echo "Manifest: ${manifest}"
