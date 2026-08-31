#!/usr/bin/env bash
set -euo pipefail

mode="${1:---test-only}"
case "${mode}" in
    --test-only) submit_flag="--test-only" ;;
    --submit) submit_flag="" ;;
    *) echo "Usage: bash monash_exps/scripts/m0_fl/submit_spartan.sh --test-only|--submit" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace="$(cd "${script_dir}/../../.." && pwd -P)"
output_root="$(readlink -f "${workspace}/Slakshna/m0_runtime")"
playit_config="${M0_FL_PLAYIT_CONFIG:-${workspace}/monash_exps/.runtime/configs/m0_fl/spartan_playit.toml}"
run_id="${M0_FL_RUN_ID:-m0-local-fl-$(date +%Y%m%dT%H%M%S)}"

[[ -f "${playit_config}" ]] || {
    echo "Missing Playit configuration: ${playit_config}" >&2
    exit 1
}
[[ -d "${output_root}/slurm" ]] || {
    echo "Missing output directory: ${output_root}/slurm" >&2
    exit 1
}

command=(
    sbatch
    --job-name=m0-local-fl
    --account="${M0_FL_SPARTAN_ACCOUNT:-punim2961}"
    --partition="${M0_FL_SPARTAN_PARTITION:-feit-gpu-a100}"
    --qos="${M0_FL_SPARTAN_QOS:-feit}"
    --chdir="${workspace}"
    --export="ALL,M0_FL_WORKSPACE=${workspace},M0_FL_OUTPUT_ROOT=${output_root},M0_FL_RUN_ID=${run_id},M0_FL_PLAYIT_CONFIG=${playit_config}"
)
[[ -z "${submit_flag}" ]] || command+=("${submit_flag}")
command+=("${workspace}/monash_exps/scripts/m0_fl/spartan_formal_job.sbatch")

printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"

if [[ "${mode}" == "--test-only" ]]; then
    echo "Prediction only; no job was submitted."
else
    echo "Submitted run ID: ${run_id}"
    echo "Expected output: ${output_root}/local_fl/${run_id}"
fi
