#!/usr/bin/env bash
set -euo pipefail

mode="${1:---test-only}"
case "${mode}" in
    --test-only) submit_flag="--test-only" ;;
    --submit) submit_flag="" ;;
    *) echo "Usage: bash m0_fl_submit_m3.sh --test-only|--submit" >&2; exit 2 ;;
esac

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_root="$(readlink -f "${workspace}/Slakshna/m0_runtime")"
playit_config="${M0_FL_PLAYIT_CONFIG:-${workspace}/m0_fl_m3_playit.toml}"
job_script="${workspace}/monash_exps/scripts/m0_fl/m3_formal_job.sbatch"
run_id="${M0_FL_RUN_ID:-m0-local-fl-$(date +%Y%m%dT%H%M%S)}"

[[ -d "${output_root}/slurm" ]] || {
    echo "M0 output/slurm directory is missing: ${output_root}/slurm" >&2
    exit 1
}
[[ -f "${playit_config}" ]] || {
    echo "Playit configuration is missing: ${playit_config}" >&2
    exit 1
}

command=(
    sbatch
    --job-name=m0-local-fl
    --account="${M0_FL_M3_ACCOUNT:-mg61}"
    --partition="${M0_FL_M3_PARTITION:-fit}"
    --qos="${M0_FL_M3_QOS:-fitq}"
    --chdir="${workspace}"
    --export="ALL,M0_FL_WORKSPACE=${workspace},M0_FL_OUTPUT_ROOT=${output_root},M0_FL_RUN_ID=${run_id},M0_FL_PLAYIT_CONFIG=${playit_config}"
)
[[ -z "${submit_flag}" ]] || command+=("${submit_flag}")
command+=("${job_script}")

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
