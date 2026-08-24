#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
slakshna_root="${M0_SLAKSHNA_ROOT:-${repo_root}/Slakshna}"
output_root="${M0_OUTPUT_ROOT:-${slakshna_root}/m0_runtime}"
job_script="${script_dir}/t5_job.sbatch"

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
extra_sbatch_args=("$@")

[[ -d "${output_root}" ]] || {
    echo "Missing output root: ${output_root}" >&2
    echo "Create the shared m0_runtime link before submission." >&2
    exit 1
}
if (( dry_run == 0 )); then
    mkdir -p "${output_root}/slurm"
fi
partition="${M0_SLURM_PARTITION:-fit}"
qos="${M0_SLURM_QOS:-fitq}"
account_mg61="${M0_ACCOUNT_MG61:-mg61}"
account_sq58="${M0_ACCOUNT_SQ58:-sq58}"

default_runs="local_south_asia local_variant_1 local_variant_2 local_variant_3 central_variant_1 central_variant_2 central_variant_3"
read -r -a runs <<<"${M0_RUNS:-${default_runs}}"
submission_log="${output_root}/slurm/submissions-$(date +%Y%m%dT%H%M%S).tsv"
if (( dry_run == 0 )); then
    printf 'run\tjob_id\taccount\tpartition\tqos\twalltime\tsubmitted_at\n' > "${submission_log}"
fi

cd "${slakshna_root}"
for run_name in "${runs[@]}"; do
    case "${run_name}" in
        local_variant_1) walltime="02:30:00"; account="${account_mg61}" ;;
        local_south_asia) walltime="03:30:00"; account="${account_mg61}" ;;
        central_variant_1) walltime="05:30:00"; account="${account_sq58}" ;;
        local_variant_2) walltime="09:00:00"; account="${account_sq58}" ;;
        central_variant_2) walltime="12:00:00"; account="${account_mg61}" ;;
        local_variant_3) walltime="18:00:00"; account="${account_mg61}" ;;
        central_variant_3) walltime="22:00:00"; account="${account_sq58}" ;;
        *) echo "Unknown run in M0_RUNS: ${run_name}" >&2; exit 2 ;;
    esac
    cmd=(
        sbatch
        --job-name="m0-${run_name//_/-}"
        --account="${account}"
        --partition="${partition}"
        --qos="${qos}"
        --time="${walltime}"
        --chdir="${slakshna_root}"
        --export="ALL,M0_RUN=${run_name},M0_REPO_ROOT=${repo_root},M0_SLAKSHNA_ROOT=${slakshna_root}"
    )
    cmd+=("${extra_sbatch_args[@]}" "${job_script}")
    if (( dry_run == 1 )); then
        printf 'DRY RUN:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
    fi
    response="$("${cmd[@]}")"
    job_id="${response##* }"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${run_name}" "${job_id}" "${account}" "${partition}" "${qos}" \
        "${walltime}" "$(date --iso-8601=seconds)" \
        >> "${submission_log}"
    echo "${run_name} [${account}/${partition}/${qos}]: ${response}"
done

if (( dry_run == 0 )); then
    echo "Submission manifest: ${submission_log}"
fi
