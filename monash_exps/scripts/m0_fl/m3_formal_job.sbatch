#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --gres=gpu:A100:2
#SBATCH --constraint=A100-80G
#SBATCH --time=06:00:00
#SBATCH --output=m0_runtime/slurm/%x-%j.out
#SBATCH --error=m0_runtime/slurm/%x-%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

workspace="${M0_FL_WORKSPACE:?M0_FL_WORKSPACE is required}"
run_id="${M0_FL_RUN_ID:?M0_FL_RUN_ID is required}"
output_root="${M0_FL_OUTPUT_ROOT:?M0_FL_OUTPUT_ROOT is required}"
playit_config="${M0_FL_PLAYIT_CONFIG:?M0_FL_PLAYIT_CONFIG is required}"
run_root="${output_root}/local_fl/${run_id}"

[[ -x "${workspace}/monash_exps/.runtime/venvs/primary/bin/python" ]] || {
    echo "Primary environment is missing under ${workspace}" >&2
    exit 1
}
[[ -x "${workspace}/monash_exps/.runtime/cargo-target/slakshna/release/iiitd" ]] || {
    echo "Slakshna release binary is missing under ${workspace}" >&2
    exit 1
}
[[ ! -e "${run_root}" ]] || {
    echo "Run output already exists; choose a new M0_FL_RUN_ID: ${run_root}" >&2
    exit 1
}
mkdir -p "${run_root}"

export SLAKSHNA_CLUSTER=m3
export SLAKSHNA_UV_ENVIRONMENT="${workspace}/monash_exps/.runtime/venvs/primary"
# shellcheck source=/dev/null
source "${workspace}/monash_exps/scripts/cluster/activate.sh"
export M0_FL_RUN_ROOT="${run_root}"
export M0_FL_FEDERATION_ID="m0-local-fl-${run_id}"

echo "M0 local FL run: ${run_id}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "Output: ${run_root}"

srun \
    --nodes=2 \
    --ntasks=2 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --gpus-per-task=2 \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    "${SLAKSHNA_UV_ENVIRONMENT}/bin/python" \
    -m monash_exps.src.m0_fl.run_formal_site

test -f "${run_root}/COMPLETED.json"
echo "M0 LOCAL FL JOB COMPLETED: ${run_root}"
