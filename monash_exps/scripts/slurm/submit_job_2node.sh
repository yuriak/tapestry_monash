#!/usr/bin/env bash
#SBATCH --output=monash_exps/slurm_logs/%j-%x.stdout
#SBATCH --error=monash_exps/slurm_logs/%j-%x.stderr
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    workspace_root="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    workspace_root="$(cd "${script_dir}/../../.." && pwd)"
fi
experiment_root="${workspace_root}/monash_exps"
if [[ ! -f "${experiment_root}/environment/pyproject.toml" ]]; then
    echo "Workspace root is invalid: ${workspace_root}" >&2
    exit 2
fi
if [[ "$#" -lt 1 ]]; then
    echo "Usage: sbatch [options] submit_job_2node.sh <script> [args...]" >&2
    exit 2
fi

cd "${workspace_root}"
command_script="$1"
shift
test -f "${command_script}"
command_script="$(realpath "${command_script}")"

# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"
export SLAKSHNA_EXPECTED_GPUS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "job_name=${SLURM_JOB_NAME:-unknown}"
echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
echo "num_nodes=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-unknown}}"
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-unknown}"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "workspace_root=${workspace_root}"
echo "command=bash ${command_script} $*"

bash "${command_script}" "$@"
