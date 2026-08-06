#!/usr/bin/env bash
# Submit from the workspace root so Slurm can create the relative log paths.
#SBATCH --output=monash_exps/slurm_logs/%j-%x.stdout
#SBATCH --error=monash_exps/slurm_logs/%j-%x.stderr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
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
    echo "Submit this job from the repository workspace root." >&2
    exit 2
fi

cd "${workspace_root}"

if [[ "$#" -lt 1 ]]; then
    echo "Usage: sbatch [cluster options] -J <name> monash_exps/scripts/slurm/submit_job_1gpu.sh <script> [args...]" >&2
    exit 2
fi

command_script="$1"
shift
if [[ ! -f "${command_script}" ]]; then
    echo "Command script does not exist from workspace root: ${command_script}" >&2
    exit 2
fi
command_script="$(realpath "${command_script}")"

# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

export SLAKSHNA_EXPECTED_GPUS="${SLAKSHNA_EXPECTED_GPUS:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 40000)))}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1048576}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

job_tmp="${TMPDIR:-/tmp}/slakshna-${SLURM_JOB_ID:-$$}"
mkdir -p "${job_tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-${job_tmp}/ray}"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "job_name=${SLURM_JOB_NAME:-unknown}"
echo "host=$(hostname)"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
echo "workspace_root=${workspace_root}"
echo "python=$(command -v python)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "expected_gpus=${SLAKSHNA_EXPECTED_GPUS}"
echo "master_addr=${MASTER_ADDR}"
echo "master_port=${MASTER_PORT}"
echo "ray_tmpdir=${RAY_TMPDIR}"
echo "command=bash ${command_script} $*"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

bash "${command_script}" "$@"
