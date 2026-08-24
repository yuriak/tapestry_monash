#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --gres=gpu:A100:2
#SBATCH --time=1-00:00:00
#SBATCH --output=m0_runtime/slurm/%x-%j.out
#SBATCH --error=m0_runtime/slurm/%x-%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

[[ -n "${M0_RUN:-}" ]] || { echo "M0_RUN is not set" >&2; exit 2; }
slakshna_root="${M0_SLAKSHNA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
repo_root="${M0_REPO_ROOT:-}"
if [[ -z "${repo_root}" && -n "${slakshna_root}" ]]; then
    repo_root="$(cd "${slakshna_root}/.." && pwd)"
fi
train_script="${repo_root}/monash_exps/scripts/m0/baseline_runs/train_t5.sh"
[[ -n "${slakshna_root}" && -f "${train_script}" ]] || {
    echo "Cannot locate Slakshna root from M0_SLAKSHNA_ROOT/SLURM_SUBMIT_DIR: ${slakshna_root:-unset}" >&2
    exit 1
}
cd "${slakshna_root}"
exec bash "${train_script}" "${M0_RUN}"
