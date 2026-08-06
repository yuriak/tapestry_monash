#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
cd "${workspace_root}"

# The Slurm wrapper normally activates this already; sourcing is idempotent and
# also makes direct allocation-side debugging behave the same way.
# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

run_id="${1:-$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID:-unknown}_phase1bc}"
artifact_root="${experiment_root}/artifacts/phase1/${run_id}"
run1_dir="${artifact_root}/phase1b-ddp-overfit"
run2_dir="${artifact_root}/phase1c-resume"
checkpoint_dir="${artifact_root}/checkpoints"
mkdir -p "${run1_dir}" "${run2_dir}"

python "${experiment_root}/src/phase1/prepare_experiment.py" \
    --template "${experiment_root}/configs/phase1/phase1bc_run1.yaml" \
    --run-dir "${run1_dir}" \
    --checkpoint-dir "${checkpoint_dir}" \
    --run-name "${run_id}-run1"

python "${experiment_root}/src/phase1/launch_training.py" \
    --config "${run1_dir}/resolved-config.yaml" \
    --num-workers 2 \
    --run-dir "${run1_dir}" \
    2>&1 | tee "${run1_dir}/train.log"

python "${experiment_root}/src/phase1/verify_training.py" \
    --mode phase1b \
    --run-dir "${run1_dir}" \
    --expected-workers 2 \
    --expected-final-step 20 \
    --minimum-loss-drop 0.20

resume_step="$({ python - "${checkpoint_dir}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
paths = sorted(p for p in root.glob("step_*") if (p / ".complete").is_file())
if not paths:
    raise SystemExit("Run 1 produced no completed checkpoint")
print(json.loads((paths[-1] / "meta.json").read_text())["step"])
PY
} )"

python "${experiment_root}/src/phase1/prepare_experiment.py" \
    --template "${experiment_root}/configs/phase1/phase1bc_run2.yaml" \
    --run-dir "${run2_dir}" \
    --checkpoint-dir "${checkpoint_dir}" \
    --run-name "${run_id}-run2"

python "${experiment_root}/src/phase1/launch_training.py" \
    --config "${run2_dir}/resolved-config.yaml" \
    --num-workers 2 \
    --run-dir "${run2_dir}" \
    2>&1 | tee "${run2_dir}/train.log"

python "${experiment_root}/src/phase1/verify_training.py" \
    --mode phase1c \
    --run-dir "${run2_dir}" \
    --expected-workers 2 \
    --expected-resume-step "${resume_step}" \
    --expected-final-step 30 \
    --prior-run-dir "${run1_dir}"

echo "Phase 1B/1C artifact: ${artifact_root}"
