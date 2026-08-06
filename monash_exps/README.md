# Monash Cluster Experiments

This directory contains portable experiment code for the pinned `../Slakshna`
third-party submodule. Shared environment definitions, configs, Python code,
and phase runners are cluster-independent. Module loading and Slurm resource
selection are isolated at the shell boundary.

The code currently supports FIT and provides an M3 adapter whose module labels
are supplied by the user after inspecting `module avail` on M3. Neither adapter
modifies the Slakshna submodule.

## Versioned and local state

Commit these inputs:

- `environment/{pyproject.toml,uv.lock,.python-version,bhaskera-source-revision.txt}`;
- `configs/`, `scripts/`, and `src/`;
- this README and experiment-side patches added later.

Do not transfer `.runtime/`, `artifacts/`, `slurm_logs/`, model/data caches, Ray
state, virtual environments, or generated datasets. A virtual environment is
not relocatable: every clone or cluster must run the frozen sync locally.

## Cluster activation

All experiment runners source `scripts/cluster/activate.sh`. Selection order is:

1. explicit `SLAKSHNA_CLUSTER=fit|m3|generic`;
2. `SLURM_CLUSTER_NAME` when recognizable;
3. FIT's shared Spack tree as a convenience fallback;
4. `generic`, which assumes required system modules are already loaded.

FIT has pinned module labels in `scripts/cluster/fit.sh`:

```bash
export SLAKSHNA_CLUSTER=fit
source monash_exps/scripts/cluster/activate.sh
```

M3 module labels are deliberately not guessed. Set the labels available on the
target M3 environment, or preload the modules and leave the variables empty:

```bash
export SLAKSHNA_CLUSTER=m3
export SLAKSHNA_M3_COMPILER_MODULE='<M3 compiler module>'
export SLAKSHNA_M3_CUDA_MODULE='<M3 CUDA 12.8+ module>'
export SLAKSHNA_M3_GIT_MODULE='<M3 Git module>'
source monash_exps/scripts/cluster/activate.sh
```

The common layer redirects uv, Hugging Face, Torch, and XDG caches under
`monash_exps/.runtime/` and activates the selected project-local venv.

## Phase 0: build and verify the environment

Start from the repository root and activate the target cluster adapter. It is
safe to activate before the venv exists.

### 1. Install project-local uv

```bash
bash monash_exps/scripts/environment/01_install_uv.sh
```

This uses the official uv installer and does not edit shell profiles or Conda.

### 2. Resolve the environment

```bash
bash monash_exps/scripts/environment/02_lock_environment.sh
```

The script installs project-managed Python 3.11.13, snapshots tracked Bhaskera
files with `git archive`, and updates `environment/uv.lock`. Bhaskera is built
from the ignored snapshot rather than inside the submodule. Review changes to
the lockfile and `environment/bhaskera-source-revision.txt` before continuing.

### 3. Create a local venv from the lock

```bash
bash monash_exps/scripts/environment/03_sync_environment.sh primary
source monash_exps/scripts/cluster/activate.sh

which python
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

Expected Python is 3.11.13 and the PyTorch CUDA build must be at least 12.8.

### 4. Run the CPU/API preflight

```bash
bash monash_exps/scripts/environment/05_phase0_preflight.sh cpu
```

This checks dependencies, required APIs, Bhaskera config resolution, the exact
submodule revision, and source-tree cleanliness without starting training.

### 5. Run a GPU preflight in Slurm

The checked-in wrappers define node/task, GPU count, CPU, memory, time, and
project-local log paths. Cluster-specific partition, QoS, account, reservation,
and exclusion flags belong on the `sbatch` command line.

FIT example:

```bash
mkdir -p monash_exps/slurm_logs

sbatch --export=ALL,SLAKSHNA_CLUSTER=fit \
  --partition=A100 --qos=gpua100 --exclude=node[14] \
  -J Phase0GPU \
  monash_exps/scripts/slurm/submit_job_2gpu.sh \
  monash_exps/scripts/environment/05_phase0_preflight.sh gpu
```

On M3, replace only the scheduler options and export
`SLAKSHNA_CLUSTER=m3` plus any required M3 module-label variables. Submit from
the repository root because Slurm resolves log paths before starting the batch
shell.

### 6. Rebuild independently

```bash
bash monash_exps/scripts/environment/03_sync_environment.sh rebuild

export SLAKSHNA_UV_ENVIRONMENT="$PWD/monash_exps/.runtime/venvs/rebuild"
source monash_exps/scripts/cluster/activate.sh
bash monash_exps/scripts/environment/05_phase0_preflight.sh cpu
```

Unset the override to return to the primary profile.

```bash
unset SLAKSHNA_UV_ENVIRONMENT
source monash_exps/scripts/cluster/activate.sh
```

## Phase 1: single-node SFT

Phase 1 recreates two immutable public inputs in the local project cache:

- `Qwen/Qwen3-0.6B` at revision
  `c1899de289a04d12100db370d81485cdf75e47ca`;
- `HuggingFaceTB/everyday-conversations-llama3.1-2k` at revision
  `14f543216b9ba42b6b951dc5bd199460d193b162`.

The preparation script deterministically selects eight conversations and
renders them using Qwen's chat template. Generated JSONL and tokenized data are
local runtime products and must be regenerated on each cluster.

### Phase 1A: interactive single-GPU smoke test

Run inside an interactive allocation exposing exactly one CUDA GPU:

```bash
bash monash_exps/scripts/phase1/run_phase1a_single_gpu.sh
```

The runner does not require a particular GPU model. It verifies four BF16 LoRA
steps, finite metrics, a changed adapter, and complete checkpoint state.

### Phase 1B/1C: two-A100 DDP and resume

FIT example:

```bash
mkdir -p monash_exps/slurm_logs

sbatch --export=ALL,SLAKSHNA_CLUSTER=fit \
  --partition=A100 --qos=gpua100 --exclude=node[14] \
  -J Phase1BC \
  monash_exps/scripts/slurm/submit_job_2gpu.sh \
  monash_exps/scripts/phase1/run_phase1bc_2a100.sh
```

Run 1 performs 20 optimizer steps with two explicit workers. Run 2 starts a
new Python/Ray process, restores model/optimizer/cursor state, reconstructs the
missing scheduler progress, and continues through step 30. The verifier checks
rank/GPU mapping, synchronized LoRA hashes, loss reduction, exact resume state,
metric step boundaries, and complete checkpoint files.

### One-A100 fallback

When two A100s are unavailable, checkpoint creation and fresh-process resume
can be tested with one worker. It does not replace DDP synchronization evidence.

```bash
sbatch --export=ALL,SLAKSHNA_CLUSTER=fit \
  --partition=A100 --qos=gpua100 \
  -J Phase1BC1A100 \
  monash_exps/scripts/slurm/submit_job_1gpu.sh \
  monash_exps/scripts/phase1/run_phase1bc_1a100.sh
```

The one-worker configs use batch size 2, preserving the two-worker experiment's
global batch size and four optimizer steps per epoch.

## Upstream compatibility handling

The external launcher intentionally:

- limits Ray to scheduler-visible CPU/GPU resources;
- avoids Bhaskera's process-wide `ray stop --force` path;
- forwards the actual global rank during collective checkpoint finalization;
- bypasses Bhaskera 2.2.0's nested checkpoint-resume path mismatch;
- reconstructs scheduler progress because scheduler state is not checkpointed;
- rejects the Ray metrics tracker for multi-worker training because asymmetric
  `ray.train.report()` calls deadlock.

The independent verifier is fail-closed. A discovered checkpoint must restore
the expected in-memory LoRA state and step; a zero exit code alone is not
accepted as evidence.

## Source policy

`../Slakshna` must remain clean. Do not run setup commands that build inside the
submodule. If an upstream change becomes unavoidable, place a minimal,
revision-specific patch under `monash_exps/patches/` and apply it only to a
staged snapshot or disposable worktree.
