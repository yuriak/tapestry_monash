# Slakshna Phase 0/1 Test Findings

This document records issues found while testing Slakshna as an unmodified
third-party dependency on the FIT Slurm cluster. It is intended as actionable
feedback for the Slakshna maintainers. Experiment-side workarounds live under
`monash_exps/`; the `Slakshna` submodule has not been changed.

## Tested baseline

- Slakshna revision: `a3112cf7aa11316d47c6bdf749a45c7071b5f9f3`
- Installed Bhaskera distribution version: `2.2.0`
- Python: 3.11.13
- PyTorch: 2.9.0+cu128
- Ray: 2.56.1
- Transformers: 4.57.6
- Hardware exercised: one NVIDIA A40 and one node with two NVIDIA A100 GPUs
- CUDA toolkit baseline: 12.8

Phase 0 environment/API checks and Phase 1A single-GPU SFT passed. Phase 1B
two-GPU DDP SFT passed after disabling the Ray metrics tracker. Phase 1C
correctly failed its independent resume verifier because the upstream resume
path restarted training from step zero.

## Confirmed runtime defects

### 1. The trainer checkpoint wrapper passes the wrong directory to the DCP loader

Severity: critical for checkpoint resume.

`bhaskera.trainer.checkpointing.maybe_resume()` scans the configured checkpoint
root, selects its latest `step_*` child, and then passes that child to
`bhaskera.distributed.load_checkpoint()`. The exported distributed loader is
itself `bhaskera.distributed.checkpoint.maybe_resume()`, which expects the
checkpoint **root** and scans its children for `step_*` directories. It
therefore searches one directory too deep and silently returns `(0, {})`.

Observed in Slurm job 267528:

```text
Resuming from .../checkpoints/step_0000020
No valid checkpoints found (missing .complete sentinel). Starting from step 0.
```

The continuation then emitted steps 1 through 30 instead of steps 21 through
30. The expected behavior is to restore model, optimizer, metadata, and cursor
from step 20 and emit step 21 next.

Suggested fix: make the trainer wrapper call the distributed loader once with
the original checkpoint root, or add a separate lower-level function that
loads one exact checkpoint path. A discovered checkpoint that cannot be loaded
should raise an error instead of silently becoming a fresh run.

Relevant files:

- `Slakshna/Bhaskera/src/bhaskera/trainer/checkpointing.py`
- `Slakshna/Bhaskera/src/bhaskera/distributed/checkpoint.py`

### 2. RayMetricsLogger can deadlock DDP training

Severity: high for multi-GPU runs using the Ray tracker.

Bhaskera invokes some tracker calls only from rank zero, including epoch-level
metrics. `RayMetricsLogger` translates a tracker call into
`ray.train.report()`, which is collective across Ray Train workers. Rank zero
therefore waits inside Ray for a report that other ranks never issue.

In Slurm job 267527, both A100 workers formed a world of size two and completed
four synchronized optimizer steps. The job then stalled at the first
rank-zero-only epoch report and never reached checkpointing. Ray repeatedly
printed:

```text
Failed to query Ray Train Controller actor state. State API may be temporarily
unavailable. Continuing to monitor.
```

Suggested fix: every `ray.train.report()` call must occur in the same order on
every worker, or Ray reporting must be centralized outside the distributed
worker loop. Rank-zero-only logging should use a non-collective logger.

Our external launcher rejects the Ray tracker for world sizes greater than one
and verifies structured rank-zero stdout instead.

### 3. The global rank is not forwarded to the collective checkpoint writer

Severity: high; identified by source audit and avoided before it caused data
loss.

`bhaskera.trainer.checkpointing.save_and_prune()` receives `rank`, but its call
to `save_checkpoint()` omits `rank=rank`. The distributed writer consequently
uses its default `rank=0` on every process. All ranks may then execute rank-zero
filesystem operations such as rename, sentinel creation, adapter export, and
checkpoint pruning against the same paths.

Suggested fix: forward both `rank=rank` and the configured retention count to
`save_checkpoint()`. The experiment currently injects the actual Ray global
rank at the external boundary.

### 4. Scheduler state is not checkpointed or restored

Severity: medium for faithful continuation.

The DCP payload contains model and optimizer state only. The training loop
constructs a new scheduler before loading the checkpoint and neither saves nor
restores scheduler state. A resumed run can therefore use an LR schedule that
does not correspond to its restored global step.

Suggested fix: include `scheduler.state_dict()` in the checkpoint payload and
restore it with the model and optimizer. The experiment temporarily advances
the deterministic scheduler to the recorded global step.

### 5. Tokenizer cache metadata reports a different Bhaskera version

Severity: low, but it weakens cache provenance.

The installed project declares Bhaskera 2.2.0, while
`bhaskera/data/tokenize.py` hard-codes `_BHASKERA_VERSION = "2.3.0"`. Cache
metadata produced by this tested revision therefore does not identify the
installed distribution accurately.

Suggested fix: derive the value from package metadata or a single package
version constant.

### 6. Unknown configuration keys are silently ignored

Severity: medium configuration-safety risk; identified by source audit.

`Config.from_dict()` manually reads known fields with dictionary `.get()`
calls, but it does not reject unrecognized top-level or nested keys. A typo in
a training-critical YAML field can therefore be accepted while Bhaskera uses a
default value, leaving the user with a valid-looking but unintended run.

Suggested fix: validate input against a strict schema and report the complete
path of every unknown field. If backward compatibility requires permissive
loading, provide it only as an explicit opt-in mode.

## Packaging and cluster-integration findings

These findings affected Phase 0/1 portability, but are separated from the
confirmed training defects above.

### In-tree Python builds leave generated files in the third-party checkout

Building/installing Bhaskera directly from its source tree created
`Bhaskera/build/` and `Bhaskera/src/bhaskera.egg-info/`, making the submodule
dirty. The experiment now builds a non-editable wheel from a `git archive`
snapshot. An upstream `src`-layout/build configuration that keeps generated
metadata out of the checkout would make clean submodule consumption easier.

### Resource discovery should honor the scheduler allocation

Code paths that use host-wide `os.cpu_count()` can advertise or launch more
work than Slurm assigned to the job. The experiment derives CPU counts from
`SLURM_CPUS_PER_TASK`/`SLURM_CPUS_ON_NODE` and GPU counts from the visible
CUDA devices. Slakshna launchers should prefer scheduler/cgroup-visible
resources over physical host totals.

### A launcher should not terminate unrelated Ray sessions

The stock launch path invokes `ray stop --force`. On a shared cluster this can
affect another same-user Ray workload on the node. The experiment initializes
and shuts down only its own local Ray runtime.

## Cluster-specific failures that are not Slakshna defects

- Slurm executes a spooled copy of a batch script, so resolving the repository
  from `BASH_SOURCE[0]` was incorrect in our original wrapper. It now uses
  `SLURM_SUBMIT_DIR`.
- Long artifact paths exceeded the 107-byte AF_UNIX socket path limit during
  Ray startup. Ray temporary sockets now use a short allocation-local `/tmp`
  directory; durable evidence remains in project storage.
- An early Rust build failed to find `stddef.h` while compiling `zstd-sys`.
  This was a FIT compiler/sysroot configuration issue and was resolved by the
  experiment environment setup, not by changing Slakshna.

## External validation policy

The experiment does not accept a run merely because the process exits zero.
It independently checks worker/rank/GPU mapping, synchronized LoRA hashes,
finite metrics, expected step boundaries, complete DCP state, adapter content,
and equality between the Run 1 final LoRA state and the in-memory state loaded
at the start of Run 2. This policy exposed the silent restart in issue 1.
