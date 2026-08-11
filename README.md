# Slakshna Experiment Test Findings

This document records issues found while testing Slakshna as an unmodified
third-party dependency on the FIT Slurm cluster. It is intended as actionable
feedback for the Slakshna maintainers. Experiment-side workarounds live under
`monash_exps/`; the `Slakshna` submodule has not been changed.

## Tested baselines

- Phase 1--7 acceptance revision: `a3112cf7aa11316d47c6bdf749a45c7071b5f9f3`
- Phase 8 candidate revision: `f09eff9a73ae8f1080d4f0b43114b3a8aa5e99bb`
- Installed Bhaskera distribution version: `2.2.0`
- Python: 3.11.13
- PyTorch: 2.9.0+cu128
- Ray: 2.56.1
- Transformers: 4.57.6
- Hardware exercised: one NVIDIA A40, one node with two NVIDIA A100 40 GB
  GPUs, and two M3 nodes with one NVIDIA A100-SXM4-80GB GPU each
- CUDA toolkit baseline: 12.8

Phase 0 environment/API checks and Phase 1A single-GPU SFT passed. Phase 1B
two-GPU DDP SFT passed after disabling the Ray metrics tracker. Phase 1C
correctly failed its independent resume verifier because the upstream resume
path restarted training from step zero.

On M3, the experiment-side resume workaround subsequently passed the complete
two-A100 sequence: Run 1 trained through step 20 and Run 2 restored the exact
in-memory LoRA state in a new process before emitting only steps 21 through 30.
This validates the workaround; it does not remove the upstream defect.
Phase 2 also passed a one-A100 dense LoRA-delta round trip, including an exact
fresh-process warm start and one continuation optimizer step.
Phase 3 subsequently passed the real single-node Rust/Python boundary: the
unmodified Slakshna binary triggered A100 training and recorded the verified
canonical update in its API-visible history.
Phase 4 then passed a two-client, two-round local FedAvg simulation with four
fresh optimizer runs and exact aggregate reconstruction.
Phase 5 passed two real Slakshna peers on one M3 node: both completed two
training rounds, exchanged canonical deltas through Iroh Gossip, reconstructed
an identical aggregate, converged their histories/leaderboards, and preserved
transport identity plus known-peer state across restart. The run also confirmed
the history-persistence and review-version defects documented below.
Phase 6 passed the same lifecycle across two distinct M3 compute nodes using
only direct non-loopback QUIC/Gossip. It proved cross-node convergence and
process cleanup, but also exposed that the persisted known-peer cache cannot by
itself restore a dialable address after restart (defect 14).
Phase 7 passed five federated rounds on both one-node/two-GPU and true two-node
placements. Each peer completed 50 local epochs and 3,600 optimizer steps;
both peers reconstructed identical G1--G5 adapters, and macro held-out negative
log likelihood improved by 26.96%. The run also quantified the fixed epoch
barrier utilization issue documented as defect 15.

The Phase 8 candidate was subsequently accepted by source audit, locked Cargo
build, three Rust identity tests, three Python delta-transport tests, CPU/GPU
environment preflights, and a four-step single-A100 LoRA training run. The
installed Bhaskera Python tree exactly matched the pinned source snapshot, and
the training run produced a complete 112-tensor adapter checkpoint with finite
loss. Source review confirms fixes for the scheduler-assigned GPU visibility
defect and the DDP epoch-metrics collective ordering defect. Other findings
below remain applicable unless explicitly marked otherwise.

The experiment-side Phase 8 bridge now removes Phase 7's shared-filesystem
assumption. Slakshna transports an opaque, versioned dense-delta envelope whose
sender EndpointId, site role, round number, base-state hash, file hash, tensor
hash, shape cardinality, and bounded byte counts are checked before FedAvg.
This is an external compatibility layer, not an upstream fix: Slakshna's
ML-engine failure-handling limitation in defect 9 remains particularly
important for cross-cluster operation, so Phase 8 uses independent site and
paired audit documents rather than treating node liveness or exit status as
acceptance evidence.

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

Status at `f09eff9`: fixed by source audit; all ranks now execute the
epoch-metrics tracker call. Runtime DDP revalidation remains part of the
upgraded environment acceptance rather than the Phase 8 single-GPU path.

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

### 7. Federated adapter injection can be overwritten by DCP resume

Severity: critical for federated rounds; identified by source audit.

`Slakshna/ml_engine.py` writes an aggregated LoRA state to
`adapter_model.safetensors` and to a separate `*_base_lora.pth` file. The next
Bhaskera model build can load the latter through `lora.resume_path`, but the
training loop then resumes the existing DCP checkpoint, whose model shards
still contain the pre-aggregation parameters. DCP can therefore overwrite the
aggregated adapter before training continues. Replacing only
`adapter_model.safetensors` also leaves that checkpoint internally inconsistent
with its DCP model state.

Suggested fix: define federated aggregation as an explicit warm-start boundary.
Load the aggregated adapter into a fresh training state with a new optimizer,
or update model and optimizer checkpoint state together. Do not mutate one
representation inside an otherwise completed checkpoint. The Phase 2 external
experiment uses the fresh-state approach while leaving the submodule unchanged.

### 8. The stock ML engine overrides the scheduler-assigned GPU visibility

Severity: critical on Slurm and other resource-isolated schedulers; identified
by source audit before Phase 3 execution.

Status at `f09eff9`: fixed by source audit. The Python ML engine no longer
replaces `CUDA_VISIBLE_DEVICES` with the hard-coded `1,2,3` set; GPU visibility
is inherited from the Rust child environment and resolved as logical device
zero inside that visible set. Experiment launchers still align Rust's
configured `gpu_id` with scheduler-visible logical device indices.

The Rust node pins its Python child using the configured `gpu_id`, but
`Slakshna/ml_engine.py` later launches Bhaskera with a new environment that
unconditionally sets `CUDA_VISIBLE_DEVICES="1,2,3"`. This discards the GPU
selection made by Rust and can request devices outside a one-GPU allocation.
It can also make CUDA unavailable when the allocation exposes only logical
device 0.

Suggested fix: never rewrite `CUDA_VISIBLE_DEVICES` inside the ML engine.
Treat the inherited visible device set as authoritative, validate its size,
and let the scheduler or parent launcher perform physical GPU selection. The
Phase 3 bridge requires exactly one inherited visible GPU.

### 9. ML-engine contract failures are not surfaced as node failures

Severity: high for operational correctness; identified by source audit.

The Rust training loop accepts only the last stdout line from
`python ml_engine.py`. If the child exits successfully but that line does not
deserialize as `MLEngineOutput`, the parse failure is ignored. The loop then
tries to record a placeholder update with an empty record hash; history rejects
it, but the node continues running. Successful-child stderr is also discarded,
which hides the Python training log from the node operator.

Suggested fix: treat spawn, exit, UTF-8, empty-output, JSON, and required-field
failures as explicit round failures with structured logs and status/API state.
Make the Python executable and ML-engine path configurable instead of relying
on the names `python` and `ml_engine.py` in the current working directory. The
Phase 3 verifier rejects missing updates and the `error_hash` placeholder.

### 10. Aggregation weights are not renormalized after peers are unavailable or rejected

Severity: high for federated update correctness; identified by source audit
while defining the Phase 4 aggregation oracle.

`Slakshna/ml_engine.py` computes softmax trust weights across every configured
node before it knows which peer deltas are present and valid. Aggregation later
iterates only over `available_deltas`, but continues using the original weights.
If a peer update is missing, malformed, non-finite, or rejected by the norm
limit, the retained weights sum to less than one. The aggregate is therefore
silently shrunk toward a zero update instead of being a weighted average of the
accepted participants.

Suggested fix: first validate the complete key/shape/dtype schema of every
candidate update, form the accepted participant set, and renormalize its
non-negative weights to sum to exactly one. Log both exclusions and final
effective weights. Phase 4 uses sample-weighted FedAvg with a strict sum-one
oracle; trust weighting remains deferred until the real-peer phase.

### 11. Model-update history is not persisted across node restarts

Severity: high for federated durability; identified by source audit while
defining the Phase 5 recovery contract.

`UpdateHistory::new()` always constructs an empty in-memory `HashMap`. Although
RocksDB persists the node keypair, round field, and known-peer cache, locally
created and remotely received `ModelUpdate`/`PeerReview` records are not written
to it and are not restored. Restarting a node therefore loses its hash-chain
history, trust evidence, latest peer deltas, and API-visible audit trail.

Suggested fix: atomically persist every accepted record keyed by origin and
sequence, validate each chain while loading, and restore history before network
or training tasks start. Phase 5 separately checks the state that is currently
durable (EndpointId and known peers) and records the empty post-restart update
API as an upstream limitation.

### 12. Network update signatures are placeholders and are never verified

Severity: critical for adversarial or cross-institution deployments; identified
by source audit before Phase 5 P2P execution.

The training loop writes strings such as `node_signature_<epoch>` and
`review_sig_<epoch>` into `UpdateRecord.signature`. `record_update()` verifies
only the unkeyed SHA-256 content hash; it does not verify an Ed25519 signature
or bind `record.node_id` to the authenticated Iroh sender. A connected peer can
therefore claim another participant's logical identity or fabricate reviews
with internally consistent hashes.

Suggested fix: sign a domain-separated canonical encoding of every record with
the persisted federation key, verify it before history insertion, and require
the claimed node identity to match the authenticated transport identity. Phase
5 validates content/hash-chain integrity only and does not claim Byzantine
authentication.

### 13. Peer reviews can bind to a newer update than the ML engine evaluated

Severity: high for version-specific trust; identified while reviewing the
concurrent two-peer Phase 5 lifecycle.

The Python child evaluates the peer deltas that Rust staged before local
training. Only after that child returns does Rust choose the review target by
asking history for the peer's latest update. If the remote peer broadcasts its
next-round update during the local training window, the review can therefore
name that newer record even though the score was computed from the older
payload.

Suggested fix: pass the exact evaluated record hash through the ML-engine
contract and construct the review against that immutable hash. Phase 5 requires
every review target to be a genuine recorded update from the claimed peer and
reports the observed target round, but it does not claim version-specific
review binding until upstream carries this provenance explicitly.

### 14. Known-peer persistence loses direct addressing information

Severity: high for closed-network and cross-institution recovery; confirmed by
the Phase 6 two-node restart test.

`remember_peer()` persists only `peer:<EndpointId> -> last_seen` in RocksDB,
and `known_peers()` consequently restores endpoint IDs without an
`EndpointAddr`. Configured direct seeds are inserted into the in-memory lookup,
but remembered peers are not restored with direct IPs or relay URLs. After
Phase 6 restarted Peer A, its API correctly listed Peer B as known while the
transport logged `No addressing information available` when trying to dial it.

Suggested fix: persist the latest validated `EndpointAddr` (including direct
and relay addresses) with each peer, restore it into the memory lookup before
joining Gossip, and refresh it after successful connections. Phase 6 therefore
claims identity/cache durability, not autonomous network reconnection after a
peer restart.

### 15. The mid-epoch barrier waits until its deadline after all peers are ready

Severity: medium for accelerator utilization; confirmed by the Phase 7
one-node and two-node convergence runs.

The training loop detects when `expected_peers` have been collected and logs
that it completed early, but it then unconditionally sleeps until
`epoch_start + sync_deadline_secs`. With a 600-second epoch and a 570-second
deadline, Phase 7 local training and artifact production took approximately
186--207 seconds per peer, followed by roughly six idle minutes before trust
ranking and the next boundary. The underlying 720-step GPU training interval
was only 145--157 seconds, yielding approximately 24--26% GPU duty cycle.

Suggested fix: make the barrier track current-epoch ModelUpdates and reviews,
then advance as soon as every expected participant has supplied the required
records. Retain the deadline only as a timeout for missing or slow peers. A
fixed clock-aligned mode may remain useful, but it should be explicit rather
than imposing idle time after the completion condition has already passed.

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
