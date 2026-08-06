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

## Phase 2: minimal LoRA update round trip

Phase 2 deliberately tests only the state transition needed by later
federated stages. On one visible GPU it:

1. saves the initial LoRA adapter and trains four local steps;
2. writes the FP32 parameter delta as `delta.safetensors`;
3. applies that delta to the initial adapter in a separate process;
4. checks that the reconstructed state matches the trained adapter within
   `1e-7` maximum absolute error; and
5. loads the applied adapter in another fresh training process and completes
   one optimizer step with finite metrics.

This phase does not yet test multiple clients, FedAvg, sparsification,
differential privacy, error feedback, or P2P transport. The canonical update is
the safetensors file. `applied_adapter.pth` is only a local compatibility bridge
for Bhaskera 2.2.0's existing `lora.resume_path` loader.

### Interactive single-GPU run

Start from an activated allocation and expose exactly one GPU:

```bash
export SLAKSHNA_CLUSTER=m3
export SLAKSHNA_M3_COMPILER_MODULE=gcc/10.2.0
export SLAKSHNA_M3_CUDA_MODULE=cuda/12.8
export SLAKSHNA_UV_ENVIRONMENT="$PWD/monash_exps/.runtime/venvs/primary"
export CUDA_VISIBLE_DEVICES=0
source monash_exps/scripts/cluster/activate.sh

bash monash_exps/scripts/phase2/run_phase2_adapter_delta.sh
```

The final verdict is written to
`artifacts/phase2/<run-id>/verification-summary.json`. A passing run must report
non-zero delta values, a successful fresh-process reconstruction, and one
successful continuation step.

### Accepted M3 result

Phase 2 passed on 2026-08-06 in M3 Slurm job `58796699` using one
NVIDIA A100-SXM4-40GB. Run
`20260806_173806_58796699_m3_phase2` produced 112 delta tensors containing
1,146,880 non-zero values. Applying the update reproduced the trained adapter
with maximum absolute error `2.91e-11` (limit `1e-7`); the fresh process loaded
the applied adapter with zero error and completed checkpoint step 1. The final
verdict was `PHASE2 PASSED`.

### M3 batch submission

```bash
mkdir -p monash_exps/slurm_logs

sbatch \
  --export=ALL,SLAKSHNA_CLUSTER=m3,SLAKSHNA_M3_COMPILER_MODULE=gcc/10.2.0,SLAKSHNA_M3_CUDA_MODULE=cuda/12.8 \
  --partition=fit --qos=fitq --account=mg61 --gres=gpu:A100:1 \
  -J Phase2 \
  monash_exps/scripts/slurm/submit_job_1gpu.sh \
  monash_exps/scripts/phase2/run_phase2_adapter_delta.sh
```

## Phase 3: one real Slakshna node

Phase 3 validates one deliberately small integration boundary: the unmodified
Slakshna Rust binary must start the experiment ML bridge, the bridge must train
the Phase 1 LoRA workload on one scheduler-visible GPU, and Rust must record the
returned canonical delta as a valid `ModelUpdate`. There are no peers, FedAvg,
trust updates, sparsification, differential privacy, or P2P exchange in this
phase.

The runner uses an isolated artifact directory as the Rust process working
directory. Its local file named `ml_engine.py` is a copy of the experiment
bridge, because the pinned Rust revision hard-codes that relative command. The
Slakshna submodule remains unchanged and Cargo writes its build output under
the ignored `.runtime/` tree.

### Rust toolchain bootstrap

M3 does not currently advertise a Rust module. Install a workspace-local
toolchain once if `cargo` is unavailable:

```bash
bash monash_exps/scripts/environment/06_install_rust.sh
```

The script uses the official rustup TLS installer and places `cargo` and
`rustup` under `monash_exps/.runtime/`; it does not modify the login shell.
The exact resolved `rustc`, `cargo`, toolchain, Cargo lock, and Slakshna revision
are captured by each run. `tmp.sh` performs this bootstrap automatically when
needed.

The M3 runner keeps GCC 10 loaded for CUDA compatibility, but removes its old
`libstdc++` from `LD_LIBRARY_PATH` only at the Rust boundary and builds Rust C/C++
dependencies with the host GCC 11. This is required because M3's LLVM 21
`libclang` needs `GLIBCXX_3.4.29`; the training process still inherits CUDA
12.8 and the scheduler-selected GPU.

### Interactive one-A100 run

```bash
export SLAKSHNA_CLUSTER=m3
export SLAKSHNA_M3_COMPILER_MODULE=gcc/10.2.0
export SLAKSHNA_M3_CUDA_MODULE=cuda/12.8
export SLAKSHNA_UV_ENVIRONMENT="$PWD/monash_exps/.runtime/venvs/primary"
export CUDA_VISIBLE_DEVICES=0
source monash_exps/scripts/cluster/activate.sh

bash monash_exps/scripts/phase3/run_phase3_single_node.sh
```

The runner builds the locked Rust dependency graph, chooses loopback ports,
starts a real Slakshna node, waits for exactly one update through its HTTP API,
then terminates the perpetual node and verifies the complete lifecycle. It
writes `artifacts/phase3/<run-id>/verification-summary.json` and must end with
`PHASE3 PASSED`.

Acceptance requires all of the following:

- Rust logs the live node, GPU pin, and successful ML-engine completion;
- the Rust-triggered training reaches checkpoint step 4 on exactly one GPU;
- the base64/zlib payload decodes byte-for-byte to `delta.safetensors`;
- its SHA-256, tensor structure, finite values, and non-zero count are valid;
- the API contains exactly one non-placeholder `ModelUpdate` with a valid
  Rust history hash; and
- the pinned Slakshna source remains clean.

### Accepted M3 result

Phase 3 passed on 2026-08-06 in M3 Slurm job `58796699` using one
NVIDIA A100-SXM4-40GB. Run `20260806_185401_58796699_m3_phase3` built the
pinned Rust source, started a real node, triggered checkpoint step 4, and
recorded exactly one API-visible `ModelUpdate`. The update contained 112
tensors and 1,146,880 non-zero values; its 4,602,216 canonical bytes compressed
to 4,013,752 bytes before base64 transport. The verifier matched the payload,
SHA-256, and Rust history-chain hash and reported `PHASE3 PASSED`. The runner
then stopped the node with no Slakshna or ML-engine process left behind.

### M3 batch submission

```bash
mkdir -p monash_exps/slurm_logs

sbatch \
  --export=ALL,SLAKSHNA_CLUSTER=m3,SLAKSHNA_M3_COMPILER_MODULE=gcc/10.2.0,SLAKSHNA_M3_CUDA_MODULE=cuda/12.8 \
  --partition=fit --qos=fitq --account=mg61 --gres=gpu:A100:1 \
  -J Phase3 \
  monash_exps/scripts/slurm/submit_job_1gpu.sh \
  monash_exps/scripts/phase3/run_phase3_single_node.sh
```

## Phase 4: two-client, two-round local FedAvg

Phase 4 tests the federated algorithm without Rust or network transport. Two
logical clients run sequentially on one visible GPU for two optimizer steps per
round. Client A receives source rows `[0, 2, 4, 6]`; client B receives
`[1, 3, 5, 7]`. The shards are deterministic, disjoint, and together cover the
same eight pinned Phase 1 rows. They are derived from the local materialized
JSONL, so shard creation does not query the dataset service again.

Round 1 client A captures the common random LoRA initialization `G0`; client B
loads that exact state. Each client produces a dense FP32 delta, and the runner
computes sample-weighted FedAvg. Both shards contain four samples, so:

```text
delta_global = 0.5 * delta_client_a + 0.5 * delta_client_b
G1           = G0 + delta_global
```

Both Round 2 clients then warm-start from `G1` in fresh training/checkpoint
directories with new optimizer state. A second identical aggregation produces
`G2`. Compression, DP, trust weighting, Rust peers, and P2P are intentionally
absent.

### Interactive one-A100 run

```bash
export SLAKSHNA_CLUSTER=m3
export SLAKSHNA_M3_COMPILER_MODULE=gcc/10.2.0
export SLAKSHNA_M3_CUDA_MODULE=cuda/12.8
export SLAKSHNA_UV_ENVIRONMENT="$PWD/monash_exps/.runtime/venvs/primary"
export CUDA_VISIBLE_DEVICES=0
source monash_exps/scripts/cluster/activate.sh

bash monash_exps/scripts/phase4/run_phase4_local_fedavg.sh
```

The final verdict is written to
`artifacts/phase4/<run-id>/verification-summary.json`. Acceptance requires four
successful step-2 checkpoints, exact common starts within `1e-7`, non-zero and
different client deltas, correct sum-one weights, exact FedAvg reconstruction,
Round 2 consumption of `G1`, four fresh optimizer states, and clean Slakshna
source. The result must end with `PHASE4 PASSED`.

### Accepted M3 result

Phase 4 passed on 2026-08-06 in M3 Slurm job `58796699` using one
NVIDIA A100-SXM4-40GB. Run `20260806_191316_58796699_m3_phase4` completed all
four step-2 local trainings with DCP resume step 0. Both clients started each
round from the exact same adapter, and Round 2 consumed Round 1's `G1` with
zero error. Client deltas were non-zero and distinct in both rounds; both
FedAvg delta and global-adapter reconstruction errors were exactly zero.
`G0` to `G2` changed by maximum absolute value `0.00150012935`. The verifier
reported `PHASE4 PASSED`, no Ray/training process remained, and the Slakshna
source stayed clean.

### M3 batch submission

```bash
mkdir -p monash_exps/slurm_logs

sbatch \
  --export=ALL,SLAKSHNA_CLUSTER=m3,SLAKSHNA_M3_COMPILER_MODULE=gcc/10.2.0,SLAKSHNA_M3_CUDA_MODULE=cuda/12.8 \
  --partition=fit --qos=fitq --account=mg61 --gres=gpu:A100:1 \
  -J Phase4 \
  monash_exps/scripts/slurm/submit_job_1gpu.sh \
  monash_exps/scripts/phase4/run_phase4_local_fedavg.sh
```

## Phase 5: two real Slakshna peers

Phase 5 runs two unmodified Slakshna Rust nodes on the same M3 compute node.
Peer A owns logical GPU 0 and the even-row shard; Peer B owns logical GPU 1 and
the odd-row shard. They use separate ports, RocksDB directories, Ray clusters,
checkpoints, and logs. The allocation's CPUs are divided between the two ML
children.

Public discovery is disabled. The runner starts Peer A, reads its actual Iroh
EndpointId, and starts Peer B with the explicit direct seed
`EndpointId@127.0.0.1:p2p_port`. Both APIs must report a bidirectional Gossip
neighbor before the first wall-clock-aligned training epoch.

### Two-round lifecycle

Both peers load a common `G0` created without an optimizer step. In Round 1,
they train two steps concurrently and broadcast
`base64(zlib(delta.safetensors))` through the real Rust history/Gossip path.
Before Round 2, Rust stages the remote payload under its peer-specific
`network_deltas/` directory. Each experiment bridge independently validates the
transported bytes and computes:

```text
G1 = G0 + 0.5 * delta_peer_a_round_1 + 0.5 * delta_peer_b_round_1
```

The two `G1` states must be identical. Each peer then starts a fresh optimizer,
trains two steps from `G1`, and broadcasts its Round 2 update. The bridges
return 0.5/0.5 review weights, so Rust creates real `PeerReview` records and a
leaderboard on both nodes.

After the histories converge, both perpetual nodes are stopped. Peer A is
restarted from the same RocksDB with its next epoch placed far in the future.
Its EndpointId and cached Peer B identity must survive. The current upstream
history is in-memory only, so the recovery API is expected to contain zero
updates; this is documented as defect 11 rather than presented as successful
history recovery.

Phase 5 does not enable the stock Python ML engine, dynamic trust learning,
Top-K/error feedback, differential privacy, malicious peers, or cross-node
networking. Content and hash-chain integrity are checked, but upstream record
signatures are placeholders as documented in defect 12. The verifier also
reports which recorded round each review targets: upstream selects the latest
peer record only after Python returns, so concurrent progress can bind a score
to a newer update than the one actually evaluated (defect 13).

### Interactive two-A100 run

```bash
export SLAKSHNA_CLUSTER=m3
export SLAKSHNA_M3_COMPILER_MODULE=gcc/10.2.0
export SLAKSHNA_M3_CUDA_MODULE=cuda/12.8
export SLAKSHNA_UV_ENVIRONMENT="$PWD/monash_exps/.runtime/venvs/primary"
export CUDA_VISIBLE_DEVICES=0,1
source monash_exps/scripts/cluster/activate.sh

bash monash_exps/scripts/phase5/run_phase5_two_peers.sh
```

The runner aligns both peers to 120-second epochs, completes exactly two
rounds, captures both APIs, performs the recovery probe, and terminates every
Rust/Ray child. The final verdict is
`artifacts/phase5/<run-id>/verification-summary.json` and must end with
`PHASE5 PASSED`.

### M3 batch submission

```bash
mkdir -p monash_exps/slurm_logs

sbatch \
  --export=ALL,SLAKSHNA_CLUSTER=m3,SLAKSHNA_M3_COMPILER_MODULE=gcc/10.2.0,SLAKSHNA_M3_CUDA_MODULE=cuda/12.8 \
  --partition=fit --qos=fitq --account=mg61 --gres=gpu:A100:2 \
  -J Phase5 \
  monash_exps/scripts/slurm/submit_job_2gpu.sh \
  monash_exps/scripts/phase5/run_phase5_two_peers.sh
```

### Accepted M3 result

Phase 5 passed on allocation `58796699` with two A100 40 GB GPUs. The accepted
artifact is
`artifacts/phase5/20260806_200417_58796699_m3_phase5` and its verifier reports
`status: PASS` with no failures. Both peers completed two fresh two-step runs,
their eight-record histories converged, and both independently reconstructed
an identical G1 (maximum error 0.0) using 0.5/0.5 FedAvg. The G0-to-G1 maximum
change was `0.0007500899955630302`.

Peer A's median losses were `3.0797` and `2.18275`; Peer B's were `3.4171` and
`2.36355`. Restart preserved Peer A's EndpointId and known-peer cache while the
history API correctly exposed the known upstream persistence gap with zero
records. The observed review target rounds were `[1, 1, 1, 2]`, a concrete
reproduction of defect 13 rather than a verifier failure. The Slakshna source
remained clean at `a3112cf7aa11316d47c6bdf749a45c7071b5f9f3`.

Ray 2.51 still logs a non-fatal detached placement-group-cleaner ambiguity and
temporary resource-budget warnings when two independent clusters share one
host. Phase 5 reserves four of each peer's 24 CPUs for control/data work; all
four datasets and step-2 checkpoints completed despite those warnings.

## Phase 6: two M3 compute nodes

Phase 6 changes only the placement and transport boundary proven in Phase 5.
A single Slurm allocation contains exactly two distinct compute nodes, one task
and one A100 per node. Peer A and Peer B each run an isolated Rust process, Ray
cluster, GPU, ports, RocksDB directory, and data shard. On both nodes the worker
reads Slurm's single numeric `CUDA_VISIBLE_DEVICES` value and writes that value
to the Rust TOML; it does not assume the scheduler assigned physical GPU 0.

The batch controller resolves both node IPv4 addresses, starts Peer A through a
node-pinned `srun`, obtains its real EndpointId, and then starts Peer B on the
other node with:

```text
EndpointId@peer_a_non_loopback_ipv4:p2p_port
```

mDNS, DHT, DNS, and relay remain disabled. A passing mesh therefore proves a
direct M3 node-to-node QUIC/Gossip path rather than same-host loopback or a
public fallback. The HTTP APIs bind to `0.0.0.0` only for the lifetime of the
private compute-node job so the controller can perform fail-closed checks.

### Minimal lifecycle and acceptance boundary

The training contract is intentionally identical to Phase 5: two rounds, two
steps per peer per round, even/odd four-row shards, canonical
`base64(zlib(safetensors))` deltas, and 0.5/0.5 G1 FedAvg. After both eight-record
histories converge, the controller captures APIs, stops both peers, restarts
Peer A on its original node, and verifies its EndpointId and known-peer cache.

The Phase 6 verifier additionally rejects:

- fewer than two distinct hostnames or duplicate/loopback node addresses;
- a seed that does not contain Peer A's actual EndpointId, IP, and P2P port;
- any public discovery or relay setting;
- missing one-A100 Slurm evidence or a Rust GPU id different from the scheduler
  assignment;
- missing direct-address evidence in Peer B's Rust log; or
- allocation-scoped Rust, Python, or Ray process residue on either node.

This minimal milestone does not test public DHT/DNS discovery, DERP relay,
cross-institution NAT, malicious peers, DP, or Top-K compression. If direct M3
QUIC is blocked, the failure artifact is retained and a relay-only networking
smoke test becomes the single fallback experiment.

### Submit and wait from the current environment

The repository-root helper validates the request, submits the real job, prints
queue status every 30 seconds, and displays the final job log and verifier:

```bash
bash tmp.sh | tee output.txt
```

Equivalent direct submission:

```bash
mkdir -p monash_exps/slurm_logs

sbatch \
  --export=ALL,SLAKSHNA_CLUSTER=m3,SLAKSHNA_M3_COMPILER_MODULE=gcc/10.2.0,SLAKSHNA_M3_CUDA_MODULE=cuda/12.8 \
  --partition=fit --qos=fitq --account=mg61 \
  --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=20 \
  --mem=128G --gres=gpu:A100:1 --time=02:00:00 \
  -J SlakshnaPhase6 \
  monash_exps/scripts/slurm/submit_job_2node.sh \
  monash_exps/scripts/phase6/run_phase6_two_nodes.sh
```

The completion pointer is `artifacts/phase6/latest-<job-id>.txt`; its target
must contain `verification-summary.json` ending with `PHASE6 PASSED`.

### Accepted M3 result

Phase 6 passed in Slurm job `58802558`. Peer A ran on `m3u007` at
`172.16.205.203` and Peer B on `m3u008` at `172.16.205.204`, each with one
A100-SXM4-80GB selected by Slurm. With every public discovery and relay path
disabled, Peer B used Peer A's exact `EndpointId@172.16.205.203:32558` seed and
the two peers converged through the direct inter-node path.

All four fresh two-step trainings completed. Both peers converged to identical
eight-record histories and leaderboards, independently reconstructed identical
G1 adapters (maximum absolute error `0.0`), and changed G0 by
`0.0007500899955630302`. Restarting Peer A preserved its EndpointId and cached
Peer B identity; both nodes had zero allocation-scoped process residue after
the controlled shutdown. The accepted artifact is
`artifacts/phase6/20260806_202851_58802558_m3_phase6/`.

The restart also established a narrower recovery boundary: the known-peer
cache stores Peer B's identity but not its dialable address, so the restarted
peer logged `No addressing information available`. This is root README defect
14; Phase 6 does not claim automatic mesh reconnection after restart. The
remaining Ray controller-state warnings were non-fatal, and the final `srun`
termination messages are expected because the controller deliberately stops
the long-lived peer steps after all evidence and cleanup checks are captured.

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
