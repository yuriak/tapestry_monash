# Cross-Cluster Federated Training Validation Report

**Reporting date:** 11 August 2026  
**Coverage:** Phase 8

## Executive Summary

Phase 8 completed a real five-round federated fine-tuning experiment between
two independently deployed compute sites with no shared filesystem. The sites
prepared disjoint non-IID data locally, installed one cryptographically
verified initial adapter, connected through an authenticated Iroh QUIC/Gossip
mesh over a managed UDP ingress, exchanged dense LoRA updates, and independently
reconstructed every global model from G1 through G5. A sixth invocation
performed aggregation only, ensuring that both sites finalized the fifth peer
update before termination.

The accepted run met its core objective: the current training and networking
stack can complete foundational cross-site federated training and exhibit a
convergent training-loss trajectory without relying on shared storage. Each
site completed 50 local epochs and 3,600 optimizer steps. Median training loss
from the beginning of Round 1 to the end of Round 5 decreased by 36.84% at Site
A and 30.81% at Site B. Independent site audits passed, and a final paired
verifier matched all ten transmitted and received update envelopes as well as
all five global model hashes.

## Experimental Contract

The experiment used the pinned `train` split of
`databricks/databricks-dolly-15k` at revision
`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`. The partition was deterministic,
disjoint by source index, and non-IID by task category. Site A owned the closed
QA, open QA, and information-extraction categories. Site B owned brainstorming,
creative-writing, and summarization examples. Each site retained 1,152 records
for training and 128 records for local validation; neither site read the other
site's private files.

| Site | Assigned categories and counts | Training records | Validation records |
|---|---|---:|---:|
| A | Closed QA: 335; open QA: 670; information extraction: 275 | 1,152 | 128 |
| B | Brainstorming: 608; creative writing: 247; summarization: 425 | 1,152 | 128 |

| Item | Accepted value |
|---|---|
| Base model | Qwen3-0.6B at pinned revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| Precision and attention | BF16 model path, FP32 dense adapter deltas, PyTorch SDPA |
| Quantization | None |
| LoRA | Rank 8, alpha 16, dropout 0.0, `q_proj` and `v_proj` targets |
| Adapter structure | 112 tensors and 1,146,880 trainable parameters |
| Sequence configuration | ChatML, 256-token maximum, no sequence packing |
| Local batch configuration | Batch size 16, gradient accumulation 1 |
| Optimizer schedule | Peak learning rate `5e-5`, 20 warmup steps, cosine decay |
| Regularization | Weight decay 0.0, maximum gradient norm 1.0 |
| Federated algorithm | Equal-weight dense FedAvg with weights 0.5 and 0.5 |
| Training rounds | 5 plus one aggregation-only finalizer |
| Local budget per round | 10 data epochs and 720 optimizer steps |
| Total budget per site | 50 local epochs and 3,600 optimizer steps |
| Synchronization | Globally aligned 600-second epochs with a 570-second deadline |

The full path-independent training contract was hashed before execution. Both
sites installed the same G0 adapter and verified its safetensors file hash,
logical tensor-state hash, PyTorch resume-state hash, tensor schema, and
training-contract hash. This made G0 a common independently verifiable starting
point rather than an assumption based on a transferred filename.

## Cross-Site Bridge and Operational Code

Phase 8 added a site-local bridge that preserves the existing Rust/Python
boundary while removing the shared-filesystem assumption used by the preceding
phase. Slakshna treats `compressed_delta` as an opaque value; the bridge places
a versioned JSON envelope in that field. The envelope binds the payload to its
site role, authenticated endpoint identity, federation round, base-state hash,
delta file hash, logical tensor-state hash, byte counts, tensor count, and
parameter count. The update itself is a zlib-compressed FP32 safetensors file
encoded as base64. Decoding rejects unknown fields, wrong senders, stale or
future rounds, incorrect bases, malformed compression, excessive sizes,
schema mismatches, and non-finite tensors before aggregation.

Additional code prepared deterministic private shards, generated and verified
portable G0 bundles, rendered cluster-neutral Slakshna runtimes, managed the
external UDP agent lifecycle, performed an authenticated preflight probe, and
exported path-free site audits. Tracked site runners own preparation, endpoint
identity bootstrap, reciprocal allowlists, foreground process cleanup, failed
run reset, runtime monitoring, and final audit generation. A paired verifier
then compares the independent site documents without requiring either site's
private data or local runtime paths.

## Execution and Recovery

The external UDP preflight first established bidirectional datagram delivery
from the initiating site to the ingress site. The first full training attempt
then connected the real Iroh mesh and completed the first round at both sites.
It stopped during Site B's second-round post-training verification. Training
itself had completed all 720 steps and produced the expected 112-tensor
checkpoint, but a verifier inherited from the initial single-site phase
required every local round to show non-negative median loss improvement. The
observed stochastic change was only -0.678%.

That requirement was inappropriate for later federated rounds, which begin
from an already-trained peer-averaged adapter and may fluctuate under shuffled
local data. The bridge was corrected to tolerate at most a 5% local regression
while retaining strict finite-metric, step-count, checkpoint, resource, and
adapter-structure checks. Convergence is judged from the complete multi-round
training trajectory rather than monotonicity inside every local round. The
failed sites had reached different bridge states, so explicit reset actions
were added to retain prepared private data and verified G0 while deleting
derived rounds and issuing fresh transport identities. All eight Phase 8 unit
tests passed after these changes.

The failure also provided runtime evidence for an upstream error-handling
limitation. After the Python ML engine exited nonzero, the Rust node remained
live, attempted to record an invalid empty-hash update, and broadcast a small
placeholder payload. The external runner detected the logged error and stopped
the site, but a production implementation should fail the round before history
insertion or network broadcast.

## Accepted Results

The retry completed without Python, Rust, panic, timeout, or cleanup errors.
The table reports median per-step training loss for each fresh local optimizer
run and the independently reconstructed global-state hash shared by both sites.

| Round | Site A initial | Site A final | Site A change | Site B initial | Site B final | Site B change | Common global state |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 3.0153 | 2.0873 | +30.78% | 3.1207 | 2.3289 | +25.37% | `e993744d…2ff38` |
| 2 | 2.2260 | 2.0231 | +9.12% | 2.2557 | 2.2710 | -0.68% | `2ae74feb…9f9b3` |
| 3 | 2.1639 | 1.9748 | +8.74% | 2.2014 | 2.2318 | -1.38% | `90eb3475…b3949` |
| 4 | 2.1218 | 1.9355 | +8.78% | 2.1603 | 2.1953 | -1.62% | `f1243b44…7362` |
| 5 | 2.0833 | 1.9046 | +8.58% | 2.1253 | 2.1591 | -1.59% | `d66ea553…5651f` |

Site A's final median loss decreased in every round. Site B's later individual
runs showed small negative changes, but its cross-round starting loss decreased
from 3.1207 to 2.1253 and its final loss decreased from 2.3289 to 2.1591. The
complete Round-1-initial to Round-5-final changes were therefore 36.84% and
30.81%. These results establish training-loss convergence; they do not add a
new held-out negative-log-likelihood claim beyond the separate evaluation
already completed in Phase 7.

Site A transmitted 27,997,759 wire bytes across five envelopes and Site B
transmitted 28,001,747 bytes. The paired verifier matched every local outbound
envelope to the remote received envelope, including sender identity, round,
base state, file digest, and tensor-state digest. G1 through G5 were identical
between the two independent reconstructions. The final G5 state was
`d66ea5539d65bcdfb5a8f652f5e249e865a09689ed77801d4cc6ea2cc525651f`,
and the immutable training-contract hash was
`86c7132ba77364c8c3380e37b59e11f81a04bc4a1ee15b4e72625e9dcc0a6bcb`.

## Timing and Conclusion

The accepted node lifecycle lasted approximately 58 minutes 46 seconds from
startup through site-audit completion. This included about 8 minutes 26 seconds
waiting for the first globally aligned epoch boundary. Each 720-step local
training interval took roughly three minutes on the observed accelerator, with
the remainder of each ten-minute epoch spent behind the fixed synchronization
deadline. This repeats the utilization behavior measured in Phase 7 but does
not affect the correctness claim.

Phase 8 is accepted. It demonstrates that the current code can deploy at two
independent sites, connect through a constrained public ingress, keep training
data and runtime state private to each site, exchange and authenticate model
updates, reconstruct identical global adapters, complete the intended training
budget, and produce a convergent training-loss trajectory. The result is a
validation of foundational training capability, not a claim that the framework
is production-hardened or Byzantine-secure.
