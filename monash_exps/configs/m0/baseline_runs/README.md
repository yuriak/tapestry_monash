# M0 T5 Baseline Training Package

This package launches the seven direct-Bhaskera OLMo 2 7B baseline runs used
for M0. The frozen configurations live under
`monash_exps/configs/m0/baseline_runs`, while the launchers under
`monash_exps/scripts/m0/baseline_runs` resolve their portable YAML path tokens.
No Python configuration generator or scheduler is required.

## Frozen training contract

Every run uses OLMo 2 7B Instruct, BF16, FlashAttention 2, Liger kernels,
AdamW at `1e-4`, zero weight decay, sequence length 1024, and rank-16 LoRA on
`q_proj` and `v_proj`. Training uses one node with two A100 GPUs, DDP,
per-device batch 2, gradient accumulation 4, and an effective global batch of
16. The budget is two passes over each prepared view. Warmup is approximately
3% of total optimizer steps, followed by Bhaskera's native cosine decay.

| Run | Rows | Optimizer steps | Adapter interval | Expected snapshots |
|---|---:|---:|---:|---:|
| `local_south_asia` | 15,331 | 1,916 | 80 | 24 |
| `local_variant_1` | 9,337 | 1,166 | 50 | 24 |
| `local_variant_2` | 45,047 | 5,630 | 225 | 26 |
| `local_variant_3` | 89,910 | 11,238 | 450 | 25 |
| `central_variant_1` | 24,668 | 3,082 | 125 | 25 |
| `central_variant_2` | 60,378 | 7,546 | 300 | 26 |
| `central_variant_3` | 105,241 | 13,154 | 525 | 26 |

The step budgets account for DDP sharding and `drop_last=true`, so each data
epoch contains `floor(rows / 16)` optimizer steps. Each run saves one recent
full DCP recovery checkpoint and approximately 24–26 immutable adapter-only
trajectory snapshots. The exact final adapter is always saved even when the
final step is not an interval multiple.

## Required paths

`m0_runtime` must be a directory or a manually created link to shared storage.
The launcher never creates or changes this link. By default it uses the
repository's `monash_exps/.runtime` as the asset root when available; otherwise
it uses `m0_runtime/assets`. Override either location explicitly with
`M0_ASSET_ROOT` or `M0_OUTPUT_ROOT`, and use `M0_SLAKSHNA_ROOT` when Slakshna
is not checked out at `./Slakshna`.

The asset root must contain:

```text
models/m0/OLMo-2-1124-7B-Instruct/
artifacts/m0/g0/olmo2-7b-r16-qv-seed20260820.pth
data/m0/tokenized/olmo2-7b-chatml-seq1024/<view>/local_train_c4253f2a8d6a7d19/
```

Every run writes only beneath `m0_runtime/<run-name>/`, including its resolved
configuration, training log, GPU telemetry, Ray results, bounded DCP state,
adapter history, and final `COMPLETED` record. A failed job can be resubmitted
unchanged; Bhaskera resumes from the latest completed DCP and existing adapter
history is not overwritten.

## Launching

From the outer repository root, a direct run inside an existing two-GPU
allocation is:

```bash
bash monash_exps/scripts/m0/baseline_runs/train_t5.sh local_variant_1
```

Resolve and load one configuration without requiring GPUs or starting Ray:

```bash
M0_PREFLIGHT_ONLY=1 bash monash_exps/scripts/m0/baseline_runs/train_t5.sh local_variant_1
```

Preview all seven Slurm submissions without submitting:

```bash
bash monash_exps/scripts/m0/baseline_runs/submit_all.sh --dry-run
```

Submit all runs with the validated M3 defaults (`partition=fit`, `qos=fitq`).
The launcher balances estimated GPU-hours across accounts `mg61` and `sq58`:

```bash
bash monash_exps/scripts/m0/baseline_runs/submit_all.sh
```

The default account assignment is `mg61` for `local_south_asia`,
`local_variant_1`, `local_variant_3`, and `central_variant_2`; the remaining
three runs use `sq58`. Override the two account names with `M0_ACCOUNT_MG61`
and `M0_ACCOUNT_SQ58`, or override the site fields with
`M0_SLURM_PARTITION` and `M0_SLURM_QOS`.

To submit a subset, provide its names through `M0_RUNS`, for example:

```bash
M0_RUNS="local_variant_1 central_variant_1" \
  bash monash_exps/scripts/m0/baseline_runs/submit_all.sh
```

The submission command records every returned job ID under
`m0_runtime/slurm/`. Do not launch production jobs until the asset root,
shared-output capacity, partition, account, GPU type, and requested wall times
have been checked for the target cluster.

The submitter passes the absolute Slakshna root and sets the Slurm working
directory explicitly. The copied script under `/var/spool/slurmd/` therefore
never attempts to locate project files relative to its temporary spool path.

## Version provenance

These launchers were used with Slakshna commit
`036e8fbe570bd2dabfcf4b65d44569b2fba11876`. The corresponding local Bhaskera
changes are preserved under `monash_exps/patches/slakshna`; they target the
older vendored Bhaskera layout and must be reviewed rather than applied
blindly to a newer Slakshna checkout.
