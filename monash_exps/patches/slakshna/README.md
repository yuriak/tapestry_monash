# Archived Slakshna Changes for M0

The patch in this directory preserves the local Slakshna/Bhaskera changes used
to run the M0 baseline suite before updating the Slakshna submodule.

| Field | Value |
|---|---|
| Slakshna base commit | `036e8fbe570bd2dabfcf4b65d44569b2fba11876` |
| Patch file | `20260824_m0_bhaskera_local_changes.patch` |
| Patch SHA-256 | `6c2fa594a72abbee56eb32706d5cc136d58b60c1997a407affe3260fb43014c1` |
| Upstream main observed on 2026-08-24 | `b1317dc97093a31476976d55ff76f29f2a04d4b3` |
| Nested Bhaskera revision in that upstream tree | `d737ced2c5f61cf9d96c1e041969ca099b1785fa` |

The changes add Slurm-aware CPU allocation, scheduler state restoration,
bounded DCP retention, and immutable adapter-only trajectory snapshots. They
also document the corresponding checkpoint behavior and add the required
configuration fields.

The newer Slakshna tree changes Bhaskera from vendored source to a nested Git
submodule and also introduces upstream checkpoint behavior of its own. This
patch therefore records provenance; it must not be applied blindly after the
submodule update. Reconcile each behavior against the new Bhaskera API and
retain only what the next FL experiment still needs.

The seven frozen M0 configurations and their launchers were moved out of the
Slakshna worktree to `monash_exps/configs/m0/baseline_runs` and
`monash_exps/scripts/m0/baseline_runs`, respectively. Runtime checkpoints,
model weights, token caches, and machine-specific logs are intentionally not
included in Git.
