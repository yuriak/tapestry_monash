# M0 local-FL operator scripts

Run these scripts from the repository root. Generated state, credentials and
Playit endpoint files belong below `monash_exps/.runtime/` or the external
runtime linked at `Slakshna/m0_runtime`; none of them belongs in Git.

## M3

1. `01_upgrade_environment.sh` reproduces the pinned local-FL software stack.
2. `02_download_and_audit_data_team_cache.sh` audits the supplied token cache.
3. `manage_m3_playit.sh` manages the M3 backup Playit agent and writes the
   machine-local endpoint config to `.runtime/configs/m0_fl/m3_playit.toml`.
4. `04_submit_m3.sh --test-only` prints and validates the Slurm submission;
   `--submit` performs it.
5. `03_evaluate_local_fl.sh` finalizes and evaluates an accepted completed run.

The accepted repaired run used the revisions recorded by
`01_upgrade_environment.sh`. Do not override its revision gates merely to make
an updated checkout pass; audit the runtime patch against the new source first.

## Spartan

The Spartan scripts are retained here as versioned deployment code rather than
copied root-level helpers:

- `sync_spartan_assets.sh` transfers only large data/cache assets and defaults
  to `--dry-run`;
- `spartan_prepare.sh` prepares the environment and external runtime link;
- `manage_spartan_playit.sh` manages the India ingress agent;
- `submit_spartan.sh` submits `spartan_formal_job.sbatch`.

Do not run the Spartan workflow until the reviewed M3 changes have been
committed and the Spartan checkout and both submodules have been synchronized.
The remote runtime and Playit secrets must be preserved during that update.
