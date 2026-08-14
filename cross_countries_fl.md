# Phase 9 Cross-Countries Federated Learning Runbook

This runbook covers both the two-client Playit rehearsal on one interactive GPU node and the later Australia–India run. Both modes use the official Slakshna `v0.1.1-alpha` code path without replacing its training, aggregation, compression, trust, or networking logic.

## Canonical files

The only deployment configuration that normally needs editing is:

```text
Slakshna/configs/phase9/cross_countries_fl.yaml
```

The outer repository tracks its clean deployment default at
`monash_exps/configs/phase9/cross_countries_fl.yaml`. Phase 9 native preparation
installs that default into Slakshna, and the launcher installs it only when the
deployment copy is absent. It never overwrites an existing deployment copy, so
site-specific endpoint edits survive repeated launches. Commit changes to the
tracked default only when the shared experiment protocol itself changes; make
per-session endpoint edits in the Slakshna deployment copy above.

The common launcher is:

```text
5_cross_countries_fl.sh
```

The launcher generates effective per-client Rust TOML and Bhaskera YAML files under the run artifact directory. Generated files are evidence, not files that should be edited manually.

Do not put a Playit account secret, claim URL, tunnel token, password, or API key in the YAML file or in Git. The agent secret stays under `monash_exps/.runtime/secrets/playit/` with mode `0600`.

## Fixed training recipe

The rehearsal uses two symmetric stock Slakshna clients in the same federation. The Australia client uses the complete Australia split and the India client uses the complete India split. Each client performs 50 local optimizer steps per federated round with OLMo-1B, LoRA rank 8, sequence length 512, batch size 2, gradient accumulation 4, BF16, FlashAttention 2, and learning rate `1e-4`. Two consecutive federated rounds are required.

The first round deliberately starts without a tokenized cache. Stock `ml_engine.py` invokes the stock Bhaskera tokenizer against the full raw JSONL file. The generated cache is reused unchanged in round two. Existing Phase 9 offline caches elsewhere in the repository are not copied into either client runtime.

Both clients disable mDNS, DHT, pkarr/DNS discovery, and Iroh relay fallback. Each generated node config contains exactly one pinned peer of the form:

```text
<remote Iroh EndpointId>@<remote Playit public IPv4>:<remote Playit public UDP port>
```

The public hostname is resolved to an IPv4 address before the TOML is generated. Loopback and private addresses are rejected. Thus the two-client rehearsal cannot silently fall back to localhost discovery even though both processes run on one allocation.

## One-node Playit rehearsal

The Playit dashboard must assign two custom UDP tunnels to the already claimed M3 agent before starting the rehearsal:

| Client | Playit local address | Current public endpoint |
| --- | --- | --- |
| Australia | `127.0.0.1:38080` | `147.185.221.231:51716` |
| India | `127.0.0.1:38081` | `147.185.221.231:42482` |

In the Playit dashboard, confirm that both tunnels are UDP tunnels, are enabled, are assigned to the same online agent, and target the exact local ports above. A TCP-only tunnel is not suitable for Iroh QUIC.

After creating the second tunnel, edit these two values in `Slakshna/configs/phase9/cross_countries_fl.yaml`:

```yaml
clients:
  india:
    public_host: REPLACE_WITH_SECOND_PLAYIT_HOST
    public_port: 0
```

Keep the Australia endpoint unchanged unless the existing tunnel assignment has changed. The current deployment uses Playit's numeric public IPv4 for both clients because DNS is unavailable from the allocated compute node; the distinct UDP ports select the two tunnels. The `endpoint_id` placeholders do not need to be edited for the one-node rehearsal; the launcher creates two persistent identities, reads both Iroh EndpointIds, and wires them together automatically.

The rehearsal configuration assigns Australia to visible GPU `0` and India to visible GPU `1`, matching the current two-GPU allocation. If a later rehearsal has only one sufficiently large GPU, set `clients.india.cuda_visible_devices` to `"0"`. In the live two-cluster run, each site will normally set its own client to the locally visible GPU `0`.

Run the complete rehearsal with:

```bash
bash 5_cross_countries_fl.sh local | tee output.txt
```

The launcher performs the following checks before training:

1. The Playit agent reports at least two assigned tunnels.
2. Each public UDP endpoint completes an authenticated public-to-local echo probe.
3. Both stock Slakshna identities persist across a restart.
4. Each node config pins only the other client's public Playit endpoint.
5. Both REST APIs report a live gossip neighbour before the first training boundary.

The run may wait for the next 420-second wall-clock boundary. Two full rounds can therefore take roughly 15–20 minutes. The terminal prints a progress line every 30 seconds. A successful run ends with:

```text
PHASE 9 CROSS-COUNTRIES STOCK FL PASSED
```

Evidence is written under:

```text
monash_exps/artifacts/phase9/cross-countries/<run-id>/
```

The final audit requires two rounds per client, 100 finite loss records per client, one live-generated full-data token cache per client, an unchanged cache between rounds, a step-50 adapter checkpoint, received peer updates, a staged and decoded peer delta in round two, and a two-member ML-engine trust state.

## Australia–India live run

Both clusters must use the same Git commit, Slakshna release commit, model revision, federation ID, model configuration, and wire-format configuration. On each cluster, complete the Phase 9 preparation sequence first:

```bash
bash 1_setup_env.sh
bash 2_prepare_data.sh
bash 3_prepare_native.sh
```

`3_prepare_native.sh` may create reference offline caches, but the cross-country runtime does not consume them. Its first round still follows live stock tokenization.

On each cluster, edit the same canonical YAML file. The following fields must match on both sides:

- `federation.id`
- `federation.expected_peers`
- `federation.rounds`
- `federation.epoch_duration_secs`
- `federation.sync_deadline_secs`
- both clients' public Playit hosts and UDP ports
- both clients' Iroh EndpointIds after identity exchange

Site-local fields may differ:

- `clients.<site>.local_p2p_port` must equal that site's Playit local tunnel port;
- `clients.<site>.local_api_port` and `local_ws_port` only need to be free locally;
- `clients.<site>.cuda_visible_devices` must refer to an allocated GPU as seen after scheduler isolation.

Each cluster needs only one assigned UDP tunnel for the live run. Claim or reuse its Playit agent with:

```bash
bash 5_cross_countries_fl.sh claim-agent
```

Before training, create the stable local identity on each cluster:

```bash
# Australia cluster
bash 5_cross_countries_fl.sh identity australia

# India cluster
bash 5_cross_countries_fl.sh identity india
```

Each command prints two different identifiers. Exchange the 64-character hexadecimal `endpoint_id`; do not exchange the `slakshna1...` node ID as the network seed. Put the exchanged values into both copies of the YAML file:

```yaml
clients:
  australia:
    endpoint_id: <AUSTRALIA_64_HEX_IROH_ENDPOINT_ID>
  india:
    endpoint_id: <INDIA_64_HEX_IROH_ENDPOINT_ID>
```

The identity is stored below `monash_exps/.runtime/phase9/cross-country/`. Do not delete that directory between identity exchange and training, because deleting its RocksDB state changes both identifiers.

Start both sides during the same coordination window:

```bash
# Australia cluster
bash 5_cross_countries_fl.sh site australia | tee australia_output.txt

# India cluster
bash 5_cross_countries_fl.sh site india | tee india_output.txt
```

Each launcher waits past an unsafe near-boundary window when necessary, verifies its own public Playit tunnel, pins the remote public endpoint, and refuses to train until the remote gossip neighbour is visible. No source-code edit is required when moving from rehearsal to the live run.

## Operational cautions

The federation ID is part of the gossip topic. A one-character mismatch creates two healthy-looking but disjoint federations. The Iroh EndpointId and the `slakshna1...` ML node ID are different and must not be interchanged. Playit forwards only the P2P UDP port; the REST API and WebSocket ports remain loopback-only.

Stock Slakshna checkpoints include a full DDP state of roughly 2.6 GB per client even though the portable adapter is only about 4 MB. `keep_last_n=1` is retained to limit growth. Allow at least 10 GB of free shared storage for a two-client run.

The stock Rust `/status` round remains zero because the release does not call `State::set_round`; the authoritative completed-round evidence is the per-client ML-engine state plus accepted Rust update records. This known release behavior is recorded but is not treated as a failed training round.

To inspect or stop only the managed Phase 9 Playit agent:

```bash
bash 5_cross_countries_fl.sh agent-status
bash 5_cross_countries_fl.sh stop-agent
```

The normal local and site launchers stop their managed Playit agent automatically on exit or interruption.
