#!/usr/bin/env python3
"""Run stock Slakshna clients through pinned Playit UDP endpoints."""
from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import math
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import tomllib
import yaml
from run_native_smoke import dump_flat_toml, sha256_file, write_json

EXPECTED_REVISION = "9f93ec45ae0d3eb9c901aff3b50d4325b5050488"
SITES = ("australia", "india")
IDENTITY_RE = re.compile(r"Node Identity:\s+(slakshna1[0-9a-z]+)")
FAILURE_MARKERS = (
    "Python ML Engine failed",
    "Failed to start Python process",
    "panicked at",
)


def repository_roots() -> tuple[Path, Path, Path]:
    experiment_root = Path(__file__).resolve().parents[2]
    workspace_root = experiment_root.parent
    return workspace_root, experiment_root, workspace_root / "Slakshna"


def load_settings(path: Path, require_public: bool) -> dict[str, Any]:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if settings.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported cross-countries schema in {path}")
    federation = settings.get("federation", {})
    if not federation.get("id") or not federation.get("name"):
        raise RuntimeError("Federation id and name are required")
    numeric_rules = {
        "expected_peers": (2, 2),
        "rounds": (2, 20),
        "epoch_duration_secs": (120, 3600),
        "sync_deadline_secs": (60, 3599),
        "run_timeout_secs": (600, 14400),
    }
    for field, (minimum, maximum) in numeric_rules.items():
        value = federation.get(field)
        if not isinstance(value, int) or not minimum <= value <= maximum:
            raise RuntimeError(f"Invalid federation.{field}: {value!r}")
    if federation["sync_deadline_secs"] >= federation["epoch_duration_secs"]:
        raise RuntimeError("sync_deadline_secs must be below epoch_duration_secs")
    if settings.get("playit", {}).get("protocol") != "udp":
        raise RuntimeError("Playit protocol must be udp")

    clients = settings.get("clients", {})
    if set(clients) != set(SITES):
        raise RuntimeError(f"Exactly these clients are required: {SITES}")
    local_ports = []
    public_pairs = []
    for site in SITES:
        client = clients[site]
        if client.get("profile") != f"{site}-full":
            raise RuntimeError(f"{site} must use the {site}-full profile")
        for field in ("local_p2p_port", "local_api_port", "local_ws_port"):
            value = client.get(field)
            if not isinstance(value, int) or not 1024 <= value <= 65535:
                raise RuntimeError(f"Invalid clients.{site}.{field}: {value!r}")
            local_ports.append(value)
        cuda = client.get("cuda_visible_devices")
        if not isinstance(cuda, str) or not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", cuda):
            raise RuntimeError(f"Invalid CUDA device list for {site}: {cuda!r}")
        if require_public:
            host = str(client.get("public_host", ""))
            port = client.get("public_port")
            if not host or host.startswith("REPLACE_"):
                raise RuntimeError(
                    f"Set clients.{site}.public_host in {path} before launching"
                )
            if not isinstance(port, int) or not 1024 <= port <= 65535:
                raise RuntimeError(
                    f"Set clients.{site}.public_port in {path} before launching"
                )
            public_pairs.append((host, port))
    if len(local_ports) != len(set(local_ports)):
        raise RuntimeError("All local P2P/API/WS ports must be distinct")
    if require_public and len(public_pairs) != len(set(public_pairs)):
        raise RuntimeError("The two clients must use distinct Playit public endpoints")
    return settings


def resolve_public_ipv4(host: str, port: int) -> str:
    addresses = sorted(
        {
            item[4][0]
            for item in socket.getaddrinfo(
                host, port, socket.AF_INET, socket.SOCK_DGRAM
            )
        }
    )
    if not addresses:
        raise RuntimeError(f"No IPv4 address resolved for {host}")
    address = ipaddress.ip_address(addresses[0])
    if not address.is_global:
        raise RuntimeError(f"Playit endpoint must resolve to a public IPv4 address: {address}")
    return str(address)


def api_get(port: int, endpoint: str, timeout: float = 3.0) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{endpoint}", timeout=timeout
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_api(port: int, endpoint: str = "/status", timeout: int = 60) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return api_get(port, endpoint)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(0.25)
    raise TimeoutError(f"API {endpoint} on port {port} did not become ready: {last_error}")


class ClientProcess:
    def __init__(
        self,
        site: str,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
    ):
        self.site = site
        self.command = command
        self.cwd = cwd
        self.env = env
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.node_id: str | None = None
        self.local_completions = 0
        self.epoch_completions = 0
        self.peer_joins = 0
        self.received_updates = 0
        self.broadcasts = 0
        self.failures: list[str] = []

    def start(self, echo: bool = True) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self.log_path.open("w", encoding="utf-8", buffering=1)
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def consume() -> None:
            assert self.process is not None and self.process.stdout is not None
            try:
                for line in self.process.stdout:
                    log_handle.write(line)
                    if echo:
                        print(f"[{self.site}] {line}", end="", flush=True)
                    identity = IDENTITY_RE.search(line)
                    if identity:
                        self.node_id = identity.group(1)
                    self.local_completions += line.count("Local Training Complete!")
                    self.epoch_completions += line.count(
                        "Most-trusted cohort this epoch"
                    )
                    self.peer_joins += line.count("Peer joined gossip mesh")
                    self.received_updates += line.count(
                        "Gossiped model update received"
                    )
                    self.broadcasts += line.count("Broadcasting model update to swarm")
                    for marker in FAILURE_MARKERS:
                        if marker in line:
                            self.failures.append(line.strip())
            finally:
                log_handle.close()

        self.thread = threading.Thread(target=consume, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
        if self.thread is not None:
            self.thread.join(timeout=5)

    def wait_identity(self, timeout: int = 60) -> str:
        assert self.process is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.node_id:
                return self.node_id
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.site} exited before identity initialization")
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for {self.site} identity")


def base_environment(cuda_devices: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_devices,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    rust_ld = environment.get("PHASE9_RUST_LD_LIBRARY_PATH")
    if rust_ld is not None:
        environment["LD_LIBRARY_PATH"] = rust_ld
    return environment


def native_outputs(experiment_root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = experiment_root / ".runtime/manifests/phase9/prepare-native.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("slakshna_revision") != EXPECTED_REVISION:
        raise RuntimeError(
            "Run bash monash_exps/scripts/phase9/prepare_native.sh "
            "for the pinned Phase 9 release"
        )
    rust_binary = experiment_root / ".runtime/cargo-target/phase9-stock/release/iiitd"
    if not os.access(rust_binary, os.X_OK):
        raise RuntimeError(f"Missing release binary: {rust_binary}")
    return manifest, rust_binary.resolve()


def prepare_engine_runtime(
    *,
    site: str,
    runtime: Path,
    slakshna_root: Path,
    native_manifest: dict[str, Any],
) -> dict[str, Any]:
    runtime.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    source_files = [slakshna_root / "ml_engine.py"]
    source_files.extend(
        sorted((slakshna_root / "federated_communication").rglob("*.py"))
    )
    for source in source_files:
        relative = source.relative_to(slakshna_root)
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"Runtime copy differs from stock {relative}")
        copied[relative.as_posix()] = sha256_file(destination)

    profile_name = f"{site}-full"
    profile = native_manifest["profiles"][profile_name]
    source_config = Path(profile["resolved_config"])
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["data"]["tokenized_path"] = None
    config["data"]["val_tokenized_path"] = None
    config["data"]["cache_dir"] = None
    config["data"]["overwrite_cache"] = False
    config["logging"]["run_name"] = f"phase9-cross-countries-{site}"
    config["training"]["max_steps"] = int(profile["source"]["max_steps"])
    config["checkpoint"]["keep_last_n"] = 1
    node_template = runtime / "node_template.yaml"
    node_template.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    from bhaskera.config import load_config

    audited = load_config(str(node_template))
    if audited.data.tokenized_path is not None or audited.data.cache_dir is not None:
        raise RuntimeError("Cross-countries runtime unexpectedly enables a token cache")
    if audited.data.train_path != profile["config_audit"]["train_path"]:
        raise RuntimeError(f"Raw full-data path mismatch for {site}")
    if audited.training.max_steps != 50:
        raise RuntimeError(f"Expected 50 local steps for {site}")
    return {
        "profile": profile_name,
        "stock_source_hashes": copied,
        "node_template": str(node_template),
        "node_template_sha256": sha256_file(node_template),
        "raw_train_path": audited.data.train_path,
        "expected_train_rows": profile["train_cache"]["rows"],
        "max_steps": audited.training.max_steps,
        "seq_len": audited.data.seq_len,
        "batch_size": audited.training.batch_size,
        "grad_accum": audited.training.grad_accum,
    }


def generate_node_config(
    *,
    settings: dict[str, Any],
    site: str,
    runtime: Path,
    peer: str | None,
    allowed_peer: str | None,
) -> Path:
    federation = settings["federation"]
    client = settings["clients"][site]
    config: dict[str, Any] = {
        "federation": {"id": federation["id"], "name": federation["name"]},
        "training": {
            "epoch_duration_secs": federation["epoch_duration_secs"],
            "sync_deadline_secs": federation["sync_deadline_secs"],
            "expected_peers": federation["expected_peers"],
        },
        "compression": {
            "enabled": True,
            "sparsity": 0.1,
            "quantization": "symmetric_int8",
            "allow_legacy_delta_format": False,
            "max_payload_bytes": 7_340_032,
            "max_tensor_elements": 10_000_000,
        },
        "node": {
            "id": f"phase9-{site}",
            "data_dir": str((runtime / "node-state").resolve()),
            "gpu_id": 0,
            "num_gpus": 1,
        },
        "network": {
            "host": "127.0.0.1",
            "p2p_port": client["local_p2p_port"],
            "ws_port": client["local_ws_port"],
            "api_port": client["local_api_port"],
            "peers": [peer] if peer else [],
            "allowed_peers": [allowed_peer] if allowed_peer else [],
        },
        "discovery": {"mdns": False, "dht": False, "dns": False, "relay": False},
        "logging": {"level": "info"},
    }
    output = runtime / "node.toml"
    output.write_text(dump_flat_toml(config), encoding="utf-8")
    with output.open("rb") as handle:
        parsed = tomllib.load(handle)
    if parsed != config:
        raise RuntimeError(f"Generated TOML round-trip failed for {site}")
    return output


def bootstrap_identity(
    *,
    site: str,
    runtime: Path,
    settings: dict[str, Any],
    rust_binary: Path,
) -> dict[str, Any]:
    config_path = generate_node_config(
        settings=settings, site=site, runtime=runtime, peer=None, allowed_peer=None
    )
    client = settings["clients"][site]
    process = ClientProcess(
        site,
        [str(rust_binary), "--config", str(config_path)],
        runtime,
        base_environment(client["cuda_visible_devices"]),
        runtime / "identity-bootstrap.log",
    )
    process.start(echo=False)
    try:
        node_id = process.wait_identity()
        status = wait_api(client["local_api_port"])
        endpoint_id = status.get("endpoint_id")
        if not isinstance(endpoint_id, str) or not re.fullmatch(r"[0-9a-f]{64}", endpoint_id):
            raise RuntimeError(f"Invalid Iroh EndpointId for {site}: {endpoint_id!r}")
    finally:
        process.stop()
    identity = {
        "site": site,
        "node_id": node_id,
        "endpoint_id": endpoint_id,
        "data_dir": str((runtime / "node-state").resolve()),
    }
    write_json(runtime / "identity.json", identity)
    return identity


def tunnel_probe(
    *,
    site: str,
    client: dict[str, Any],
    output_root: Path,
    experiment_root: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    host = str(client["public_host"])
    public_port = int(client["public_port"])
    resolved = resolve_public_ipv4(host, public_port)
    token = f"phase9-{site}-{secrets.token_hex(12)}"
    server_json = output_root / f"playit-{site}-server.json"
    client_json = output_root / f"playit-{site}-client.json"
    server_log = output_root / f"playit-{site}-server.log"
    probe_script = experiment_root / "src/phase8/udp_probe.py"
    with server_log.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                str(probe_script),
                "server",
                "--bind",
                "127.0.0.1",
                "--port",
                str(client["local_p2p_port"]),
                "--token",
                token,
                "--wait",
                "90",
                "--output",
                str(server_json),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            time.sleep(1)
            subprocess.run(
                [
                    sys.executable,
                    str(probe_script),
                    "client",
                    "--host",
                    host,
                    "--port",
                    str(public_port),
                    "--token",
                    token,
                    "--attempts",
                    str(settings["playit"]["probe_attempts"]),
                    "--timeout",
                    str(settings["playit"]["probe_timeout_secs"]),
                    "--interval",
                    "1",
                    "--output",
                    str(client_json),
                ],
                check=True,
            )
            server.wait(timeout=15)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
    server_result = json.loads(server_json.read_text(encoding="utf-8"))
    client_result = json.loads(client_json.read_text(encoding="utf-8"))
    if server_result.get("status") != "PASS" or client_result.get("status") != "PASS":
        raise RuntimeError(f"Playit UDP preflight failed for {site}")
    return {
        "site": site,
        "public_host": host,
        "public_ipv4": resolved,
        "public_port": public_port,
        "local_port": client["local_p2p_port"],
        "round_trip_ms": client_result["round_trip_ms"],
        "evidence": {"server": str(server_json), "client": str(client_json)},
    }


def wait_until_safe_start(epoch_duration: int, minimum_window: int = 120) -> None:
    now = int(time.time())
    remaining = epoch_duration - (now % epoch_duration)
    if remaining < minimum_window:
        wait = remaining + 2
        print(
            f"Only {remaining}s remain in the current federation interval; "
            f"waiting {wait}s so networking can connect before training.",
            flush=True,
        )
        time.sleep(wait)


def wait_connected(processes: dict[str, ClientProcess], settings: dict[str, Any]) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        connected = {}
        for site, process in processes.items():
            if process.failures:
                raise RuntimeError(f"{site}: {'; '.join(process.failures)}")
            if process.process is not None and process.process.poll() is not None:
                raise RuntimeError(f"{site} exited before Playit mesh connectivity")
            try:
                status = api_get(settings["clients"][site]["local_api_port"], "/status")
                connected[site] = int(status.get("peers", 0)) >= 1
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                connected[site] = False
        if connected and all(connected.values()):
            print("Both clients joined the gossip mesh through Playit.", flush=True)
            return
        time.sleep(0.5)
    raise TimeoutError("The clients did not join each other through Playit within 120 seconds")


def directory_inventory(path: Path) -> list[dict[str, Any]]:
    result = []
    if not path.is_dir():
        return result
    for item in sorted(path.rglob("*")):
        if item.is_file():
            result.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
    return result


def monitor_rounds(
    processes: dict[str, ClientProcess],
    runtimes: dict[str, Path],
    identities: dict[str, dict[str, Any]],
    rounds: int,
    timeout: int,
) -> dict[str, list[dict[str, Any]]]:
    snapshots: dict[str, list[dict[str, Any]]] = {site: [] for site in processes}
    observed = {site: 0 for site in processes}
    deadline = time.monotonic() + timeout
    next_progress = time.monotonic() + 30
    while time.monotonic() < deadline:
        all_done = True
        for site, process in processes.items():
            if process.failures:
                raise RuntimeError(f"{site}: {'; '.join(process.failures)}")
            if process.process is not None and process.process.poll() is not None:
                raise RuntimeError(f"{site} exited during federated training")
            while observed[site] < process.local_completions:
                observed[site] += 1
                node_id = identities[site]["node_id"]
                cache_root = runtimes[site] / "data" / f"data_{node_id}" / "tokenized_cache"
                base_lora = runtimes[site] / "ml_models" / f"{node_id}_base_lora.pth"
                snapshot = {
                    "round": observed[site],
                    "cache": directory_inventory(cache_root),
                    "base_lora_sha256": sha256_file(base_lora) if base_lora.is_file() else None,
                }
                snapshots[site].append(snapshot)
                print(
                    f"[{site}] accepted local training result {observed[site]}/{rounds}",
                    flush=True,
                )
            if process.local_completions < rounds or process.epoch_completions < rounds:
                all_done = False
        if all_done:
            return snapshots
        if time.monotonic() >= next_progress:
            progress = ", ".join(
                f"{site}: training={process.local_completions}/{rounds}, "
                f"epochs={process.epoch_completions}/{rounds}"
                for site, process in processes.items()
            )
            print(f"Federated run progress — {progress}", flush=True)
            next_progress = time.monotonic() + 30
        time.sleep(0.25)
    progress = {
        site: {
            "local": process.local_completions,
            "epoch": process.epoch_completions,
        }
        for site, process in processes.items()
    }
    raise TimeoutError(f"Federated run exceeded {timeout}s: {progress}")


def parse_losses(path: Path, node_id: str, rounds: int, steps: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("node_id") != node_id:
                continue
            loss = float(row["loss"])
            if not math.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for {node_id}: {row}")
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "step": int(row["step"]),
                    "loss": loss,
                    "perplexity": float(row["perplexity"]),
                }
            )
    if len(rows) != rounds * steps:
        raise RuntimeError(f"Expected {rounds * steps} losses for {node_id}, got {len(rows)}")
    for index in range(rounds):
        actual = [row["step"] for row in rows[index * steps : (index + 1) * steps]]
        if actual != list(range(1, steps + 1)):
            raise RuntimeError(f"Unexpected step sequence for {node_id}, round {index + 1}")
    return rows


def audit_cache(cache_root: Path, expected_rows: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    cache_dirs = [path for path in cache_root.iterdir() if path.is_dir()]
    if len(cache_dirs) != 1:
        raise RuntimeError(f"Expected one generated cache in {cache_root}: {cache_dirs}")
    cache = cache_dirs[0]
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    parquet = list(cache.glob("*.parquet"))
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet)
    if metadata.get("num_rows") != expected_rows or rows != expected_rows:
        raise RuntimeError(
            f"Live tokenization row mismatch at {cache}: metadata={metadata.get('num_rows')}, parquet={rows}"
        )
    return {
        "path": str(cache),
        "rows": rows,
        "metadata": metadata,
        "files": directory_inventory(cache),
    }


def audit_client(
    *,
    site: str,
    runtime: Path,
    artifact_root: Path,
    settings: dict[str, Any],
    identity: dict[str, Any],
    preparation: dict[str, Any],
    process: ClientProcess,
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    rounds = settings["federation"]["rounds"]
    steps = preparation["max_steps"]
    node_id = identity["node_id"]
    losses = parse_losses(runtime / "logs/epoch_loss_tracking.csv", node_id, rounds, steps)
    state_path = runtime / "ml_states" / f"{node_id}_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("round") != rounds or len(state.get("alpha", {})) != 2:
        raise RuntimeError(f"{site} did not aggregate a two-client federation: {state}")
    cache_root = runtime / "data" / f"data_{node_id}" / "tokenized_cache"
    cache = audit_cache(cache_root, preparation["expected_train_rows"])
    if len(snapshots) != rounds or snapshots[0]["cache"] != snapshots[-1]["cache"]:
        raise RuntimeError(f"{site} token cache changed after its first federated round")
    checkpoint = runtime / "ml_models" / f"ckpt_{node_id}" / "step_0000050"
    adapter = checkpoint / "adapter_model.safetensors"
    required = [checkpoint / ".complete", checkpoint / "meta.json", adapter]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing {site} checkpoint outputs: {missing}")
    metadata = json.loads((checkpoint / "meta.json").read_text(encoding="utf-8"))
    if metadata.get("step") != steps:
        raise RuntimeError(f"Unexpected {site} checkpoint metadata: {metadata}")
    runtime_log = (runtime / "logs/runtime_comm.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if "peer_delta_loaded" not in runtime_log:
        raise RuntimeError(f"{site} never loaded a peer delta in round 2")
    if process.peer_joins < 1 or process.received_updates < rounds - 1:
        raise RuntimeError(f"Insufficient network evidence for {site}")
    node_log_text = process.log_path.read_text(encoding="utf-8", errors="replace")
    if "127.0.0.1" in "\n".join(
        line for line in node_log_text.splitlines() if "Pinned direct address" in line
    ):
        raise RuntimeError(f"{site} used a localhost pinned peer")
    result = {
        "status": "PASS",
        "site": site,
        "identity": identity,
        "preparation": preparation,
        "live_tokenization": cache,
        "round_snapshots": snapshots,
        "training": {
            "rounds": rounds,
            "steps_per_round": steps,
            "losses": losses,
            "round_summaries": [
                {
                    "round": index + 1,
                    "first_loss": losses[index * steps]["loss"],
                    "final_loss": losses[(index + 1) * steps - 1]["loss"],
                    "mean_loss": sum(
                        row["loss"] for row in losses[index * steps : (index + 1) * steps]
                    )
                    / steps,
                }
                for index in range(rounds)
            ],
        },
        "federated_state": state,
        "checkpoint": {
            "path": str(checkpoint),
            "metadata": metadata,
            "adapter_sha256": sha256_file(adapter),
            "adapter_bytes": adapter.stat().st_size,
        },
        "network": {
            "peer_joins": process.peer_joins,
            "received_updates": process.received_updates,
            "broadcasts": process.broadcasts,
            "node_log": str(process.log_path),
        },
    }
    write_json(artifact_root / site / "audit.json", result)
    return result


def prepare_site_runtime(
    *,
    site: str,
    runtime: Path,
    settings: dict[str, Any],
    native_manifest: dict[str, Any],
    slakshna_root: Path,
    rust_binary: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preparation = prepare_engine_runtime(
        site=site,
        runtime=runtime,
        slakshna_root=slakshna_root,
        native_manifest=native_manifest,
    )
    identity_path = runtime / "identity.json"
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        # Re-open the stock node briefly to prove the persisted identity still matches.
        observed = bootstrap_identity(
            site=site,
            runtime=runtime,
            settings=settings,
            rust_binary=rust_binary,
        )
        if observed["endpoint_id"] != identity["endpoint_id"] or observed["node_id"] != identity["node_id"]:
            raise RuntimeError(f"Persisted identity changed for {site}")
        identity = observed
    else:
        identity = bootstrap_identity(
            site=site,
            runtime=runtime,
            settings=settings,
            rust_binary=rust_binary,
        )
    return preparation, identity


def run_clients(
    *,
    selected_sites: tuple[str, ...],
    runtimes: dict[str, Path],
    settings: dict[str, Any],
    native_manifest: dict[str, Any],
    rust_binary: Path,
    slakshna_root: Path,
    experiment_root: Path,
    artifact_root: Path,
    local_mode: bool,
) -> None:
    artifact_root.mkdir(parents=True, exist_ok=False)
    preparations: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for site in selected_sites:
        preparations[site], identities[site] = prepare_site_runtime(
            site=site,
            runtime=runtimes[site],
            settings=settings,
            native_manifest=native_manifest,
            slakshna_root=slakshna_root,
            rust_binary=rust_binary,
        )
        write_json(artifact_root / site / "identity.json", identities[site])
        write_json(artifact_root / site / "preparation.json", preparations[site])

    probes = {}
    for site in selected_sites:
        probes[site] = tunnel_probe(
            site=site,
            client=settings["clients"][site],
            output_root=artifact_root / "network-preflight",
            experiment_root=experiment_root,
            settings=settings,
        )
    write_json(artifact_root / "playit-preflight.json", probes)

    public_ips = {
        site: resolve_public_ipv4(
            settings["clients"][site]["public_host"],
            settings["clients"][site]["public_port"],
        )
        for site in SITES
    }
    if local_mode:
        endpoint_ids = {site: identities[site]["endpoint_id"] for site in SITES}
    else:
        endpoint_ids = {
            site: str(settings["clients"][site].get("endpoint_id", "")) for site in SITES
        }
        for site, endpoint in endpoint_ids.items():
            if not re.fullmatch(r"[0-9a-f]{64}", endpoint):
                raise RuntimeError(
                    f"Set clients.{site}.endpoint_id to the exchanged Iroh EndpointId"
                )
        local_site = selected_sites[0]
        if endpoint_ids[local_site] != identities[local_site]["endpoint_id"]:
            raise RuntimeError(
                f"Configured {local_site} endpoint does not match its persisted identity"
            )

    processes: dict[str, ClientProcess] = {}
    for site in selected_sites:
        remote = "india" if site == "australia" else "australia"
        remote_client = settings["clients"][remote]
        seed = (
            f"{endpoint_ids[remote]}@{public_ips[remote]}:"
            f"{remote_client['public_port']}"
        )
        config_path = generate_node_config(
            settings=settings,
            site=site,
            runtime=runtimes[site],
            peer=seed,
            allowed_peer=endpoint_ids[remote],
        )
        shutil.copy2(config_path, artifact_root / site / "node.toml")
        client = settings["clients"][site]
        processes[site] = ClientProcess(
            site,
            [str(rust_binary), "--config", str(config_path)],
            runtimes[site],
            base_environment(client["cuda_visible_devices"]),
            artifact_root / site / "rust-node.log",
        )

    wait_until_safe_start(settings["federation"]["epoch_duration_secs"])
    try:
        for process in processes.values():
            process.start(echo=True)
        for site in selected_sites:
            wait_api(settings["clients"][site]["local_api_port"])
        wait_connected(processes, settings)
        snapshots = monitor_rounds(
            processes,
            runtimes,
            identities,
            settings["federation"]["rounds"],
            settings["federation"]["run_timeout_secs"],
        )
        api_evidence = {}
        for site in selected_sites:
            port = settings["clients"][site]["local_api_port"]
            api_evidence[site] = {
                "status": api_get(port, "/status"),
                "peers": api_get(port, "/peers"),
                "updates": api_get(port, "/updates"),
            }
            write_json(artifact_root / site / "api-evidence.json", api_evidence[site])
    finally:
        for process in processes.values():
            process.stop()

    audits = {}
    for site in selected_sites:
        audits[site] = audit_client(
            site=site,
            runtime=runtimes[site],
            artifact_root=artifact_root,
            settings=settings,
            identity=identities[site],
            preparation=preparations[site],
            process=processes[site],
            snapshots=snapshots[site],
        )
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "mode": "local-playit-rehearsal" if local_mode else "cross-country-site",
        "federation_id": settings["federation"]["id"],
        "rounds": settings["federation"]["rounds"],
        "sites": {
            site: {
                "node_id": identities[site]["node_id"],
                "endpoint_id": identities[site]["endpoint_id"],
                "tokenized_rows": audits[site]["live_tokenization"]["rows"],
                "round_summaries": audits[site]["training"]["round_summaries"],
                "received_updates": audits[site]["network"]["received_updates"],
            }
            for site in selected_sites
        },
    }
    write_json(artifact_root / "summary.json", summary)
    print("\nPHASE 9 CROSS-COUNTRIES STOCK FL PASSED", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def identity_action(settings: dict[str, Any], site: str) -> None:
    _, experiment_root, slakshna_root = repository_roots()
    native_manifest, rust_binary = native_outputs(experiment_root)
    federation_key = re.sub(r"[^A-Za-z0-9._-]+", "-", settings["federation"]["id"])
    runtime = experiment_root / ".runtime/phase9/cross-country" / federation_key / site
    preparation, identity = prepare_site_runtime(
        site=site,
        runtime=runtime,
        settings=settings,
        native_manifest=native_manifest,
        slakshna_root=slakshna_root,
        rust_binary=rust_binary,
    )
    print(json.dumps({"identity": identity, "preparation": preparation}, indent=2))
    print(f"Persisted identity: {runtime / 'identity.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("local", "site", "identity"):
        command = subparsers.add_parser(action)
        command.add_argument("--config", type=Path, required=True)
        if action != "local":
            command.add_argument("--site", choices=SITES, required=True)
    args = parser.parse_args()
    require_public = args.action != "identity"
    settings = load_settings(args.config.resolve(), require_public=require_public)
    if args.action == "identity":
        identity_action(settings, args.site)
        return

    _, experiment_root, slakshna_root = repository_roots()
    revision = subprocess.check_output(
        ["git", "-C", str(slakshna_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"Unexpected Slakshna revision: {revision}")
    native_manifest, rust_binary = native_outputs(experiment_root)
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{os.environ.get('SLURM_JOB_ID', 'interactive')}"
    artifact_root = experiment_root / "artifacts/phase9/cross-countries" / run_id
    if args.action == "local":
        selected_sites = SITES
        runtimes = {site: artifact_root / site / "runtime" for site in SITES}
        local_mode = True
    else:
        selected_sites = (args.site,)
        federation_key = re.sub(r"[^A-Za-z0-9._-]+", "-", settings["federation"]["id"])
        runtimes = {
            args.site: experiment_root
            / ".runtime/phase9/cross-country"
            / federation_key
            / args.site
        }
        local_mode = False
    run_clients(
        selected_sites=selected_sites,
        runtimes=runtimes,
        settings=settings,
        native_manifest=native_manifest,
        rust_binary=rust_binary,
        slakshna_root=slakshna_root,
        experiment_root=experiment_root,
        artifact_root=artifact_root,
        local_mode=local_mode,
    )


if __name__ == "__main__":
    main()
