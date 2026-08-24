#!/usr/bin/env python3
"""Run one site of the two-node M0 local FL job inside a Slurm task."""

from __future__ import annotations

import atexit
import ipaddress
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import toml
import tomllib
import torch

IDENTITY_RE = re.compile(r"Node Identity:\s+(slakshna1[0-9a-z]+)")
ENDPOINT_RE = re.compile(r"^[0-9a-f]{64}$")
FAILURE_MARKERS = ("Python ML Engine failed", "panicked at", "Traceback (most recent call last)")
SITES = ("au", "india")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def api_get(port: int, endpoint: str = "/status") -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{endpoint}", timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_api(port: int, timeout: int = 60) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return api_get(port)
        except Exception as error:  # endpoint is not ready yet
            last_error = error
            time.sleep(0.25)
    raise TimeoutError(f"API on port {port} did not become ready: {last_error}")


class NodeProcess:
    def __init__(self, command: list[str], cwd: Path, log_path: Path):
        self.command = command
        self.cwd = cwd
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.node_id: str | None = None
        self.local_completions = 0
        self.epoch_completions = 0
        self.peer_joins = 0
        self.received_updates = 0
        self.failures: list[str] = []

    def start(self, echo: bool) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w", encoding="utf-8", buffering=1)
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=os.environ.copy(),
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
                    log.write(line)
                    if echo:
                        print(line, end="", flush=True)
                    match = IDENTITY_RE.search(line)
                    if match:
                        self.node_id = match.group(1)
                    self.local_completions += line.count("Local Training Complete!")
                    self.epoch_completions += line.count("Most-trusted cohort this epoch")
                    self.peer_joins += line.count("Peer joined gossip mesh")
                    self.received_updates += line.count("Gossiped model update received")
                    for marker in FAILURE_MARKERS:
                        if marker in line:
                            self.failures.append(line.strip())
            finally:
                log.close()

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


def wait_for_file(path: Path, timeout: int = 300) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for shared state: {path}")


def load_playit_site(workspace: Path, site: str, local_port: int) -> dict[str, Any]:
    config_path = Path(os.environ["M0_FL_PLAYIT_CONFIG"])
    if not config_path.is_absolute():
        config_path = workspace / config_path
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    value = config.get(site)
    if not isinstance(value, dict):
        raise RuntimeError(f"missing [{site}] in Playit config: {config_path}")
    host = str(value.get("public_host", "")).strip()
    public_port = int(value.get("public_port", 0))
    configured_local_port = int(value.get("local_port", 0))
    secret_path = Path(str(value.get("secret_path", "")))
    if not secret_path.is_absolute():
        secret_path = workspace / secret_path
    if not host or not 1024 <= public_port <= 65535:
        raise RuntimeError(f"invalid Playit public endpoint for {site}")
    if configured_local_port != local_port:
        raise RuntimeError(
            f"Playit tunnel for {site} must target 127.0.0.1:{local_port}; "
            f"config declares {configured_local_port}"
        )
    if not secret_path.is_file() or secret_path.stat().st_size == 0:
        raise RuntimeError(f"Playit agent secret is missing for {site}: {secret_path}")
    if secret_path.stat().st_mode & 0o077:
        raise RuntimeError(f"Playit agent secret must have mode 600: {secret_path}")
    addresses = sorted(
        {
            item[4][0]
            for item in socket.getaddrinfo(
                host, public_port, socket.AF_INET, socket.SOCK_DGRAM
            )
        }
    )
    public_addresses = [
        address for address in addresses if ipaddress.ip_address(address).is_global
    ]
    if not public_addresses or len(public_addresses) != len(addresses):
        raise RuntimeError(
            f"Playit endpoint for {site} must resolve only to global IPv4: {host} -> {addresses}"
        )
    return {
        "config_path": str(config_path.resolve()),
        "public_host": host,
        "public_port": public_port,
        "public_ipv4": public_addresses[0],
        "local_port": local_port,
        "secret_path": str(secret_path.resolve()),
    }


class PlayitProcess:
    def __init__(self, workspace: Path, runtime: Path, site: str, settings: dict[str, Any]):
        runtime_tools = workspace / "monash_exps/.runtime/tools/playit/bin"
        self.daemon = runtime_tools / "playitd"
        self.cli = runtime_tools / "playit"
        self.site = site
        self.settings = settings
        self.socket_path = Path(
            f"/tmp/m0-fl-playit-{os.getuid()}-{os.environ.get('SLURM_JOB_ID', 'local')}-{site}.sock"
        )
        self.log_path = runtime / "playit-agent.log"
        self.stdio_path = runtime / "playit-agent-stdio.log"
        self.process: subprocess.Popen[bytes] | None = None
        self.handle: Any = None

    def start(self) -> None:
        if not self.daemon.is_file() or not self.cli.is_file():
            raise RuntimeError("pinned Playit binaries are missing from monash_exps/.runtime/tools")
        self.socket_path.unlink(missing_ok=True)
        self.handle = self.stdio_path.open("wb", buffering=0)
        self.process = subprocess.Popen(
            [
                str(self.daemon),
                "--secret-path",
                self.settings["secret_path"],
                "--socket-path",
                str(self.socket_path),
                "--log-path",
                str(self.log_path),
            ],
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(120):
            if self.process.poll() is not None:
                raise RuntimeError(f"Playit agent for {self.site} exited during startup")
            if self.socket_path.is_socket():
                try:
                    status = subprocess.run(
                        [str(self.cli), "--socket-path", str(self.socket_path), "status"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    (self.log_path.parent / "playit-status.txt").write_text(
                        status.stdout + status.stderr, encoding="utf-8"
                    )
                    log_text = self.log_path.read_text(encoding="utf-8", errors="replace")
                    counts = [
                        int(value)
                        for value in re.findall(
                            r"playit connected; tunnels loaded.*?tunnel_count=(\d+)",
                            log_text,
                        )
                    ]
                    if counts and counts[-1] >= 1:
                        return
                except (subprocess.SubprocessError, OSError):
                    pass
            time.sleep(1)
        raise TimeoutError(f"Playit agent for {self.site} did not load its assigned tunnel")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if self.process.poll() is None:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=5)
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.socket_path.unlink(missing_ok=True)


def probe_own_playit_tunnel(
    python: Path, experiment: Path, runtime: Path, settings: dict[str, Any]
) -> dict[str, Any]:
    token = f"m0-fl-{secrets.token_hex(12)}"
    script = experiment / "src/phase8/udp_probe.py"
    server_json = runtime / "playit-probe-server.json"
    client_json = runtime / "playit-probe-client.json"
    with (runtime / "playit-probe-server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [
                str(python), str(script), "server", "--bind", "127.0.0.1",
                "--port", str(settings["local_port"]), "--token", token,
                "--wait", "90", "--output", str(server_json),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            time.sleep(1)
            subprocess.run(
                [
                    str(python), str(script), "client",
                    "--host", settings["public_host"],
                    "--port", str(settings["public_port"]),
                    "--token", token, "--attempts", "20", "--timeout", "2",
                    "--interval", "1", "--output", str(client_json),
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
    return {
        "server": json.loads(server_json.read_text(encoding="utf-8")),
        "client": json.loads(client_json.read_text(encoding="utf-8")),
    }


def bootstrap_config(runtime: Path, federation_id: str, ports: dict[str, int]) -> Path:
    config = {
        "federation": {"id": federation_id, "name": "M0 Local FL bootstrap"},
        "training": {"epoch_duration_secs": 780, "sync_deadline_secs": 720, "expected_peers": 2},
        "compression": {
            "enabled": True,
            "sparsity": 0.1,
            "quantization": "symmetric_int8",
            "allow_legacy_delta_format": False,
            "max_payload_bytes": 7_340_032,
            "max_tensor_elements": 10_000_000,
        },
        "node": {
            "id": "m0-fl-bootstrap",
            "data_dir": str((runtime / "node-state").resolve()),
            "gpu_id": 0,
            "num_gpus": 2,
        },
        "network": {
            "host": "0.0.0.0",
            "p2p_port": ports["p2p"],
            "ws_port": ports["ws"],
            "api_port": ports["api"],
            "peers": [],
            "allowed_peers": [],
        },
        "discovery": {"mdns": False, "dht": False, "dns": False, "relay": False},
        "logging": {"level": "info"},
    }
    path = runtime / "bootstrap-node.toml"
    path.write_text(toml.dumps(config), encoding="utf-8")
    return path


def bootstrap_identity(
    runtime: Path,
    federation_id: str,
    rust_binary: Path,
    ports: dict[str, int],
) -> tuple[str, str]:
    config = bootstrap_config(runtime, federation_id, ports)
    node = NodeProcess(
        [str(rust_binary), "--config", str(config)],
        runtime,
        runtime / "identity-bootstrap.log",
    )
    node.start(echo=False)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and node.node_id is None:
            assert node.process is not None
            if node.process.poll() is not None:
                raise RuntimeError("Slakshna exited before creating its identity")
            time.sleep(0.1)
        if node.node_id is None:
            raise TimeoutError("Slakshna identity was not reported")
        status = wait_api(ports["api"])
        endpoint = status.get("endpoint_id")
        if not isinstance(endpoint, str) or not ENDPOINT_RE.fullmatch(endpoint):
            raise RuntimeError(f"invalid Iroh EndpointId: {endpoint!r}")
        return node.node_id, endpoint
    finally:
        node.stop()


def gpu_and_cpu_inventory() -> dict[str, Any]:
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected two visible GPUs, found {torch.cuda.device_count()}")
    gpus = []
    for index in range(2):
        props = torch.cuda.get_device_properties(index)
        if "A100" not in props.name or props.total_memory < 75_000 * 1024 * 1024:
            raise RuntimeError(f"GPU {index} is not an 80 GB A100: {props.name}, {props.total_memory}")
        gpus.append({"index": index, "name": props.name, "bytes": props.total_memory})
    return {
        "hostname": socket.gethostname(),
        "ipv4": socket.gethostbyname(socket.gethostname()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpus,
        "os_cpu_count": os.cpu_count(),
        "affinity_cpu_count": len(os.sched_getaffinity(0)),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    }


def start_gpu_monitor(output: Path) -> tuple[subprocess.Popen[bytes], Any]:
    handle = output.open("ab", buffering=0)
    process = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
            "--loop=2",
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, handle


def main() -> int:
    rank = int(os.environ.get("SLURM_PROCID", "-1"))
    if rank not in (0, 1):
        raise RuntimeError("run_formal_site.py requires exactly two Slurm tasks")
    site = SITES[rank]
    peer_site = SITES[1 - rank]
    workspace = Path(os.environ["M0_FL_WORKSPACE"]).resolve()
    run_root = Path(os.environ["M0_FL_RUN_ROOT"]).resolve()
    federation_id = os.environ["M0_FL_FEDERATION_ID"]
    experiment = workspace / "monash_exps"
    python = experiment / ".runtime/venvs/primary/bin/python"
    rust_binary = experiment / ".runtime/cargo-target/slakshna/release/iiitd"
    runtime = run_root / site
    runtime.mkdir(parents=True, exist_ok=False)
    ports = {"p2p": 39080, "api": 39401, "ws": 39402}

    os.environ.update(
        {
            "SLAKSHNA_NUM_GPUS": "2",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "VIRTUAL_ENV": str(python.parent.parent),
            "PATH": f"{python.parent}:{os.environ['PATH']}",
        }
    )
    inventory = gpu_and_cpu_inventory()
    print(f"[{site}] allocation: {json.dumps(inventory, sort_keys=True)}", flush=True)

    playit_sites = {
        name: load_playit_site(workspace, name, ports["p2p"]) for name in SITES
    }
    playit = PlayitProcess(workspace, runtime, site, playit_sites[site])
    playit.start()
    atexit.register(playit.stop)
    playit_probe = probe_own_playit_tunnel(
        python, experiment, runtime, playit_sites[site]
    )
    atomic_json(
        runtime / "playit-preflight.json",
        {
            "site": site,
            "public_host": playit_sites[site]["public_host"],
            "public_ipv4": playit_sites[site]["public_ipv4"],
            "public_port": playit_sites[site]["public_port"],
            "local_port": playit_sites[site]["local_port"],
            "probe": playit_probe,
        },
    )
    print(
        f"[{site}] Playit UDP preflight passed via "
        f"{playit_sites[site]['public_host']}:{playit_sites[site]['public_port']}",
        flush=True,
    )

    node_id, endpoint = bootstrap_identity(runtime, federation_id, rust_binary, ports)
    identity = {
        "site": site,
        "node_id": node_id,
        "endpoint_id": endpoint,
        "playit_public_ipv4": playit_sites[site]["public_ipv4"],
        "playit_public_port": playit_sites[site]["public_port"],
        **inventory,
    }
    atomic_json(runtime / "identity.json", identity)
    peer = wait_for_file(run_root / peer_site / "identity.json")
    print(f"[{site}] identity exchange complete with {peer_site}", flush=True)

    subprocess.run(
        [
            str(python),
            "-m",
            "monash_exps.src.m0_fl.prepare_site_runtime",
            "--site",
            site,
            "--runtime",
            str(runtime),
            "--node-id",
            node_id,
            "--peer-endpoint",
            peer["endpoint_id"],
            "--peer-public-ip",
            playit_sites[peer_site]["public_ipv4"],
            "--p2p-port",
            str(ports["p2p"]),
            "--api-port",
            str(ports["api"]),
            "--ws-port",
            str(ports["ws"]),
            "--peer-p2p-port",
            str(playit_sites[peer_site]["public_port"]),
            "--federation-id",
            federation_id,
        ],
        cwd=workspace,
        check=True,
        stdout=(runtime / "preparation.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    atomic_json(runtime / "READY.json", {"site": site, "ready_at": datetime.now().astimezone().isoformat()})
    wait_for_file(run_root / peer_site / "READY.json")

    launch_path = run_root / "LAUNCH.json"
    if rank == 0:
        now = int(time.time())
        remaining = 780 - (now % 780)
        launch_at = now + 5 if remaining >= 120 else now + remaining + 5
        atomic_json(launch_path, {"launch_at": launch_at})
    launch = wait_for_file(launch_path)
    while time.time() < launch["launch_at"]:
        time.sleep(0.1)

    monitor, monitor_handle = start_gpu_monitor(runtime / "gpu.csv")
    node = NodeProcess(
        [str(rust_binary), "--config", str(runtime / "node.toml")],
        runtime,
        runtime / "rust-node.log",
    )
    node.start(echo=True)
    try:
        status = wait_api(ports["api"], timeout=120)
        if status.get("endpoint_id") != endpoint:
            raise RuntimeError("persisted Iroh EndpointId changed after bootstrap")
        deadline = time.monotonic() + 10_800
        while time.monotonic() < deadline:
            assert node.process is not None
            if node.failures:
                raise RuntimeError("; ".join(node.failures[-3:]))
            if node.process.poll() is not None:
                raise RuntimeError(f"Slakshna exited early with status {node.process.returncode}")
            if node.epoch_completions >= 10:
                break
            time.sleep(1)
        else:
            raise TimeoutError(
                f"site timed out: local={node.local_completions}, epochs={node.epoch_completions}"
            )
        time.sleep(5)
        evidence = {
            "status": api_get(ports["api"], "/status"),
            "peers": api_get(ports["api"], "/peers"),
            "updates": api_get(ports["api"], "/updates"),
        }
    finally:
        node.stop()
        monitor.terminate()
        try:
            monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait(timeout=5)
        monitor_handle.close()

    if node.local_completions < 10 or node.epoch_completions < 10:
        raise RuntimeError(
            f"incomplete FL run: local={node.local_completions}, epochs={node.epoch_completions}"
        )
    completion = {
        "status": "COMPLETED",
        "site": site,
        "completed_at": datetime.now().astimezone().isoformat(),
        "local_training_completions": node.local_completions,
        "federated_epoch_completions": node.epoch_completions,
        "peer_joins": node.peer_joins,
        "received_updates": node.received_updates,
        "api": evidence,
    }
    atomic_json(runtime / "COMPLETED.json", completion)
    wait_for_file(run_root / peer_site / "COMPLETED.json", timeout=300)
    if rank == 0:
        atomic_json(
            run_root / "COMPLETED.json",
            {
                "status": "COMPLETED",
                "federation_id": federation_id,
                "completed_at": datetime.now().astimezone().isoformat(),
                "sites": list(SITES),
            },
        )
    print(f"[{site}] M0 local FL completed", flush=True)
    playit.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
