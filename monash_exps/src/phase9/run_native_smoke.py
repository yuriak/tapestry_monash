#!/usr/bin/env python3
"""Run and audit one stock Slakshna Phase 9 smoke-training epoch."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import tomllib
import yaml

EXPECTED_SLAKSHNA_REVISION = "9f93ec45ae0d3eb9c901aff3b50d4325b5050488"
IDENTITY_RE = re.compile(r"Node Identity:\s+(slakshna1[0-9a-z]+)")
LOCAL_COMPLETE = "Local Training Complete!"
EPOCH_COMPLETE = "Most-trusted cohort this epoch"
FAILURE_MARKERS = (
    "Python ML Engine failed",
    "Failed to start Python process",
    "panicked at",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def allocate_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(int(sock.getsockname()[1]))
    finally:
        for sock in sockets:
            sock.close()
    return ports


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def dump_flat_toml(config: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, values in config.items():
        if not isinstance(values, dict):
            raise TypeError(f"Expected TOML table for {section!r}")
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {toml_value(value)}")
    return "\n".join(lines) + "\n"


def generate_config(
    template: Path,
    output: Path,
    run_id: str,
    data_dir: Path,
    ports: list[int],
    epoch_duration: int,
) -> dict[str, Any]:
    with template.open("rb") as handle:
        config = tomllib.load(handle)
    config["federation"]["id"] = f"slakshna-phase9-native-smoke-{run_id}"
    config["federation"]["name"] = f"Slakshna Phase 9 Native Smoke {run_id}"
    config["training"]["epoch_duration_secs"] = epoch_duration
    config["training"]["sync_deadline_secs"] = max(1, epoch_duration - 15)
    config["training"]["expected_peers"] = 1
    config["node"]["id"] = "phase9-native-smoke"
    config["node"]["data_dir"] = str(data_dir.resolve())
    config["node"]["gpu_id"] = 0
    config["node"]["num_gpus"] = 1
    config["network"]["host"] = "127.0.0.1"
    config["network"]["p2p_port"] = ports[0]
    config["network"]["ws_port"] = ports[1]
    config["network"]["api_port"] = ports[2]
    config["network"]["peers"] = []
    config["discovery"] = {"mdns": False, "dht": False, "dns": False, "relay": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dump_flat_toml(config), encoding="utf-8")
    with output.open("rb") as handle:
        reparsed = tomllib.load(handle)
    if reparsed != config:
        raise RuntimeError("Generated stock node configuration failed round-trip audit")
    return config


class NodeRun:
    def __init__(self, command: list[str], cwd: Path, env: dict[str, str], log: Path):
        self.command = command
        self.cwd = cwd
        self.env = env
        self.log = log
        self.identity: str | None = None
        self.local_complete = False
        self.epoch_complete = False
        self.failures: list[str] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def start(self, echo: bool = True) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self.log.open("w", encoding="utf-8", buffering=1)
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
                        print(line, end="", flush=True)
                    match = IDENTITY_RE.search(line)
                    if match:
                        self.identity = match.group(1)
                    if LOCAL_COMPLETE in line:
                        self.local_complete = True
                    if EPOCH_COMPLETE in line:
                        self.epoch_complete = True
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
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
        if self.thread is not None:
            self.thread.join(timeout=5)

    def wait_for_identity(self, timeout: int) -> str:
        assert self.process is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.identity:
                return self.identity
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(f"Node exited before identity initialization: {code}")
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for the stock Slakshna identity")


def cache_inventory(cache_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(cache_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(cache_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not files or not any(item["path"].endswith(".parquet") for item in files):
        raise RuntimeError(f"Cache contains no Parquet data: {cache_dir}")
    return {"path": str(cache_dir), "files": files}


def stage_cache(source: Path, destination_root: Path) -> tuple[Path, dict[str, Any]]:
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source.name
    if destination.exists():
        raise RuntimeError(f"Fresh peer unexpectedly has an existing cache: {destination}")
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    source_inventory = cache_inventory(source)
    destination_inventory = cache_inventory(destination)
    source_signature = [(x["path"], x["bytes"], x["sha256"]) for x in source_inventory["files"]]
    destination_signature = [
        (x["path"], x["bytes"], x["sha256"]) for x in destination_inventory["files"]
    ]
    if source_signature != destination_signature:
        raise RuntimeError("Staged token cache differs from its audited source")
    return destination, destination_inventory


def read_api(port: int, endpoint: str, timeout: float = 3.0) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{endpoint}", timeout=timeout
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_api(port: int, endpoint: str, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return read_api(port, endpoint)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(0.25)
    raise TimeoutError(f"API {endpoint} did not become ready: {last_error}")


def follow_file(path: Path, stop_event: threading.Event) -> None:
    offset = 0
    while not stop_event.wait(0.5):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            content = handle.read()
            offset = handle.tell()
        if content:
            for line in content.splitlines():
                print(f"[Bhaskera] {line}", flush=True)


def wait_for_training(run: NodeRun, crash_log: Path, timeout: int) -> None:
    assert run.process is not None
    stop_tail = threading.Event()
    tail_thread = threading.Thread(
        target=follow_file, args=(crash_log, stop_tail), daemon=True
    )
    tail_thread.start()
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if run.failures:
                raise RuntimeError("; ".join(run.failures))
            if run.epoch_complete:
                if not run.local_complete:
                    raise RuntimeError(
                        "Rust epoch completed without accepting a valid ML-engine result"
                    )
                return
            code = run.process.poll()
            if code is not None:
                raise RuntimeError(f"Stock Slakshna node exited unexpectedly: {code}")
            time.sleep(0.25)
        raise TimeoutError(f"Native smoke training exceeded {timeout} seconds")
    finally:
        stop_tail.set()
        tail_thread.join(timeout=3)


def parse_losses(loss_csv: Path, peer_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with loss_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("node_id") != peer_id:
                continue
            loss = float(row["loss"])
            perplexity = float(row["perplexity"])
            if not math.isfinite(loss) or not math.isfinite(perplexity):
                raise RuntimeError(f"Non-finite training metric: {row}")
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "epoch": int(row["epoch"]),
                    "step": int(row["step"]),
                    "loss": loss,
                    "perplexity": perplexity,
                }
            )
    if [row["step"] for row in rows] != list(range(1, 21)):
        raise RuntimeError(f"Expected loss steps 1..20, got {[row['step'] for row in rows]}")
    return rows


def audit_adapter(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    tensors = load_file(path)
    if not tensors:
        raise RuntimeError(f"Adapter contains no tensors: {path}")
    nonfinite = []
    nonzero = 0
    parameters = 0
    for name, tensor in tensors.items():
        parameters += tensor.numel()
        if not tensor.isfinite().all():
            nonfinite.append(name)
        if tensor.count_nonzero().item() > 0:
            nonzero += 1
    if nonfinite or nonzero == 0:
        raise RuntimeError(
            f"Invalid adapter: nonfinite={nonfinite}, nonzero_tensors={nonzero}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "tensors": len(tensors),
        "parameters": parameters,
        "nonzero_tensors": nonzero,
    }


def audit_run(
    *,
    slakshna_root: Path,
    run_root: Path,
    peer_id: str,
    staged_cache: Path,
    staged_before: dict[str, Any],
    node_log: Path,
    api_status: dict[str, Any],
    api_update: dict[str, Any],
    config_path: Path,
    started_at: float,
) -> dict[str, Any]:
    staged_after = cache_inventory(staged_cache)
    if staged_before["files"] != staged_after["files"]:
        raise RuntimeError("Offline token cache changed during training")
    cache_children = [path for path in staged_cache.parent.iterdir() if path.is_dir()]
    if cache_children != [staged_cache]:
        raise RuntimeError(f"Unexpected cache directories after training: {cache_children}")

    effective_config_path = slakshna_root / f"config_{peer_id}.yaml"
    effective = yaml.safe_load(effective_config_path.read_text(encoding="utf-8"))
    if Path(effective["data"]["tokenized_path"]).resolve() != staged_cache.resolve():
        raise RuntimeError("Stock ML engine did not select the staged offline cache")

    loss_rows = parse_losses(slakshna_root / "logs" / "epoch_loss_tracking.csv", peer_id)
    checkpoint_root = slakshna_root / "ml_models" / f"ckpt_{peer_id}"
    checkpoint = checkpoint_root / "step_0000020"
    required = [
        checkpoint / ".complete",
        checkpoint / "meta.json",
        checkpoint / "adapter_model.safetensors",
        slakshna_root / "ml_models" / f"{peer_id}_base_lora.pth",
        slakshna_root / "ml_models" / f"{peer_id}_delta.pth",
        slakshna_root / "ml_states" / f"{peer_id}_state.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Native smoke outputs are missing: {missing}")
    checkpoint_meta = json.loads((checkpoint / "meta.json").read_text(encoding="utf-8"))
    if checkpoint_meta.get("step") != 20:
        raise RuntimeError(f"Unexpected checkpoint metadata: {checkpoint_meta}")
    state = json.loads(required[-1].read_text(encoding="utf-8"))
    if state.get("round") != 1 or not math.isfinite(float(state.get("score", math.nan))):
        raise RuntimeError(f"Invalid ML-engine state: {state}")

    update = api_update.get("update", {})
    update_kind = update.get("kind", {})
    model_update = update_kind.get("ModelUpdate", update_kind.get("model_update", {}))
    compressed = model_update.get("compressed_delta", "") if isinstance(model_update, dict) else ""
    delta_hash = model_update.get("delta_hash", "") if isinstance(model_update, dict) else ""
    if not api_update.get("success") or not compressed or delta_hash == "error_hash":
        raise RuntimeError("Rust API does not expose a valid accepted model update")

    node_text = node_log.read_text(encoding="utf-8", errors="replace")
    if LOCAL_COMPLETE not in node_text or EPOCH_COMPLETE not in node_text:
        raise RuntimeError("Stock Rust completion markers are absent")
    if any(marker in node_text for marker in FAILURE_MARKERS):
        raise RuntimeError("Stock Rust log contains an ML-engine failure marker")

    adapter = audit_adapter(checkpoint / "adapter_model.safetensors")
    result = {
        "schema_version": 1,
        "status": "PASS",
        "run_id": run_root.name,
        "peer_id": peer_id,
        "slakshna_revision": EXPECTED_SLAKSHNA_REVISION,
        "execution": {
            "elapsed_seconds": round(time.time() - started_at, 3),
            "node_log": str(node_log),
            "config": str(config_path),
            "effective_bhaskera_config": str(effective_config_path),
        },
        "offline_cache": {
            "status": "PASS",
            "path": str(staged_cache),
            "unchanged_during_training": True,
            "files": staged_after["files"],
        },
        "training": {
            "steps": len(loss_rows),
            "first_loss": loss_rows[0]["loss"],
            "final_loss": loss_rows[-1]["loss"],
            "minimum_loss": min(row["loss"] for row in loss_rows),
            "losses": loss_rows,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "metadata": checkpoint_meta,
            "adapter": adapter,
            "base_lora": str(required[3]),
            "local_delta": str(required[4]),
            "ml_engine_state": state,
        },
        "rust_acceptance": {
            "status": "PASS",
            "api_status": api_status,
            "record_hash": update.get("hash"),
            "delta_hash": delta_hash,
            "compressed_delta_base64_bytes": len(compressed),
        },
        "known_upstream_observation": {
            "rust_status_round": api_status.get("round"),
            "note": (
                "The stock Rust State round remains unchanged because main.rs does not "
                "call State::set_round; the ML-engine state independently completed round 1."
            ),
        },
    }
    write_json(run_root / "audit.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slakshna-root", type=Path, required=True)
    parser.add_argument("--rust-binary", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--epoch-duration", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    slakshna_root = args.slakshna_root.resolve()
    rust_binary = args.rust_binary.resolve()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    if args.epoch_duration < 30:
        raise SystemExit("--epoch-duration must be at least 30 seconds")
    if args.timeout < 300:
        raise SystemExit("--timeout must be at least 300 seconds")

    revision = subprocess.check_output(
        ["git", "-C", str(slakshna_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_SLAKSHNA_REVISION:
        raise RuntimeError(f"Unexpected Slakshna revision: {revision}")
    native_manifest = json.loads(args.native_manifest.read_text(encoding="utf-8"))
    if native_manifest.get("status") != "PASS":
        raise RuntimeError("Phase 9 native preparation manifest is not PASS")
    profile = native_manifest["profiles"]["australia-smoke"]
    source_cache = Path(profile["train_cache"]["path"]).resolve()

    ports = allocate_ports(3)
    node_data = run_root / "node-data"
    config_path = run_root / "node.toml"
    config = generate_config(
        slakshna_root / "configs" / "phase9" / "node-smoke.toml",
        config_path,
        run_root.name,
        node_data,
        ports,
        args.epoch_duration,
    )
    write_json(run_root / "resolved-config.json", config)

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    rust_ld = environment.get("PHASE9_RUST_LD_LIBRARY_PATH")
    if rust_ld is not None:
        environment["LD_LIBRARY_PATH"] = rust_ld
    command = [str(rust_binary), "--config", str(config_path)]

    print("=== Initialize the stock Slakshna identity ===", flush=True)
    bootstrap = NodeRun(command, slakshna_root, environment, run_root / "bootstrap.log")
    bootstrap.start(echo=True)
    try:
        peer_id = bootstrap.wait_for_identity(timeout=60)
    finally:
        bootstrap.stop()
    print(f"Stable peer identity: {peer_id}", flush=True)

    cache_root = slakshna_root / "data" / f"data_{peer_id}" / "tokenized_cache"
    staged_cache, staged_inventory = stage_cache(source_cache, cache_root)
    write_json(
        run_root / "cache-deployment.json",
        {
            "source": str(source_cache),
            "destination": str(staged_cache),
            "inventory": staged_inventory,
        },
    )
    print(f"Offline cache staged at: {staged_cache}", flush=True)

    print("=== Start one complete stock Slakshna training epoch ===", flush=True)
    node_log = run_root / "rust-node.log"
    native = NodeRun(command, slakshna_root, environment, node_log)
    native.start(echo=True)
    api_status: dict[str, Any] = {}
    api_update: dict[str, Any] = {}
    crash_log = slakshna_root / "logs" / f"{peer_id}_bhaskera_crash.log"
    try:
        api_status_initial = wait_api(ports[2], "/status", timeout=60)
        write_json(run_root / "api-status.initial.json", api_status_initial)
        wait_for_training(native, crash_log, args.timeout)
        api_status = wait_api(ports[2], "/status", timeout=15)
        api_update = wait_api(ports[2], "/updates/latest", timeout=15)
        write_json(run_root / "api-status.final.json", api_status)
        write_json(run_root / "api-update.latest.json", api_update)
    finally:
        native.stop()

    result = audit_run(
        slakshna_root=slakshna_root,
        run_root=run_root,
        peer_id=peer_id,
        staged_cache=staged_cache,
        staged_before=staged_inventory,
        node_log=node_log,
        api_status=api_status,
        api_update=api_update,
        config_path=config_path,
        started_at=started_at,
    )
    summary = {
        "status": result["status"],
        "run_root": str(run_root),
        "peer_id": peer_id,
        "steps": result["training"]["steps"],
        "first_loss": result["training"]["first_loss"],
        "final_loss": result["training"]["final_loss"],
        "adapter_sha256": result["checkpoint"]["adapter"]["sha256"],
        "compressed_delta_base64_bytes": result["rust_acceptance"][
            "compressed_delta_base64_bytes"
        ],
        "audit": str(run_root / "audit.json"),
    }
    print("\nPHASE 9 STOCK NATIVE SMOKE PASSED", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
