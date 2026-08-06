#!/usr/bin/env python3
"""Fail-closed verifier for the two-node Phase 6 lifecycle."""
from __future__ import annotations

import argparse
import ipaddress
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


PEERS = ("peer-a", "peer-b")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def short_hostname(value: str) -> str:
    return value.split(".", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    experiment_root = Path(__file__).resolve().parents[2]

    topology = load_json(root / "cluster-topology.json")
    require(topology.get("slurm_job_num_nodes") == 2, "Phase 6 did not allocate two nodes")
    require(set(topology.get("nodes", {})) == set(PEERS), "topology peer set mismatch")
    hosts = {peer: topology["nodes"][peer]["hostname"] for peer in PEERS}
    ips = {peer: topology["nodes"][peer]["ip"] for peer in PEERS}
    require(short_hostname(hosts["peer-a"]) != short_hostname(hosts["peer-b"]), "same host used twice")
    require(ips["peer-a"] != ips["peer-b"], "same node IP used twice")
    for peer in PEERS:
        address = ipaddress.ip_address(ips[peer])
        require(address.version == 4 and not address.is_loopback, f"invalid cross-node IP for {peer}")
    require(topology.get("discovery") == {"mdns": False, "dht": False, "dns": False, "relay": False}, "public discovery was enabled")

    node_evidence: dict[str, dict[str, Any]] = {}
    visible: dict[str, str] = {}
    for peer in PEERS:
        evidence = load_json(root / peer / "node-execution.json")
        node_evidence[peer] = evidence
        require(evidence["peer_name"] == peer, f"{peer} node evidence identity mismatch")
        require(
            short_hostname(evidence["hostname"]) == short_hostname(hosts[peer]),
            f"{peer} ran on unexpected host",
        )
        require(evidence["node_ip"] == ips[peer], f"{peer} node IP evidence mismatch")
        require(evidence["gpu_count"] == 1, f"{peer} did not receive exactly one GPU")
        require("A100" in evidence["gpu_name"], f"{peer} did not run on an A100")
        cvd = evidence["cuda_visible_devices"]
        require(cvd.isdigit() and "," not in cvd, f"{peer} scheduler GPU id is not singular")
        require(evidence["gpu_id_configured"] == int(cvd), f"{peer} Rust GPU id ignored Slurm")
        require(evidence["slurm"]["SLURM_JOB_ID"] == topology["slurm_job_id"], "job id mismatch")
        visible[peer] = cvd

        config = tomllib.loads((root / peer / "runtime/node.toml").read_text(encoding="utf-8"))
        require(config["network"]["host"] == "0.0.0.0", f"{peer} API was not node-reachable")
        require(config["node"]["gpu_id"] == int(cvd), f"{peer} TOML GPU id mismatch")
        require(all(value is False for value in config["discovery"].values()), "discovery enabled")

    expected_seed = (
        f"{topology['nodes']['peer-a']['endpoint_id']}@"
        f"{ips['peer-a']}:{topology['nodes']['peer-a']['p2p_port']}"
    )
    require(topology.get("direct_seed") == expected_seed, "direct seed evidence mismatch")
    require(node_evidence["peer-a"]["seed_peer"] is None, "Peer A unexpectedly had a seed")
    require(node_evidence["peer-b"]["seed_peer"] == expected_seed, "Peer B used wrong seed")
    peer_b_log = (root / "peer-b/rust-node.log").read_text(encoding="utf-8", errors="replace")
    require(f"Pinned direct address" in peer_b_log and ips["peer-a"] in peer_b_log, "direct cross-node address was not pinned")
    for peer in PEERS:
        log = (root / peer / "rust-node.log").read_text(encoding="utf-8", errors="replace")
        require("Discovery: mdns=false dht=false dns=false relay=false" in log, f"{peer} discovery log mismatch")

    base_summary_path = root / "phase6-core-summary.json"
    command = [
        sys.executable,
        str(experiment_root / "src/phase5/verify_phase5.py"),
        "--artifact-root",
        str(root),
        "--output",
        str(base_summary_path),
        "--mode",
        "phase6-core",
        "--expected-visible-a",
        visible["peer-a"],
        "--expected-visible-b",
        visible["peer-b"],
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Phase 5 core contract failed:\n{completed.stdout}\n{completed.stderr}"
        )
    base_summary = load_json(base_summary_path)
    require(base_summary.get("status") == "PASS", "core verifier did not pass")

    residue: dict[str, dict[str, Any]] = {}
    for peer in PEERS:
        payload = load_json(root / peer / "process-residue.json")
        require(payload.get("residue") == [], f"{peer} left allocation-scoped processes")
        require(
            short_hostname(payload["hostname"]) == short_hostname(hosts[peer]),
            f"{peer} residue probe ran on wrong node",
        )
        residue[peer] = payload

    checks = base_summary["checks"]
    checks["cross_node_slurm_transport"] = {
        "status": "PASS",
        "detail": {
            "slurm_job_id": topology["slurm_job_id"],
            "hosts": hosts,
            "ips": ips,
            "scheduler_visible_gpus": visible,
            "direct_seed": expected_seed,
            "public_discovery": topology["discovery"],
        },
    }
    checks["process_cleanup"] = {
        "status": "PASS",
        "detail": {peer: {"hostname": residue[peer]["hostname"], "residue": []} for peer in PEERS},
    }
    summary = {"status": "PASS", "mode": "phase6", "checks": checks, "failures": []}
    write_json(args.output.resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("PHASE6 PASSED")


if __name__ == "__main__":
    main()
