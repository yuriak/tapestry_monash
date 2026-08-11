from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
VERIFY = EXPERIMENT_ROOT / "src/phase8/verify_pair.py"


def envelope(site: str, node: str, number: int, base: str, marker: str) -> dict:
    return {
        "format": "slakshna-phase8-dense-delta", "version": 1,
        "codec": "base64+zlib+safetensors-fp32", "sender_site": site,
        "sender_node_id": node, "round": number, "base_state_sha256": base,
        "delta_file_sha256": marker * 64, "delta_state_sha256": marker * 64,
        "raw_bytes": 100, "compressed_bytes": 80, "tensor_count": 2,
        "parameter_count": 10,
    }


def site_audit(site: str) -> dict:
    other = "site-b" if site == "site-a" else "site-a"
    node = "endpoint-a" if site == "site-a" else "endpoint-b"
    peer = "endpoint-b" if site == "site-a" else "endpoint-a"
    prior = "0" * 64
    rounds = []
    for number in range(1, 6):
        marker_self = "a" if site == "site-a" else "b"
        marker_peer = "b" if site == "site-a" else "a"
        global_hash = str(number) * 64
        rounds.append({
            "round": number,
            "base_state_sha256": prior,
            "outbound": {**envelope(site, node, number, prior, marker_self), "wire_bytes": 120},
            "received": {**envelope(other, peer, number, prior, marker_peer), "transport_node_id": peer},
            "aggregation": {"global_state_sha256": global_hash},
            "training": {},
        })
        prior = global_hash
    indices = [1, 2] if site == "site-a" else [3, 4]
    return {
        "format": "slakshna-phase8-site-audit", "version": 1, "status": "PASS",
        "site": site, "node_id": node, "peer_node_id": peer,
        "dataset": {"id": "dataset", "revision": "revision"},
        "partition_type": "disjoint-category-non-iid",
        "shard": {"categories": [site], "category_counts": {},
                  "train": {"source_indices": indices[:1]},
                  "validation": {"source_indices": indices[1:]}},
        "training_contract": {"contract": 1},
        "training_contract_sha256": "c" * 64,
        "g0": {"adapter_state_sha256": "0" * 64},
        "rounds": rounds,
        "totals": {"training_rounds": 5, "local_epochs": 50, "optimizer_steps": 3600},
        "final_global_state_sha256": prior,
    }


class PairVerifierTests(unittest.TestCase):
    def run_verifier(self, root: Path) -> subprocess.CompletedProcess[str]:
        environment = {**os.environ, "PYTHONPATH": str(EXPERIMENT_ROOT / "src")}
        return subprocess.run(
            [sys.executable, str(VERIFY), "--site-a-audit", str(root / "a.json"),
             "--site-b-audit", str(root / "b.json"), "--output", str(root / "pair.json")],
            text=True, capture_output=True, env=environment, check=False,
        )

    def test_accepts_reciprocal_five_round_audits_and_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase8-pair-") as temporary:
            root = Path(temporary)
            a, b = site_audit("site-a"), site_audit("site-b")
            (root / "a.json").write_text(json.dumps(a))
            (root / "b.json").write_text(json.dumps(b))
            passed = self.run_verifier(root)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn("PHASE8 PAIRED AUDIT PASSED", passed.stdout)
            b["rounds"][2]["received"]["delta_file_sha256"] = "f" * 64
            (root / "b.json").write_text(json.dumps(b))
            failed = self.run_verifier(root)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("envelope mismatch", failed.stderr)


if __name__ == "__main__":
    unittest.main()
