from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
from phase2.adapter_delta import max_abs_error, state_sha256  # noqa: E402
from phase8.protocol import aggregate_equal_weight, decode_delta_payload, encode_delta_file  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="phase8-test-")
        self.path = Path(self.temporary.name) / "delta.safetensors"
        self.delta = {"a": torch.tensor([[1.0, -2.0]]), "b": torch.tensor([0.25])}
        save_file(self.delta, str(self.path))
        self.base_hash = "a" * 64
        self.payload, self.manifest = encode_delta_file(
            self.path,
            sender_site="site-a",
            sender_node_id="endpoint-a",
            round_number=3,
            base_state_sha256=self.base_hash,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def decode(self, payload: str | None = None):
        return decode_delta_payload(
            payload or self.payload,
            expected_sender_site="site-a",
            expected_sender_node_id="endpoint-a",
            expected_round=3,
            expected_base_state_sha256=self.base_hash,
        )

    def test_round_trip(self) -> None:
        decoded = self.decode()
        self.assertEqual(decoded.envelope["delta_file_sha256"], self.manifest["delta_file_sha256"])
        self.assertEqual(state_sha256(decoded.state), state_sha256(self.delta))

    def test_rejects_wrong_provenance(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "round mismatch"):
            decode_delta_payload(
                self.payload,
                expected_sender_site="site-a",
                expected_sender_node_id="endpoint-a",
                expected_round=2,
                expected_base_state_sha256=self.base_hash,
            )
        with self.assertRaisesRegex(RuntimeError, "sender_node_id mismatch"):
            decode_delta_payload(
                self.payload,
                expected_sender_site="site-a",
                expected_sender_node_id="endpoint-b",
                expected_round=3,
                expected_base_state_sha256=self.base_hash,
            )

    def test_rejects_tampering_and_extra_fields(self) -> None:
        envelope = json.loads(self.payload)
        envelope["data"] = envelope["data"][:-2] + "AA"
        with self.assertRaises(RuntimeError):
            self.decode(json.dumps(envelope, separators=(",", ":"), sort_keys=True))
        envelope = json.loads(self.payload)
        envelope["unexpected"] = True
        with self.assertRaisesRegex(RuntimeError, "fields differ"):
            self.decode(json.dumps(envelope))

    def test_equal_weight_is_symmetric(self) -> None:
        base = {"x": torch.tensor([1.0, 2.0])}
        a = {"x": torch.tensor([0.2, -0.4])}
        b = {"x": torch.tensor([-0.1, 0.8])}
        _, left = aggregate_equal_weight(base, a, b)
        _, right = aggregate_equal_weight(base, b, a)
        self.assertEqual(max_abs_error(left, right), 0.0)
        self.assertTrue(torch.equal(left["x"], torch.tensor([1.05, 2.2])))


if __name__ == "__main__":
    unittest.main()
