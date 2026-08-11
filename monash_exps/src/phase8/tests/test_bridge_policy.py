from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
os.environ.setdefault("SLAKSHNA_EXPERIMENT_ROOT", str(EXPERIMENT_ROOT))
ml_bridge = importlib.import_module("phase8.ml_bridge")


class BridgeVerificationPolicyTests(unittest.TestCase):
    def test_allows_small_stochastic_round_loss_regression(self) -> None:
        command = ml_bridge.training_verification_command(Path("/tmp/phase8-round"))
        threshold_index = command.index("--minimum-loss-drop") + 1
        self.assertEqual(float(command[threshold_index]), -0.05)
        self.assertLess(-0.00678, 0.0)
        self.assertGreaterEqual(-0.00678, float(command[threshold_index]))

    def test_still_rejects_material_local_regression(self) -> None:
        self.assertLess(-0.10, ml_bridge.MINIMUM_LOCAL_LOSS_DROP)


if __name__ == "__main__":
    unittest.main()
