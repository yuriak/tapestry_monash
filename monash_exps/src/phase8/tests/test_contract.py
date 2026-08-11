from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
from phase8.g0_bundle import contract_sha256, training_contract  # noqa: E402


class ContractTests(unittest.TestCase):
    def make_site(self, root: Path, site: str) -> None:
        (root / "bootstrap").mkdir(parents=True)
        config = {
            "model": {"name": f"/{site}/model", "dtype": "bfloat16", "attn_impl": "sdpa", "quantization": "none"},
            "data": {"format": "chatml", "seq_len": 256, "pack_sequences": False,
                     "train_path": f"/{site}/private.jsonl", "tokenized_path": f"/{site}/tokens"},
            "lora": {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.0,
                     "target_modules": ["q_proj", "v_proj"], "resume_path": f"/{site}/g0.pth"},
            "training": {"batch_size": 16, "grad_accum": 1, "lr": 5e-5,
                         "weight_decay": 0.0, "warmup_steps": 20, "max_grad_norm": 1.0,
                         "seed": 20260806, "deterministic": False,
                         "distributed": {"strategy": "ddp"}},
            "checkpoint": {"save_dir": f"/{site}/checkpoints"},
            "logging": {"run_name": site},
        }
        (root / "bootstrap/resolved-config.yaml").write_text(yaml.safe_dump(config))
        manifest = {
            "site": site,
            "partition_type": "disjoint-category-non-iid",
            "dataset": {"id": "dataset", "revision": "revision", "split": "train", "license": "x"},
            "model": {"id": "model", "revision": "model-revision", "snapshot_path": f"/{site}/model", "license": "y"},
            "shard": {"train": {"rows": 1152}, "validation": {"rows": 128}},
        }
        (root / "site-manifest.json").write_text(json.dumps(manifest))

    def test_contract_excludes_cluster_local_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase8-contract-") as temporary:
            root = Path(temporary)
            a, b = root / "a", root / "b"
            self.make_site(a, "site-a")
            self.make_site(b, "site-b")
            contract_a = training_contract(a)
            contract_b = training_contract(b)
            self.assertEqual(contract_a, contract_b)
            self.assertEqual(contract_sha256(contract_a), contract_sha256(contract_b))
            serialized = json.dumps(contract_a)
            self.assertNotIn("/site-a/", serialized)
            self.assertNotIn("/site-b/", serialized)


if __name__ == "__main__":
    unittest.main()
