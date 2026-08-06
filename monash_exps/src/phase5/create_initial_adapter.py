#!/usr/bin/env python3
"""Create the common Phase 5 LoRA initialization without an optimizer step."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase1.launch_training import canonical_lora_state  # noqa: E402
from phase2.adapter_delta import state_sha256, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    from bhaskera.config import load_config
    from bhaskera.models import build_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"initial adapter creation requires exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    cfg = load_config(str(args.config.resolve()))
    seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model, _ = build_model(cfg, device)
    state = canonical_lora_state(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    save_file(state, str(temporary))
    temporary.replace(args.output)
    temporary_resume = args.resume_output.with_suffix(args.resume_output.suffix + ".tmp")
    torch.save(state, temporary_resume)
    temporary_resume.replace(args.resume_output)
    audit = {
        "seed": seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": torch.cuda.get_device_name(0),
        "tensor_count": len(state),
        "parameter_count": sum(value.numel() for value in state.values()),
        "state_sha256": state_sha256(state),
        "config": str(args.config.resolve()),
    }
    write_json(args.audit_output.resolve(), audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
