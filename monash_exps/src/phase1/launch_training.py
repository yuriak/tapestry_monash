#!/usr/bin/env python3
"""Allocation-aware Ray/DDP launcher with experiment-side audit evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


_STEP_RE = re.compile(r"step_(\d+)$")


def allocated_cpus() -> int:
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        raw = os.environ.get(name, "").split("(", 1)[0]
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return len(os.sched_getaffinity(0))


def configured_trackers(raw: Any) -> set[str]:
    """Normalize Bhaskera's nullable string-or-list tracker field."""
    if raw is None:
        return set()
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    return {
        str(value).strip().lower()
        for value in values
        if value is not None and str(value).strip().lower() not in {"", "none", "off", "false"}
    }


def latest_checkpoint(save_dir: str | Path) -> Path | None:
    root = Path(save_dir)
    if not root.is_dir():
        return None
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and _STEP_RE.fullmatch(path.name) and (path / ".complete").is_file()
    ]
    return max(candidates, key=lambda path: int(_STEP_RE.fullmatch(path.name).group(1))) \
        if candidates else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lora_fingerprint(model: Any) -> dict[str, Any]:
    import torch

    digest = hashlib.sha256()
    count = 0
    nonzero = 0
    squared_norm = 0.0
    names: list[str] = []
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if "lora_" not in name:
                continue
            tensor = parameter.detach().cpu().contiguous()
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            count += tensor.numel()
            nonzero += int(torch.count_nonzero(tensor).item())
            squared_norm += float(torch.sum(tensor.float() ** 2).item())
            names.append(name)
    if not names:
        raise RuntimeError("no LoRA parameters found")
    return {
        "sha256": digest.hexdigest(),
        "tensor_names": names,
        "parameter_count": count,
        "nonzero_count": nonzero,
        "l2_norm": squared_norm ** 0.5,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def audited_worker(config_dict: dict[str, Any]) -> None:
    """Bhaskera worker equivalent with rank/device and parameter-sync evidence."""
    import random

    import numpy as np
    import ray.train
    import torch
    import torch.distributed as dist

    from bhaskera.config import Config
    from bhaskera.distributed import wrap_model
    from bhaskera.distributed.checkpoint import maybe_resume as load_latest_checkpoint
    from bhaskera.models import build_model
    from bhaskera.plugins.loader import load_plugins
    from bhaskera.trainer import train
    import bhaskera.trainer.checkpointing as checkpointing
    import bhaskera.trainer.loop as training_loop
    from bhaskera.utils import build_logger

    audit = config_dict["_phase1_audit"]
    cfg = Config.from_dict(config_dict)
    load_plugins(cfg)

    context = ray.train.get_context()
    local_rank = context.get_local_rank()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    seed = int(cfg.training.seed) + int(rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Bhaskera 2.2.0 omits rank when forwarding to the collective checkpoint
    # writer. Inject it externally so only global rank 0 performs filesystem
    # rename/prune operations; the third-party source remains untouched.
    original_save_checkpoint = checkpointing.save_checkpoint

    def rank_aware_save(*args: Any, **kwargs: Any) -> None:
        kwargs["rank"] = rank
        original_save_checkpoint(*args, **kwargs)

    checkpointing.save_checkpoint = rank_aware_save

    resume_checkpoint = latest_checkpoint(cfg.checkpoint.save_dir)
    resume_step = 0
    resume_adapter_sha256 = None
    if resume_checkpoint is not None:
        resume_step = int(_STEP_RE.fullmatch(resume_checkpoint.name).group(1))
        adapter = resume_checkpoint / "adapter_model.safetensors"
        if adapter.is_file():
            resume_adapter_sha256 = file_sha256(adapter)

        # Bhaskera's DCP payload currently contains model and optimizer state,
        # but not scheduler state. Reconstruct the deterministic scheduler at
        # the saved global step before DCP restores the optimizer state.
        original_build_scheduler = training_loop.build_scheduler

        def resumed_scheduler(optimizer: Any, train_cfg: Any) -> Any:
            scheduler = original_build_scheduler(optimizer, train_cfg)
            for _ in range(resume_step):
                scheduler.step()
            return scheduler

        training_loop.build_scheduler = resumed_scheduler

    model, profile = build_model(cfg, device)
    model = wrap_model(model, cfg, local_rank, profile)
    initial = lora_fingerprint(model)

    # Capture the in-memory state immediately after Bhaskera's DCP loader has
    # restored it. This distinguishes a real resume from merely discovering a
    # checkpoint directory on disk.
    resume_loaded_lora: dict[str, Any] | None = None
    def audited_resume(model_arg: Any, optimizer_arg: Any, save_dir: str) -> Any:
        nonlocal resume_loaded_lora
        # The Bhaskera trainer wrapper discovers an exact step directory and
        # then passes that child to a lower loader which expects the parent
        # save directory. Call the lower loader directly with its documented
        # input until the upstream path contract is fixed.
        result = load_latest_checkpoint(model_arg, optimizer_arg, save_dir)
        loaded_step = int(result[0])
        if loaded_step != resume_step:
            raise RuntimeError(
                "checkpoint resume contract failed: "
                f"discovered step {resume_step}, loader restored step {loaded_step}"
            )
        if loaded_step > 0:
            resume_loaded_lora = lora_fingerprint(model_arg)
        return result

    checkpointing.maybe_resume = audited_resume
    training_loop.maybe_resume = audited_resume

    start_evidence = {
        "pid": os.getpid(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "cuda_device": torch.cuda.current_device(),
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "initial_lora": initial,
        "resume_step": resume_step,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "resume_adapter_sha256": resume_adapter_sha256,
        "scheduler_fast_forward_steps": resume_step,
    }
    evidence_root = Path(audit["evidence_dir"])
    write_json(evidence_root / f"rank-{rank}-start.json", start_evidence)

    tracker = build_logger(cfg, rank=rank, world_size=world_size)
    dataset = ray.train.get_dataset_shard("train")
    train(
        model=model,
        dataset=dataset,
        cfg=cfg,
        profile=profile,
        rank=rank,
        local_rank=local_rank,
        tracker=tracker,
        world_size=world_size,
        ray_dataset_shard=dataset,
    )

    final = lora_fingerprint(model)
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, final)
    write_json(
        evidence_root / f"rank-{rank}-final.json",
        {
            **start_evidence,
            "resume_loaded_lora": resume_loaded_lora,
            "final_lora": final,
            "all_rank_final_lora": gathered,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    import ray
    import torch
    from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    from bhaskera.config import load_config
    from bhaskera.data import build_ray_dataset

    if args.num_workers < 1:
        raise SystemExit("--num-workers must be positive")
    visible_gpus = torch.cuda.device_count()
    if visible_gpus != args.num_workers:
        raise SystemExit(f"expected exactly {args.num_workers} visible GPUs, got {visible_gpus}")

    cfg = load_config(str(args.config.resolve()))
    if cfg.training.distributed.strategy.lower() != "ddp":
        raise SystemExit("Phase 1 requires DDP")
    if not cfg.lora.enabled:
        raise SystemExit("Phase 1 requires LoRA")
    trackers = configured_trackers(cfg.logging.tracker)
    if args.num_workers > 1 and "ray" in trackers:
        raise SystemExit(
            "RayMetricsLogger is unsafe for this Bhaskera DDP loop: some metrics are "
            "reported only by rank 0, while ray.train.report is collective. Set "
            "logging.tracker: null and use the Phase 1 stdout verifier."
        )

    run_dir = args.run_dir.resolve()
    evidence_dir = run_dir / "worker-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    cpus = allocated_cpus()
    if cpus < args.num_workers + 2:
        raise SystemExit(f"need at least {args.num_workers + 2} CPUs, allocation has {cpus}")
    cpus_per_worker = max(1, (cpus - 2) // args.num_workers)
    # Keep Ray sockets under a short node-local path; descriptive artifact
    # paths can exceed the AF_UNIX 107-byte limit before Ray even starts.
    ray_temp = Path(os.environ.get("TMPDIR", "/tmp")) / f"slakshna-p1-train-{os.getpid()}"
    ray_temp.mkdir(parents=True, exist_ok=True)

    context = ray.init(
        num_cpus=cpus,
        num_gpus=visible_gpus,
        include_dashboard=False,
        _temp_dir=str(ray_temp),
    )
    resources = {key: float(value) for key, value in ray.cluster_resources().items()}
    write_json(
        run_dir / "launcher-environment.json",
        {
            "pid": os.getpid(),
            "allocated_cpus": cpus,
            "cpus_per_worker": cpus_per_worker,
            "visible_gpus": visible_gpus,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(visible_gpus)],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ray_address": context.address_info.get("address"),
            "ray_resources": resources,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "configured_trackers": sorted(trackers),
            "training_metrics_source": "rank0-structured-stdout",
        },
    )

    try:
        dataset = build_ray_dataset(cfg, world_size=args.num_workers)
        train_loop_config = cfg.as_dict()
        train_loop_config["_phase1_audit"] = {"evidence_dir": str(evidence_dir)}
        trainer = TorchTrainer(
            train_loop_per_worker=audited_worker,
            train_loop_config=train_loop_config,
            datasets={"train": dataset},
            scaling_config=ScalingConfig(
                num_workers=args.num_workers,
                use_gpu=True,
                resources_per_worker={"GPU": 1, "CPU": cpus_per_worker},
            ),
            run_config=RunConfig(
                name=cfg.logging.run_name,
                storage_path=str(run_dir / "ray-results"),
                checkpoint_config=CheckpointConfig(num_to_keep=cfg.checkpoint.keep_last_n),
                failure_config=FailureConfig(max_failures=0),
            ),
        )
        result = trainer.fit()
        write_json(
            run_dir / "ray-result.json",
            {
                "metrics": result.metrics,
                "error": str(result.error) if result.error else None,
                "path": result.path,
            },
        )
        if result.error:
            raise RuntimeError(result.error)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
