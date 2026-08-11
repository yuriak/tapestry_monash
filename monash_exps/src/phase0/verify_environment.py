#!/usr/bin/env python3
"""Fail-closed Phase 0 API, configuration, CUDA, Ray, and source verifier."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable


MIN_CUDA = (12, 8)
REQUIRED_MODULES = (
    "torch",
    "ray",
    "ray.data",
    "ray.train",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "pyarrow",
    "safetensors",
    "omegaconf",
    "yaml",
    "bhaskera",
)
PACKAGE_NAMES = (
    "torch",
    "ray",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "pyarrow",
    "safetensors",
    "omegaconf",
    "pyyaml",
    "bhaskera",
)


def version_pair(raw: str | None) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\.(\d+)", raw or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def leading_int(raw: str | None) -> int | None:
    match = re.match(r"\s*(\d+)", raw or "")
    return int(match.group(1)) if match else None


def command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def python_tree_digest(root: Path) -> tuple[str, int]:
    """Hash relative names and contents of all package Python sources."""
    digest = hashlib.sha256()
    sources = sorted(root.rglob("*.py"))
    for source in sources:
        relative = source.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = source.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest(), len(sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = args.workspace_root.resolve() / "Slakshna"
    checks: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def check(name: str, action: Callable[[], Any]) -> Any:
        try:
            detail = action()
            checks[name] = {"status": "PASS", "detail": detail}
            return detail
        except Exception as error:  # verifier must retain all independent failures
            message = f"{type(error).__name__}: {error}"
            checks[name] = {"status": "FAIL", "detail": message}
            failures.append(f"{name}: {message}")
            return None

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    def verify_python() -> dict[str, str]:
        require(sys.version_info[:2] == (3, 11), f"expected Python 3.11, got {sys.version}")
        return {"executable": sys.executable, "version": sys.version.split()[0], "prefix": sys.prefix}

    check("python", verify_python)

    imported: dict[str, Any] = {}

    def verify_imports() -> dict[str, str]:
        for module_name in REQUIRED_MODULES:
            imported[module_name] = importlib.import_module(module_name)
        from ray.train.torch import TorchTrainer  # noqa: F401
        import torch.distributed.checkpoint as dcp  # noqa: F401
        return {name: "imported" for name in REQUIRED_MODULES}

    check("required_imports", verify_imports)

    def package_versions() -> dict[str, str]:
        versions = {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}
        (output_dir / "environment.json").write_text(
            json.dumps(
                {
                    "python": sys.version.split()[0],
                    "executable": sys.executable,
                    "packages": versions,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return versions

    check("package_versions", package_versions)

    def verify_config() -> dict[str, Any]:
        from bhaskera.config import load_config
        import yaml

        config = load_config(str(args.config.resolve()))
        resolved = asdict(config)
        require(resolved["model"]["name"] == "Qwen/Qwen3-0.6B", "model name changed")
        require(resolved["model"]["dtype"] == "bfloat16", "dtype is not bfloat16")
        require(resolved["model"]["attn_impl"] == "sdpa", "attention is not SDPA")
        require(resolved["model"]["use_liger_kernel"] is False, "Liger must be disabled")
        require(resolved["training"]["distributed"]["strategy"] == "ddp", "strategy is not DDP")
        require(resolved["lora"]["enabled"] is True, "LoRA is disabled")
        (output_dir / "resolved-config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        return {
            "model": resolved["model"]["name"],
            "dtype": resolved["model"]["dtype"],
            "strategy": resolved["training"]["distributed"]["strategy"],
            "lora": resolved["lora"]["enabled"],
        }

    check("bhaskera_config", verify_config)

    def verify_submodule() -> dict[str, str]:
        git = command("git", "-C", str(source_root), "status", "--porcelain")
        require(git.returncode == 0, git.stderr.strip() or "git status failed")
        require(not git.stdout.strip(), f"submodule has changes: {git.stdout.strip()}")
        revision = command("git", "-C", str(source_root), "rev-parse", "HEAD")
        require(revision.returncode == 0, revision.stderr.strip() or "git rev-parse failed")
        commit = revision.stdout.strip()
        pin = command(
            "git", "-C", str(args.workspace_root.resolve()),
            "ls-files", "--stage", "--", "Slakshna",
        )
        require(pin.returncode == 0, pin.stderr.strip() or "cannot inspect parent submodule pin")
        fields = pin.stdout.strip().split()
        require(len(fields) >= 2 and fields[0] == "160000",
                "Slakshna is not recorded as a Git submodule in the parent index")
        require(fields[1] == commit,
                f"parent pins {fields[1]}, but submodule is checked out at {commit}")
        snapshot = args.experiment_root.resolve() / ".runtime" / "sources" / "Bhaskera"
        snapshot_marker = snapshot / ".slakshna-source-revision"
        require(snapshot_marker.is_file(), f"missing Bhaskera snapshot marker: {snapshot_marker}")
        snapshot_revision = snapshot_marker.read_text(encoding="utf-8").strip()
        require(snapshot_revision == commit,
                f"Bhaskera snapshot is {snapshot_revision}, expected {commit}")
        distribution = importlib.metadata.distribution("bhaskera")
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        installed_url = direct_url.get("url", "")
        require("/.runtime/sources/Bhaskera" in installed_url,
                f"Bhaskera was not built from the isolated snapshot: {installed_url}")
        config_module = importlib.import_module("bhaskera.config")
        installed_file = Path(config_module.__file__).resolve()
        require(source_root not in installed_file.parents,
                f"Bhaskera import resolves into the submodule: {installed_file}")
        package_paths = list(importlib.import_module("bhaskera").__path__)
        require(len(package_paths) == 1,
                f"expected one installed Bhaskera package path, got {package_paths}")
        installed_package = Path(package_paths[0]).resolve()
        snapshot_package = snapshot / "src" / "bhaskera"
        require(snapshot_package.is_dir(), f"missing snapshot package: {snapshot_package}")
        snapshot_digest, snapshot_count = python_tree_digest(snapshot_package)
        installed_digest, installed_count = python_tree_digest(installed_package)
        require(snapshot_count > 0, "Bhaskera snapshot contains no Python sources")
        require(installed_count == snapshot_count,
                f"installed Bhaskera source count {installed_count} != snapshot {snapshot_count}")
        require(installed_digest == snapshot_digest,
                "installed Bhaskera sources do not match the pinned snapshot")
        (output_dir / "submodule-revision.txt").write_text(commit + "\n", encoding="utf-8")
        return {
            "revision": commit,
            "parent_pin": fields[1],
            "snapshot_revision": snapshot_revision,
            "installed_from": installed_url,
            "installed_file": str(installed_file),
            "python_source_count": installed_count,
            "python_source_sha256": installed_digest,
            "working_tree": "clean",
        }

    check("slakshna_submodule", verify_submodule)

    if args.mode == "gpu":
        def verify_cuda() -> dict[str, Any]:
            import torch

            build = version_pair(torch.version.cuda)
            require(build is not None and build >= MIN_CUDA,
                    f"PyTorch CUDA build must be >=12.8, got {torch.version.cuda}")
            require(torch.cuda.is_available(), "torch.cuda.is_available() is false")
            count = torch.cuda.device_count()
            expected = int(os.environ.get("SLAKSHNA_EXPECTED_GPUS", str(count)))
            require(count == expected, f"expected {expected} visible GPUs, got {count}")
            gpus = []
            for index in range(count):
                name = torch.cuda.get_device_name(index)
                capability = torch.cuda.get_device_capability(index)
                require("A100" in name.upper(), f"GPU {index} is not A100: {name}")
                require(capability[0] >= 8, f"GPU {index} lacks required capability: {capability}")
                gpus.append({"index": index, "name": name, "capability": capability})
            require(torch.distributed.is_nccl_available(), "PyTorch NCCL backend is unavailable")
            return {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "visible_gpu_count": count,
                "gpus": gpus,
                "nccl_available": True,
            }

        cuda_info = check("cuda_and_gpus", verify_cuda)

        def verify_driver_and_toolkit() -> dict[str, str]:
            smi = command("nvidia-smi")
            require(smi.returncode == 0, smi.stderr.strip() or "nvidia-smi failed")
            driver_cuda_match = re.search(r"CUDA Version:\s*(\d+\.\d+)", smi.stdout)
            require(driver_cuda_match is not None, "cannot parse CUDA version from nvidia-smi")
            driver_cuda = driver_cuda_match.group(1)
            require(version_pair(driver_cuda) >= MIN_CUDA,
                    f"driver CUDA capability must be >=12.8, got {driver_cuda}")
            nvcc = command("nvcc", "--version")
            require(nvcc.returncode == 0, nvcc.stderr.strip() or "nvcc failed")
            toolkit_match = re.search(r"release\s+(\d+\.\d+)", nvcc.stdout)
            require(toolkit_match is not None, "cannot parse nvcc release")
            toolkit = toolkit_match.group(1)
            require(version_pair(toolkit) >= MIN_CUDA,
                    f"CUDA toolkit must be >=12.8, got {toolkit}")
            return {"driver_cuda": driver_cuda, "toolkit": toolkit}

        check("driver_and_toolkit", verify_driver_and_toolkit)

        def verify_ray_resources() -> dict[str, float]:
            import ray

            require(cuda_info is not None, "CUDA check failed; refusing to initialize Ray")
            allocated_cpus = (
                leading_int(os.environ.get("SLURM_CPUS_PER_TASK"))
                or leading_int(os.environ.get("SLURM_CPUS_ON_NODE"))
                or len(os.sched_getaffinity(0))
            )
            gpu_count = int(cuda_info["visible_gpu_count"])
            ray_tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"slakshna-phase0-{os.getpid()}"
            context = ray.init(
                num_cpus=allocated_cpus,
                num_gpus=gpu_count,
                include_dashboard=False,
                _temp_dir=str(ray_tmp),
            )
            try:
                resources = {key: float(value) for key, value in ray.cluster_resources().items()}
                require(resources.get("CPU", 0.0) <= allocated_cpus,
                        f"Ray registered CPUs outside allocation: {resources}")
                require(resources.get("GPU", 0.0) == gpu_count,
                        f"Ray GPU resources do not match visibility: {resources}")
                (output_dir / "torch-ray-resources.json").write_text(
                    json.dumps(
                        {
                            "ray_address": context.address_info.get("address"),
                            "allocated_cpus": allocated_cpus,
                            "resources": resources,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return resources
            finally:
                ray.shutdown()

        check("ray_resources", verify_ray_resources)

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "mode": args.mode,
        "checks": checks,
        "failures": failures,
    }
    summary_path = output_dir / "verification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    print(f"PHASE 0 {args.mode.upper()} PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
