#!/usr/bin/env python3
"""Apply the narrow M0 round-shard/fail-closed patch to a runtime source copy."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} patch point, found {count}")
    return text.replace(old, new, 1)


def patch_text(text: str) -> str:
    text = replace_once(
        text,
        '''BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")
''',
        '''BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The formal M0 runtime carries an isolated Bhaskera source overlay. Ensure
# the Ray/Bhaskera subprocess launched from the checkpoint directory imports
# that overlay rather than mutating or depending on the upstream submodule.
os.environ["PYTHONPATH"] = BASE_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")
''',
        "runtime Bhaskera overlay",
    )
    text = replace_once(
        text,
        'def prepare_bhaskera_config(my_id, is_malicious, training_mode="finetuning"):',
        'def prepare_bhaskera_config(my_id, is_malicious, training_mode="finetuning", round_index=None):',
        "prepare signature",
    )
    text = replace_once(
        text,
        '''    node_data_dir = os.path.join(DATA_DIR, f"data_{my_id}")
    node_cache_dir = os.path.join(node_data_dir, "tokenized_cache")
    node_ckpt_dir = os.path.join(MODEL_DIR, f"ckpt_{my_id}")
''',
        '''    node_data_dir = os.path.join(DATA_DIR, f"data_{my_id}")
    node_cache_dir = os.path.join(node_data_dir, "tokenized_cache")
    node_ckpt_dir = os.path.join(MODEL_DIR, f"ckpt_{my_id}")

    federated_data = config.get("federated_data", {})
    round_shards_enabled = bool(federated_data.get("round_shards_enabled", False))
    selected_round = None
    if round_shards_enabled:
        shards = federated_data.get("round_shards", [])
        if not isinstance(round_index, int) or not 1 <= round_index <= len(shards):
            raise RuntimeError(
                f"federated round {round_index!r} has no declared data shard "
                f"(declared={len(shards)})"
            )
        selected_round = shards[round_index - 1]
        if int(selected_round.get("round", -1)) != round_index:
            raise RuntimeError(f"round-shard index mismatch: {selected_round}")
        node_cache_dir = os.path.abspath(str(selected_round["path"]))
        expected_steps = int(selected_round["max_steps"])
        expected_rows = int(selected_round["rows"])
        if expected_rows != expected_steps * int(federated_data["effective_global_batch"]):
            raise RuntimeError(f"round-shard row/step mismatch: {selected_round}")
        if not os.path.isdir(node_cache_dir):
            raise RuntimeError(f"round shard is missing: {node_cache_dir}")
        parquet_files = [
            name for name in os.listdir(node_cache_dir) if name.endswith(".parquet")
        ]
        if len(parquet_files) != 1:
            raise RuntimeError(
                f"round shard must contain exactly one parquet: {node_cache_dir}"
            )
        config["training"]["max_steps"] = expected_steps
''',
        "round shard selection",
    )
    text = replace_once(
        text,
        '''    config["data"]["tokenized_path"] = node_cache_dir
    config["data"]["cache_dir"] = node_cache_dir  # Required by Bhaskera tokenizer
''',
        '''    config["data"]["tokenized_path"] = node_cache_dir
    config["data"]["cache_dir"] = node_cache_dir  # Required by Bhaskera tokenizer
    if round_shards_enabled:
        selection_log = os.path.join(LOG_DIR, "round_data_selection.jsonl")
        with open(selection_log, "a", encoding="utf-8") as selection_handle:
            selection_handle.write(json.dumps({
                "node_id": my_id,
                "round": round_index,
                "path": node_cache_dir,
                "rows": int(selected_round["rows"]),
                "max_steps": int(selected_round["max_steps"]),
                "parquet_sha256": selected_round["parquet_sha256"],
            }, sort_keys=True) + "\\n")
''',
        "selection audit",
    )
    text = replace_once(
        text,
        "prepare_bhaskera_config(my_id, is_malicious, TRAINING_MODE)",
        'prepare_bhaskera_config(my_id, is_malicious, TRAINING_MODE, state["round"])',
        "prepare call",
    )
    text = replace_once(
        text,
        '''    if step_dirs:
        latest_step_dir = str(step_dirs[-1])
''',
        '''    formal_round_shards = bool(
        template_config.get("federated_data", {}).get("round_shards_enabled", False)
    )
    if formal_round_shards and not step_dirs:
        raise RuntimeError(
            f"round {state['round']} produced no complete Bhaskera DCP checkpoint"
        )

    if formal_round_shards and TRAINING_MODE == "finetuning":
        latest_step_dir = str(step_dirs[-1])
        formal_adapter_path = os.path.join(latest_step_dir, "adapter_model.safetensors")
        if not os.path.isfile(formal_adapter_path):
            raise RuntimeError(
                f"round {state['round']} DCP did not export a LoRA adapter: "
                f"{formal_adapter_path}"
            )
        import safetensors.torch
        model_sd = {
            key: value.to(device)
            for key, value in safetensors.torch.load_file(formal_adapter_path).items()
        }
        if not old_sd:
            raise RuntimeError("formal finetuning requires an explicit common base LoRA")
        delta_i = {
            key: model_sd[key].to(device) - old_sd[key].to(device)
            for key in model_sd
            if key in old_sd
        }
    elif step_dirs:
        latest_step_dir = str(step_dirs[-1])
''',
        "DCP fail closed",
    )
    text = replace_once(
        text,
        '''        else:
            delta_i = {"dummy": torch.zeros(1, device=device)}

''' + "    # [Security & DP] \n",
        '''        else:
            if formal_round_shards:
                raise RuntimeError(
                    f"round {state['round']} has neither DCP nor adapter checkpoint"
                )
            delta_i = {"dummy": torch.zeros(1, device=device)}

    if formal_round_shards:
        expected_tensor_count = int(
            template_config["federated_data"]["expected_lora_tensors"]
        )
        expected_parameter_count = int(
            template_config["federated_data"]["expected_lora_parameters"]
        )
        if "dummy" in delta_i or len(delta_i) != expected_tensor_count:
            raise RuntimeError(
                f"invalid local delta schema: keys={len(delta_i)}, "
                f"expected={expected_tensor_count}, dummy={'dummy' in delta_i}"
            )
        if old_sd and set(delta_i) != set(old_sd):
            missing = sorted(set(old_sd) - set(delta_i))[:5]
            extra = sorted(set(delta_i) - set(old_sd))[:5]
            raise RuntimeError(
                f"local delta differs from base LoRA schema: missing={missing}, extra={extra}"
            )
        parameter_count = sum(
            tensor.numel() for tensor in delta_i.values() if torch.is_tensor(tensor)
        )
        if parameter_count != expected_parameter_count:
            raise RuntimeError(
                f"invalid local delta parameter count: {parameter_count} != "
                f"{expected_parameter_count}"
            )
        if any(
            not torch.is_tensor(tensor)
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all().item()
            for tensor in delta_i.values()
        ):
            raise RuntimeError("local delta contains non-floating or non-finite tensors")
        delta_l2 = math.sqrt(sum(
            tensor.detach().float().pow(2).sum().item() for tensor in delta_i.values()
        ))
        if not math.isfinite(delta_l2) or delta_l2 <= 0.0:
            raise RuntimeError(f"local delta must be finite and non-zero, got {delta_l2}")
        with open(os.path.join(LOG_DIR, "local_delta_audit.jsonl"), "a", encoding="utf-8") as audit_handle:
            audit_handle.write(json.dumps({
                "node_id": my_id,
                "round": state["round"],
                "tensor_count": len(delta_i),
                "parameter_count": parameter_count,
                "l2_norm": delta_l2,
                "checkpoint": latest_step_dir,
            }, sort_keys=True) + "\\n")

    # [Security & DP]
''',
        "delta fail closed",
    )
    return text


def patch_file(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    patched = patch_text(original)
    path.write_text(patched, encoding="utf-8")


def patch_bhaskera_checkpointing(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    patched = replace_once(
        original,
        '''        extra={"avg_loss": float(avg_loss)},
        cursor_meta=cursor_meta,
    )
''',
        '''        extra={"avg_loss": float(avg_loss)},
        rank=rank,
        keep_last_n=ckpt_cfg.keep_last_n,
        cursor_meta=cursor_meta,
    )
''',
        "Bhaskera checkpoint rank",
    )
    path.write_text(patched, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--bhaskera-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.bhaskera_checkpointing:
        patch_bhaskera_checkpointing(args.path)
    else:
        patch_file(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
