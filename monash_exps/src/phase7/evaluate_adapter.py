#!/usr/bin/env python3
"""Evaluate one Phase 7 global adapter on both held-out site datasets."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoTokenizer


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
from phase1.launch_training import canonical_lora_state  # noqa: E402
from phase2.adapter_delta import max_abs_error, read_state, state_sha256, write_json  # noqa: E402


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise RuntimeError(f"validation dataset is empty: {path}")
    return rows


def render(tokenizer: Any, messages: list[dict[str, str]], generation: bool = False) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=generation,
    )


def evaluate_site(
    model: Any,
    tokenizer: Any,
    path: Path,
    batch_size: int,
    seq_len: int,
) -> dict[str, Any]:
    rows = load_rows(path)
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            texts = [render(tokenizer, row["messages"]) for row in rows[start:start + batch_size]]
            batch = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            )
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            shift_logits = logits[:, :-1, :].float().contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            valid = attention_mask[:, 1:].bool()
            labels = shift_labels.masked_fill(~valid, -100)
            loss = functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            tokens = int(valid.sum().item())
            total_loss += float(loss.item())
            total_tokens += tokens
    if total_tokens == 0:
        raise RuntimeError(f"validation dataset produced no tokens: {path}")
    nll = total_loss / total_tokens
    return {
        "path": str(path),
        "examples": len(rows),
        "tokens": total_tokens,
        "negative_log_likelihood": nll,
        "perplexity": math.exp(min(nll, 20.0)),
    }


def generate_sample(model: Any, tokenizer: Any) -> dict[str, Any]:
    messages = [{"role": "user", "content": "Give two concise tips for collaborative research."}]
    prompt = render(tokenizer, messages, generation=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    if not generated.strip():
        raise RuntimeError("fresh-process inference produced empty text")
    return {"prompt": messages[0]["content"], "generated_text": generated.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-resume", type=Path, required=True)
    parser.add_argument("--peer-a-validation", type=Path, required=True)
    parser.add_argument("--peer-b-validation", type=Path, required=True)
    parser.add_argument("--global-number", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(f"evaluation requires exactly one visible GPU, got {torch.cuda.device_count()}")
    from bhaskera.config import load_config
    from bhaskera.models import build_model

    cfg = load_config(str(args.config.resolve()))
    cfg.lora.resume_path = str(args.adapter_resume.resolve())
    device = torch.device("cuda:0")
    model, _ = build_model(cfg, device)
    model.eval()
    expected_state = read_state(args.adapter.resolve())
    loaded_state = canonical_lora_state(model)
    load_error = max_abs_error(expected_state, loaded_state)
    if load_error > 1.0e-7:
        raise RuntimeError(f"fresh process loaded the wrong adapter: max error {load_error}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    sites = {
        "peer-a": evaluate_site(
            model, tokenizer, args.peer_a_validation.resolve(), args.batch_size, int(cfg.data.seq_len)
        ),
        "peer-b": evaluate_site(
            model, tokenizer, args.peer_b_validation.resolve(), args.batch_size, int(cfg.data.seq_len)
        ),
    }
    macro_nll = sum(site["negative_log_likelihood"] for site in sites.values()) / len(sites)
    result = {
        "schema_version": 1,
        "global_number": args.global_number,
        "adapter_path": str(args.adapter.resolve()),
        "adapter_state_sha256": state_sha256(expected_state),
        "fresh_load_max_abs_error": load_error,
        "sites": sites,
        "macro_negative_log_likelihood": macro_nll,
        "macro_perplexity": math.exp(min(macro_nll, 20.0)),
        "inference": generate_sample(model, tokenizer) if args.generate else None,
    }
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
