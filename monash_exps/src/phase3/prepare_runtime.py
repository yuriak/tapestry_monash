#!/usr/bin/env python3
"""Materialize the isolated Phase 3 Slakshna runtime directory."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--p2p-port", type=int, required=True)
    parser.add_argument("--ws-port", type=int, required=True)
    parser.add_argument("--api-port", type=int, required=True)
    args = parser.parse_args()

    ports = [args.p2p_port, args.ws_port, args.api_port]
    if len(set(ports)) != 3 or any(port < 1024 or port > 65535 for port in ports):
        raise RuntimeError(f"invalid Phase 3 ports: {ports}")
    runtime = args.runtime_dir.resolve()
    runtime.mkdir(parents=True, exist_ok=False)
    args.data_dir.resolve().mkdir(parents=True, exist_ok=False)

    text = args.template.resolve().read_text(encoding="utf-8")
    replacements = {
        "__DATA_DIR__": str(args.data_dir.resolve()),
        "__P2P_PORT__": str(args.p2p_port),
        "__WS_PORT__": str(args.ws_port),
        "__API_PORT__": str(args.api_port),
    }
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    unresolved = [marker for marker in replacements if marker in text]
    if unresolved:
        raise RuntimeError(f"unresolved Phase 3 config markers: {unresolved}")
    (runtime / "node.toml").write_text(text, encoding="utf-8")
    shutil.copy2(args.bridge.resolve(), runtime / "ml_engine.py")


if __name__ == "__main__":
    main()
