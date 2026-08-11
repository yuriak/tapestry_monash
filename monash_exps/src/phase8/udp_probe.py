#!/usr/bin/env python3
"""Authenticated nonce/echo probe for the Phase 8 playit UDP path."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import secrets
import socket
import time
from pathlib import Path
from typing import Any


FORMAT = "slakshna-phase8-udp-probe"
VERSION = 1


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def encode(kind: str, token: str, nonce: str) -> bytes:
    value = {"format": FORMAT, "version": VERSION, "kind": kind, "token": token, "nonce": nonce}
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode(raw: bytes, expected_kind: str, token: str) -> dict[str, Any]:
    if len(raw) > 4096:
        raise RuntimeError("UDP probe datagram exceeds 4096 bytes")
    value = json.loads(raw)
    expected_fields = {"format", "version", "kind", "token", "nonce"}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise RuntimeError("UDP probe fields differ")
    if value["format"] != FORMAT or value["version"] != VERSION:
        raise RuntimeError("UDP probe version mismatch")
    if value["kind"] != expected_kind or value["token"] != token:
        raise RuntimeError("UDP probe kind/token mismatch")
    if not isinstance(value["nonce"], str) or len(value["nonce"]) != 32:
        raise RuntimeError("UDP probe nonce is invalid")
    return value


def serve(args: argparse.Namespace) -> None:
    address = ipaddress.ip_address(args.bind)
    if address.version != 4 or not address.is_loopback:
        raise RuntimeError("the playit probe server must bind an IPv4 loopback address")
    started = time.monotonic()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((str(address), args.port))
        sock.settimeout(1.0)
        print(f"UDP probe listening on {address}:{args.port}", flush=True)
        while time.monotonic() - started < args.wait:
            try:
                raw, peer = sock.recvfrom(4096)
            except TimeoutError:
                continue
            try:
                request = decode(raw, "request", args.token)
            except Exception as error:
                print(f"Ignoring invalid UDP probe from {peer}: {error}", flush=True)
                continue
            response = encode("response", args.token, request["nonce"])
            sock.sendto(response, peer)
            result = {
                "format": FORMAT, "version": VERSION, "status": "PASS", "role": "server",
                "local_address": f"{address}:{args.port}",
                "remote_address_sha256": hashlib.sha256(f"{peer[0]}:{peer[1]}".encode()).hexdigest(),
                "nonce_sha256": hashlib.sha256(request["nonce"].encode()).hexdigest(),
            }
            write_json(args.output.resolve(), result)
            print(json.dumps(result, indent=2, sort_keys=True))
            print("PHASE8 PLAYIT UDP SERVER PROBE PASSED")
            return
    raise RuntimeError(f"no valid UDP probe arrived within {args.wait} seconds")


def resolve_ipv4(host: str, port: int) -> list[str]:
    addresses = sorted({
        item[4][0] for item in socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)
    })
    if not addresses:
        raise RuntimeError(f"no IPv4 address resolved for {host}")
    return addresses


def probe(args: argparse.Namespace) -> None:
    addresses = resolve_ipv4(args.host, args.port)
    nonce = secrets.token_hex(16)
    request = encode("request", args.token, nonce)
    started = time.monotonic()
    attempts = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(args.timeout)
        for _ in range(args.attempts):
            for address in addresses:
                attempts += 1
                sent = time.monotonic()
                sock.sendto(request, (address, args.port))
                try:
                    raw, peer = sock.recvfrom(4096)
                except TimeoutError:
                    continue
                response = decode(raw, "response", args.token)
                if response["nonce"] != nonce:
                    continue
                result = {
                    "format": FORMAT, "version": VERSION, "status": "PASS", "role": "client",
                    "target_host": args.host, "resolved_ipv4": addresses,
                    "target_port": args.port, "response_ipv4": peer[0],
                    "attempts": attempts, "round_trip_ms": (time.monotonic() - sent) * 1000,
                    "elapsed_seconds": time.monotonic() - started,
                    "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(),
                }
                write_json(args.output.resolve(), result)
                print(json.dumps(result, indent=2, sort_keys=True))
                print("PHASE8 PLAYIT UDP CLIENT PROBE PASSED")
                return
            time.sleep(args.interval)
    raise RuntimeError(
        f"no valid UDP response after {attempts} sends to {addresses}:{args.port}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("server")
    server.add_argument("--bind", default="127.0.0.1")
    server.add_argument("--port", type=int, required=True)
    server.add_argument("--token", required=True)
    server.add_argument("--wait", type=int, default=900)
    server.add_argument("--output", type=Path, required=True)
    server.set_defaults(action=serve)
    client = commands.add_parser("client")
    client.add_argument("--host", required=True)
    client.add_argument("--port", type=int, required=True)
    client.add_argument("--token", required=True)
    client.add_argument("--attempts", type=int, default=10)
    client.add_argument("--timeout", type=float, default=2.0)
    client.add_argument("--interval", type=float, default=1.0)
    client.add_argument("--output", type=Path, required=True)
    client.set_defaults(action=probe)
    args = parser.parse_args()
    if args.port < 1024 or args.port > 65535:
        raise RuntimeError(f"invalid UDP port: {args.port}")
    if not args.token or len(args.token) > 128:
        raise RuntimeError("probe token must contain 1-128 characters")
    args.action(args)


if __name__ == "__main__":
    main()
