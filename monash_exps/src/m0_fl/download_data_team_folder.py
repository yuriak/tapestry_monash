#!/usr/bin/env python3
"""Download the authenticated data-team Google Drive folder with gdown."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from monash_exps.src.m0.download_authenticated_drive import (
    DRIVE_FOLDER_MIME,
    download_file,
    install_gdown_session,
    list_folder,
    load_session,
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cookie-file", required=True, type=Path)
    parser.add_argument("--authuser", type=int, default=2)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    if not args.cookie_file.is_file() or args.cookie_file.stat().st_size == 0:
        parser.error(f"missing cookie file: {args.cookie_file}")

    probe = load_session(args.cookie_file, args.authuser).get(
        f"https://drive.google.com/drive/u/{args.authuser}/folders/{args.folder_id}",
        timeout=90,
    )
    probe.raise_for_status()
    if "accounts.google.com" in probe.url or "Google Drive: Sign-in" in probe.text:
        raise RuntimeError(
            "Google rejected the saved login session. Re-export fresh Netscape-format "
            f"cookies from the account that can open this folder: {args.cookie_file}"
        )

    install_gdown_session(args.cookie_file, args.authuser)
    args.output.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    print("Retrieving authenticated Drive inventory...", flush=True)
    inventory = list_folder(args.folder_id, args.cookie_file, args.authuser)
    if args.include:
        requested = set(args.include)
        inventory = [item for item in inventory if item[1] in requested]
        missing = sorted(requested - {item[1] for item in inventory})
        if missing:
            raise RuntimeError(f"requested Drive files were not found: {missing}")
    inventory_records = [
        {"id": file_id, "name": name, "mime": mime}
        for file_id, name, mime in inventory
    ]
    print(f"Drive inventory contains {len(inventory_records)} downloadable file(s):")
    for record in inventory_records:
        print(f"  {record['name']} [{record['mime']}]")

    if args.list_only:
        payload = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "folder_id": args.folder_id,
            "authuser": args.authuser,
            "list_only": True,
            "inventory": inventory_records,
        }
    else:
        print("Downloading/resuming authenticated Drive folder...", flush=True)
        files = []
        for file_id, name, mime in inventory:
            if mime == DRIVE_FOLDER_MIME:
                raise RuntimeError(f"nested Drive folder is outside this bounded download: {name}")
            path = download_file(file_id, args.output / name).resolve()
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"downloaded file is missing or empty: {path}")
            files.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        payload = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "folder_id": args.folder_id,
            "authuser": args.authuser,
            "list_only": False,
            "inventory": inventory_records,
            "downloaded_files": files,
            "downloaded_bytes": sum(item["bytes"] for item in files),
        }

    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Download manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
