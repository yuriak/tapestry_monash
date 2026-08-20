#!/usr/bin/env python3
"""Download authenticated Google Drive files from a selected login slot.

gdown handles confirmation pages and partial files well, but its private-folder
endpoint does not preserve Google's multi-account ``authuser`` selection. This
adapter keeps gdown for file transfer while reading the authenticated folder
inventory embedded in the normal Drive page.
"""

from __future__ import annotations

import argparse
import html
import importlib
import json
import os
import re
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import gdown
import requests


DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
IVD_PATTERN = re.compile(r"window\['_DRIVE_ivd'\]\s*=\s*'([^']*)';")


class AuthuserSession(requests.Session):
    """Requests session that retains Google's selected multi-login account."""

    def __init__(self, authuser: int) -> None:
        super().__init__()
        self.authuser = str(authuser)

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        parsed = urlsplit(url)
        if parsed.hostname == "drive.google.com":
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
            query.setdefault("authuser", self.authuser)
            url = urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
            )
        return super().request(method, url, **kwargs)


def load_session(cookie_file: Path, authuser: int) -> AuthuserSession:
    session = AuthuserSession(authuser)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/140 Safari/537.36"
            )
        }
    )
    jar = MozillaCookieJar(str(cookie_file))
    jar.load(ignore_discard=True, ignore_expires=False)
    session.cookies.update(jar)
    return session


def install_gdown_session(cookie_file: Path, authuser: int) -> None:
    download_module = importlib.import_module("gdown.download")
    folder_module = importlib.import_module("gdown.download_folder")

    def get_session(
        proxy: str | None,
        use_cookies: bool,
        user_agent: str,
    ) -> tuple[requests.Session, str]:
        session = load_session(cookie_file, authuser)
        session.headers.update({"User-Agent": user_agent})
        if proxy is not None:
            session.proxies = {"http": proxy, "https": proxy}
        if not use_cookies:
            session.cookies.clear()
        return session, str(cookie_file)

    download_module._get_session = get_session
    folder_module._get_session = get_session


def decode_inventory(page: str) -> Any:
    match = IVD_PATTERN.search(page)
    if match is None:
        raise RuntimeError("Drive page did not contain an authenticated folder inventory")
    escaped = html.unescape(match.group(1))
    decoded = re.sub(
        r"\\x([0-9a-fA-F]{2})",
        lambda item: chr(int(item.group(1), 16)),
        escaped,
    )
    return json.loads(decoded)


def walk(value: Any) -> Iterator[list[Any]]:
    if not isinstance(value, list):
        return
    if (
        len(value) >= 4
        and isinstance(value[0], str)
        and isinstance(value[2], str)
        and isinstance(value[3], str)
        and value[3].startswith("application/")
    ):
        yield value
        return
    for child in value:
        yield from walk(child)


def list_folder(
    folder_id: str,
    cookie_file: Path,
    authuser: int,
) -> list[tuple[str, str, str]]:
    session = load_session(cookie_file, authuser)
    url = f"https://drive.google.com/drive/u/{authuser}/folders/{folder_id}"
    response = session.get(url, timeout=90)
    response.raise_for_status()
    records: dict[str, tuple[str, str, str]] = {}
    for record in walk(decode_inventory(response.text)):
        file_id, name, mime = record[0], record[2], record[3]
        if not name or Path(name).name != name or name in {".", ".."}:
            raise RuntimeError(f"Unsafe Drive item name: {name!r}")
        records[file_id] = (file_id, name, mime)
    if not records:
        raise RuntimeError(f"No files found in authenticated Drive folder {folder_id}")
    return sorted(records.values(), key=lambda item: item[1])


def validate_download(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Drive download is empty or missing: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(256).lstrip().lower()
    if prefix.startswith((b"<!doctype html", b"<html")):
        raise RuntimeError(f"Drive returned an HTML error page: {path}")


def download_file(file_id: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=file_id, output=str(output), resume=True)
    if not isinstance(result, str):
        raise RuntimeError(f"gdown did not return a path for Drive file {file_id}")
    path = Path(result)
    validate_download(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("file", "folder"))
    parser.add_argument("--id", required=True, dest="drive_id")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cookie-file", required=True, type=Path)
    parser.add_argument("--authuser", type=int, default=2)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    if not args.cookie_file.is_file() or args.cookie_file.stat().st_size == 0:
        parser.error(f"missing cookie file: {args.cookie_file}")
    install_gdown_session(args.cookie_file, args.authuser)

    if args.kind == "file":
        if args.list_only:
            parser.error("--list-only is valid only for a folder")
        args.output.mkdir(parents=True, exist_ok=True)
        session = load_session(args.cookie_file, args.authuser)
        response = session.get(
            f"https://drive.google.com/file/d/{args.drive_id}/view",
            timeout=90,
        )
        response.raise_for_status()
        title = re.search(r"<title>(.*?)\s+-\s+Google Drive</title>", response.text, re.S)
        if title is None:
            raise RuntimeError("Could not resolve the authenticated Drive filename")
        filename = html.unescape(title.group(1)).strip()
        if Path(filename).name != filename:
            raise RuntimeError(f"Unsafe Drive filename: {filename!r}")
        path = download_file(args.drive_id, args.output / filename)
        print(f"Downloaded authenticated Drive file: {path}")
        return 0

    items = list_folder(args.drive_id, args.cookie_file, args.authuser)
    print(f"Authenticated folder inventory: {len(items)} item(s)")
    for file_id, name, mime in items:
        print(f"  {name} [{mime}]")
        if args.list_only:
            continue
        if mime == DRIVE_FOLDER_MIME:
            raise RuntimeError("Nested Drive folders are not supported by this bounded downloader")
        download_file(file_id, args.output / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
