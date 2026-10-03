"""Shared HTTP helpers: polite, retrying GETs with an on-disk cache."""
from __future__ import annotations

import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

_session = requests.Session()
_session.headers["User-Agent"] = "gooseprojections/0.1 (research; contact via github)"


def get(url: str, *, params: dict | None = None, retries: int = 4, timeout: int = 120) -> requests.Response:
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}")


def cached_text(path: Path, url: str, *, params: dict | None = None, refresh: bool = False, pause: float = 0.5) -> str:
    """Fetch url to path unless it already exists. Returns the text."""
    if path.exists() and not refresh:
        return path.read_text()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = get(url, params=params).text.lstrip("﻿")
    path.write_text(text)
    time.sleep(pause)
    return text
