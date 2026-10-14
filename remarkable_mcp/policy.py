"""Access and OCR policy helpers driven by environment variables.

REMARKABLE_EXCLUDE_PATHS
    Comma-separated list of folder or document paths that the server must
    never expose. Matching is case-insensitive and covers the path itself and
    everything below it, e.g. ``/Private,/Work/Clients``. Excluded items are
    hidden from browse, search, recent, read, image, export, the canvas and the
    registered MCP resources. This complements REMARKABLE_ROOT_PATH, which can
    only narrow the server to a single folder.

REMARKABLE_OCR_LANGUAGES
    Comma-separated BCP-47 language hints passed to Google Cloud Vision, e.g.
    ``fi,en``. Hints stop Vision from guessing an unrelated script for messy
    handwriting. Unset means no hints (Vision auto-detects).
"""

from __future__ import annotations

import os
from typing import List, Optional


def _normalize(path: str) -> str:
    path = path.strip()
    if not path:
        return ""
    if not path.startswith("/"):
        path = "/" + path
    path = path.rstrip("/")
    return path.lower()


def get_excluded_paths() -> List[str]:
    """Return normalized, lower-cased excluded paths (never includes '/')."""
    raw = os.environ.get("REMARKABLE_EXCLUDE_PATHS", "")
    paths = []
    for part in raw.split(","):
        norm = _normalize(part)
        if norm and norm not in paths:
            paths.append(norm)
    return paths


def is_excluded(path: Optional[str]) -> bool:
    """True when ``path`` equals or lies below an excluded path."""
    if not path:
        return False
    excluded = get_excluded_paths()
    if not excluded:
        return False
    norm = _normalize(path)
    if not norm:
        return False
    return any(norm == ex or norm.startswith(ex + "/") for ex in excluded)


def get_ocr_language_hints() -> List[str]:
    """Return Google Vision language hints from REMARKABLE_OCR_LANGUAGES."""
    raw = os.environ.get("REMARKABLE_OCR_LANGUAGES", "")
    hints = []
    for part in raw.split(","):
        code = part.strip()
        if code and code not in hints:
            hints.append(code)
    return hints


def vision_request(image_content: str) -> dict:
    """Build one Google Vision REST request for handwriting OCR."""
    request = {
        "image": {"content": image_content},
        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
    }
    hints = get_ocr_language_hints()
    if hints:
        request["imageContext"] = {"languageHints": hints}
    return request
