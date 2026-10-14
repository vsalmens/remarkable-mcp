"""Handwriting OCR through multimodal LLMs (Gemini, Anthropic Claude).

Backends
    REMARKABLE_OCR_BACKEND=gemini     GEMINI_API_KEY (or GOOGLE_API_KEY)
    REMARKABLE_OCR_BACKEND=anthropic  ANTHROPIC_API_KEY
                                      (+ ANTHROPIC_WORKSPACE_ID for keys not
                                      scoped to a workspace)

Models (override per backend)
    REMARKABLE_GEMINI_MODEL     default: gemini-3.1-pro-preview
    REMARKABLE_ANTHROPIC_MODEL  default: claude-opus-5-5

Cache
    Every transcription is stored on disk, keyed by the SHA-256 of the page
    PNG plus backend, model and prompt version. A page is sent to an API only
    once; later reads, searches and exports reuse the stored text.
    REMARKABLE_OCR_CACHE_DIR    default: ~/.remarkable/cache/ocr
    REMARKABLE_OCR_CACHE=0      disables the cache

Usage
    Token counts of real API calls (cache hits cost nothing) are added to the
    dict in USAGE_SINK, if one is set for the current context. Transcription
    jobs use this to report tokens and an estimated cost per job.
    REMARKABLE_PRICES  optional JSON {"model": [input_usd, output_usd]} per
                       million tokens, overriding the list prices below.

Language hints come from REMARKABLE_OCR_LANGUAGES (see policy.py).
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Optional

from remarkable_mcp.policy import get_ocr_language_hints

logger = logging.getLogger(__name__)

LLM_OCR_BACKENDS = ("gemini", "anthropic")

# USD per million tokens (input, output). List prices as of October 2026;
# models not listed get token counts only. Override with REMARKABLE_PRICES.
LIST_PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}

# Usage of requests sent in a Message Batch is kept under "<model> (batch)"
# and priced at half the list price.
BATCH_SUFFIX = " (batch)"
BATCH_DISCOUNT = 0.5

USAGE_SINK: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "remarkable_ocr_usage", default=None
)
_usage_lock = threading.Lock()


def _record_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    sink = USAGE_SINK.get()
    if sink is None:
        return
    with _usage_lock:
        entry = sink.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        entry["calls"] += 1
        entry["input_tokens"] += int(input_tokens or 0)
        entry["output_tokens"] += int(output_tokens or 0)


def _prices() -> dict:
    prices = dict(LIST_PRICES)
    raw = os.environ.get("REMARKABLE_PRICES", "").strip()
    if raw:
        try:
            for model, pair in json.loads(raw).items():
                prices[model] = (float(pair[0]), float(pair[1]))
        except (ValueError, TypeError, IndexError, AttributeError):
            logger.warning("Ignoring malformed REMARKABLE_PRICES")
    return prices


def estimate_cost(usage: dict) -> Optional[float]:
    """Estimated USD for a usage dict, or None if no model in it has a price."""
    prices = _prices()
    total, priced = 0.0, False
    for model, entry in usage.items():
        base = model[: -len(BATCH_SUFFIX)] if model.endswith(BATCH_SUFFIX) else model
        if base in prices:
            pin, pout = prices[base]
            share = BATCH_DISCOUNT if base != model else 1.0
            total += share * (entry["input_tokens"] * pin + entry["output_tokens"] * pout) / 1e6
            priced = True
    return round(total, 4) if priced else None


DEFAULT_MODELS = {
    "gemini": "gemini-3.1-pro-preview",
    "anthropic": "claude-opus-5-5",
}

PROMPT_VERSION = "v1"

_LANGUAGE_NAMES = {
    "fi": "Finnish",
    "en": "English",
    "sv": "Swedish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "no": "Norwegian",
    "nb": "Norwegian",
    "da": "Danish",
    "et": "Estonian",
    "ru": "Russian",
}

BLANK_MARKER = "[blank]"


def _model_for(backend: str) -> str:
    env = f"REMARKABLE_{backend.upper()}_MODEL"
    return os.environ.get(env, "").strip() or DEFAULT_MODELS[backend]


def build_prompt() -> str:
    hints = get_ocr_language_hints()
    if hints:
        names = [_LANGUAGE_NAMES.get(h.split("-")[0].lower(), h) for h in hints]
        lang = "The writing is mostly in " + " and ".join(names) + ", often mixed."
    else:
        lang = "Keep every word in the language it was written in."
    return (
        "Transcribe the handwriting on this notebook page exactly as written. "
        f"{lang}\n"
        "Rules:\n"
        "- Output only the transcription, with no commentary or preamble.\n"
        "- Follow the reading order top to bottom; keep line breaks. "
        "Indent nested items with two spaces.\n"
        "- Do not translate, correct spelling or expand abbreviations.\n"
        "- Struck-through text: ~~text~~. Checkboxes: [ ] empty, [x] ticked. "
        "Arrows: →. Keep symbols such as ♡, ?, !.\n"
        "- A word you cannot read with confidence: give your best reading "
        "followed by (?). Wholly illegible: [?].\n"
        "- Drawings or diagrams: one short description in square brackets.\n"
        f"- If the page has no writing, output exactly {BLANK_MARKER}"
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_enabled() -> bool:
    return os.environ.get("REMARKABLE_OCR_CACHE", "1").strip().lower() not in ("0", "false", "no")


def _cache_dir() -> Path:
    raw = os.environ.get("REMARKABLE_OCR_CACHE_DIR", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".remarkable" / "cache" / "ocr"


def _cache_path(png: bytes, backend: str, model: str, prompt: str) -> Path:
    h = hashlib.sha256()
    h.update(png)
    h.update(b"\0" + backend.encode() + b"\0" + model.encode())
    h.update(b"\0" + PROMPT_VERSION.encode() + b"\0" + prompt.encode())
    digest = h.hexdigest()
    return _cache_dir() / backend / digest[:2] / f"{digest}.txt"


def _cache_get(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _cache_put(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name: parallel workers may cache the same page at once
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("OCR cache write failed: %s", e)


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def _call_gemini(png: bytes, model: str, prompt: str, timeout: float) -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        logger.warning("Gemini OCR selected but GEMINI_API_KEY is not set")
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(png).decode("ascii"),
                        }
                    },
                    {"text": prompt},
                ],
            }
        ]
    }
    resp = _post_with_retry(url, json=body, headers={"x-goog-api-key": key}, timeout=timeout)
    if resp.status_code != 200:
        logger.warning(
            "Gemini OCR HTTP %s: %s", resp.status_code, " ".join(resp.text[:400].split())
        )
        return None
    data = resp.json()
    meta = data.get("usageMetadata") or {}
    _record_usage(
        model,
        meta.get("promptTokenCount", 0),
        meta.get("candidatesTokenCount", 0) + meta.get("thoughtsTokenCount", 0),
    )
    texts = []
    for cand in data.get("candidates", [])[:1]:
        for part in cand.get("content", {}).get("parts", []):
            if part.get("thought"):
                continue
            if "text" in part:
                texts.append(part["text"])
    return "".join(texts) if texts else None


def _call_anthropic(png: bytes, model: str, prompt: str, timeout: float) -> Optional[str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning("Anthropic OCR selected but ANTHROPIC_API_KEY is not set")
        return None
    body = {
        "model": model,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(png).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    if workspace:
        # Required when the API key is not scoped to a single workspace
        headers["anthropic-workspace-id"] = workspace
    from remarkable_mcp import batching

    data, billed = None, model
    batcher = batching.BATCHER.get()
    if batcher is not None:  # a library sync: sent with others as a Message Batch
        data = batcher.request(body)
        billed = model + BATCH_SUFFIX
    if data is None:
        resp = _post_with_retry(
            "https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=timeout
        )
        if resp.status_code != 200:
            logger.warning(
                "Anthropic OCR HTTP %s: %s", resp.status_code, " ".join(resp.text[:400].split())
            )
            return None
        data, billed = resp.json(), model
    used = data.get("usage") or {}
    _record_usage(
        billed,
        used.get("input_tokens", 0)
        + used.get("cache_creation_input_tokens", 0)
        + used.get("cache_read_input_tokens", 0),
        used.get("output_tokens", 0),
    )
    texts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    return "".join(texts) if texts else None


_CALLERS = {"gemini": _call_gemini, "anthropic": _call_anthropic}

_RETRY_STATUS = {429, 500, 502, 503, 504, 529}


def _post_with_retry(url, *, json, headers, timeout, attempts: Optional[int] = None):
    """POST with exponential backoff on rate limits and transient errors."""
    import time

    import requests

    if attempts is None:
        try:
            attempts = max(1, int(os.environ.get("REMARKABLE_OCR_RETRIES", "4")))
        except ValueError:
            attempts = 4
    resp = None
    for i in range(attempts):
        try:
            resp = requests.post(url, json=json, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as e:
            if i == attempts - 1:
                raise
            logger.info("OCR request error (%s); retrying", e)
            time.sleep(2 ** (i + 1))
            continue
        if resp.status_code not in _RETRY_STATUS or i == attempts - 1:
            return resp
        retry_after = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
        try:
            delay = min(float(retry_after), 60.0) if retry_after else 2 ** (i + 1)
        except (TypeError, ValueError):
            delay = 2 ** (i + 1)
        logger.info("OCR HTTP %s; retrying in %.0fs", resp.status_code, delay)
        time.sleep(delay)
    return resp


def has_credentials(backend: str) -> bool:
    if backend == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if backend == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return False


def backend_for_model(model: str) -> Optional[str]:
    """Which backend serves a model id ("gemini-…" or "claude-…")."""
    m = (model or "").strip().lower()
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("claude"):
        return "anthropic"
    return None


def ocr_png(
    png: bytes, backend: str, timeout: float = 120.0, model: Optional[str] = None
) -> Optional[str]:
    """Transcribe one page PNG.

    Returns the text, "" when the model reports a blank page, or None when the
    backend is unavailable or the request failed (callers may then fall back).
    ``model`` overrides the configured model for this call; the cache keeps
    every model's readings apart.
    """
    if backend not in _CALLERS:
        raise ValueError(f"Unknown LLM OCR backend: {backend}")
    model = (model or "").strip() or _model_for(backend)
    prompt = build_prompt()

    path = _cache_path(png, backend, model, prompt) if _cache_enabled() else None
    text = _cache_get(path) if path else None

    if text is None:
        try:
            text = _CALLERS[backend](png, model, prompt, timeout)
        except Exception as e:  # network errors, JSON errors
            logger.warning("%s OCR failed: %s", backend, e)
            return None
        if text is None:
            return None
        text = text.strip()
        if path:
            _cache_put(path, text)

    text = text.strip()
    if not text or text == BLANK_MARKER:
        return ""
    return text


def ocr_png_file(png_path: Path, backend: str) -> Optional[str]:
    return ocr_png(Path(png_path).read_bytes(), backend)


# ---------------------------------------------------------------------------
# Dual reading: resolve disagreements against the image
# ---------------------------------------------------------------------------

ADJUDICATE_VERSION = "v3"

# Uncertainty markup in adjudicated (dual) transcriptions. The same inline
# notation is used for speech transcripts (Jefferson-style conventions):
#   {likelier|other}     two (or three) plausible readings, likeliest first
#   word (?)             a single reading with low confidence
#   [?: w1 w2 w3]        a run of consecutive doubtful words (best guesses kept)
#   word→suggestion (?)  the image shows "word"; context or the writer's
#                        vocabulary suggests "suggestion" (never silently replaced)
#   [?]                  illegible


def _vocabulary_file() -> Path:
    raw = os.environ.get("REMARKABLE_OCR_VOCABULARY_FILE", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".remarkable" / "ocr-vocabulary.txt"


def get_vocabulary() -> list:
    """Names and terms the writer uses, from REMARKABLE_OCR_VOCABULARY and a file.

    The file (default ~/.remarkable/ocr-vocabulary.txt) is read on every call,
    so it can be edited without restarting the server. One term per line or
    comma-separated; lines starting with # are ignored.
    """
    terms = []
    sources = [os.environ.get("REMARKABLE_OCR_VOCABULARY", "")]
    try:
        sources.append(_vocabulary_file().read_text(encoding="utf-8"))
    except OSError:
        pass
    for source in sources:
        for line in source.splitlines() or [source]:
            line = line.split("#", 1)[0]
            for term in line.split(","):
                term = term.strip()
                if term and term not in terms:
                    terms.append(term)
    return terms


CONTEXT_CHARS = 2000


def _printed_block(printed: Optional[str]) -> str:
    """The printed passages a PDF page's notes were written beside, as a hint."""
    printed = (printed or "").strip()
    if not printed:
        return ""
    return (
        "\n\nThe handwriting was written on a printed page (a PDF). The passages the "
        "reader marked there are below, for context only; do not transcribe them. A "
        "note often names or questions what these passages say, so they may suggest a "
        "word, but never override the strokes: if the context points to a different "
        "reading than the image shows, keep what the image supports and add the "
        "context's reading as a visible suggestion: word→suggestion (?).\n"
        f"<Printed passages>\n{printed[:CONTEXT_CHARS]}\n</Printed passages>"
    )


def _context_block(context: Optional[dict]) -> str:
    """Neighbouring pages, so names and repeated words can be read consistently."""
    if not context:
        return ""
    printed = _printed_block(context.get("printed"))
    if printed:
        rest = {k: v for k, v in context.items() if k != "printed"}
        return _context_block(rest) + printed
    parts = []
    for key, label in (("previous", "Previous page"), ("next", "Next page")):
        text = (context.get(key) or "").strip()
        if text:
            parts.append(f"<{label}>\n{text[:CONTEXT_CHARS]}\n</{label}>")
    if not parts:
        return ""
    return (
        "\n\nNeighbouring pages of the same notebook, as read by one reader, for "
        "context only. Do not transcribe them. Use them to resolve names, recurring "
        "words and this writer's spelling, e.g. a signature that matches a heading on "
        "the previous page. Context may suggest but never override: an unusual word "
        "or a break in the train of thought may be exactly what the writer meant. If "
        "the context points to a different reading than the image shows, keep the "
        "word the image supports and add the context's reading as a visible "
        "suggestion: word→suggestion (?).\n" + "\n".join(parts)
    )


def build_adjudication_prompt(readings: dict, context: Optional[dict] = None) -> str:
    labels = {"anthropic": "Reader A", "gemini": "Reader B"}
    blocks = []
    for backend, text in readings.items():
        if text is None:
            continue
        blocks.append(
            f"<{labels.get(backend, backend)}>\n{text}\n</{labels.get(backend, backend)}>"
        )
    return (
        build_prompt()
        + "\n\nTwo independent readers already transcribed this page. Their readings "
        "are below. Produce the single most faithful transcription by checking the "
        "image itself: where the readers disagree, follow what the handwriting "
        "actually shows, not either reader by default. Apply strike-through only to "
        "text that is visibly struck through in the image.\n"
        "Mark uncertainty so that no plausible reading is lost:\n"
        "- Where the readers read a word or short phrase differently and the image "
        "does not clearly settle it, write both readings as {likelier|other}, the "
        "likelier one first. Use the same form when you yourself see two plausible "
        "readings. At most three options: {a|b|c}. Keep the braces tight around the "
        "differing word(s) only.\n"
        "- A single reading you are unsure of: the word followed by (?).\n"
        "- Several consecutive words you are unsure of: [?: word word word], keeping "
        "your best guesses inside.\n"
        "- Wholly illegible: [?].\n"
        "- Where the image clearly shows one reading, write it plainly with no marker, "
        "even if the readers disagreed.\n"
        + _vocabulary_hint()
        + "Output only the final transcription.\n\n"
        + "\n\n".join(blocks)
        + _context_block(context)
    )


def _vocabulary_hint() -> str:
    terms = get_vocabulary()
    if not terms:
        return ""
    return (
        "Names and terms this writer often uses (a hint, not a rule): "
        + ", ".join(terms)
        + ". Where the handwriting fits one of these at least as well as the readers' "
        "word, include it as an option, e.g. {Anna|anne}. Where only the list suggests "
        "it, keep the word the image shows and add the term as a visible suggestion: "
        "anne→Anna (?). Never replace a word with a term silently.\n"
    )


def _adjudication_cache_path(png: bytes, backend: str, model: str, prompt: str) -> Path:
    h = hashlib.sha256()
    h.update(png)
    h.update(b"\0adjudicate\0" + ADJUDICATE_VERSION.encode())
    h.update(b"\0" + backend.encode() + b"\0" + model.encode() + b"\0" + prompt.encode())
    digest = h.hexdigest()
    return _cache_dir() / "adjudicated" / digest[:2] / f"{digest}.txt"


def adjudicate_png(
    png: bytes,
    readings: dict,
    timeout: float = 180.0,
    context: Optional[dict] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """Final transcription from the image plus two readings (cached).

    ``context`` may hold the "previous" and "next" page text. ``model`` picks the
    judge for this call (a gemini-... or claude-... id); otherwise
    REMARKABLE_ADJUDICATOR chooses the backend and its configured model.
    """
    backend = backend_for_model(model) if model else None
    if backend is None:
        backend = os.environ.get("REMARKABLE_ADJUDICATOR", "anthropic").strip().lower()
        if backend not in _CALLERS:
            backend = "anthropic"
        model = None
    model = (model or "").strip() or _model_for(backend)
    prompt = build_adjudication_prompt(readings, context)
    path = None
    if _cache_enabled():
        # The page's own result is keyed without the neighbouring pages, so adding
        # or removing a page does not re-run the adjudication of its neighbours.
        path = _adjudication_cache_path(png, backend, model, build_adjudication_prompt(readings))
        cached = _cache_get(path)
        if cached is None:
            # Entries written before this key existed included the context.
            legacy = _adjudication_cache_path(png, backend, model, prompt)
            cached = _cache_get(legacy)
            if cached is not None:
                _cache_put(path, cached)
        if cached is not None:
            return cached
    try:
        text = _CALLERS[backend](png, model, prompt, timeout)
    except Exception as e:
        logger.warning("Adjudication failed: %s", e)
        return None
    if text is None:
        return None
    text = text.strip()
    if path:
        _cache_put(path, text)
    return text


# ---------------------------------------------------------------------------
# Annotations on PDF pages: tie marks, notes and symbols to the printed text
# ---------------------------------------------------------------------------

COMPOSE_VERSION = "c2"

COMPOSE_PROMPT = """You see a printed page (from a PDF) that a reader annotated with a pen: \
dark ink over the print. You also get (A) the page's printed text from the PDF text layer, \
in blocks; (B) passages a geometric pass found marked, marked words in **bold** (a hint, \
it can miss or over-reach); (C) the transcription of the reader's handwriting on this page.

Write down the reader's annotations as JSON, nothing else:
{{"items": [...], "unplaced_notes": ["..."]}}

Each item is either a passage or a table.

Passage: {{"type": "passage", "quote": "...", "note": "...", "symbols": ["?"], "label": "I"}}
- quote: the whole sentence or sentences around the mark, copied exactly from (A), with the \
words the pen marked (underlined, struck through, highlighted, circled) wrapped in **...**. \
Keep sentences whole. Put neighbouring marked sentences that belong together in one quote. \
When a vertical line, bracket or brace in the margin marks a paragraph, quote the whole \
paragraph and wrap all of it in **...**.
- note: the handwriting that belongs to this passage, copied exactly from (C) with its \
notation: words written beside or between its lines, joined to it by a line or arrow, or \
written near the reference it comments on. Leave it out when there is none.
- symbols: marks the reader drew that are not words, such as "?", "!", "!!", "*", a tick; \
attached to the passage or term they stand next to. Leave it out when there are none.
- label: a label the reader wrote at the passage, such as I, II, 1 or *1, especially when the \
same label appears in a note; without brackets. Leave it out when there is none.
- pen: the pen given in (B) for the mark this passage comes from, copied as is. Leave it out \
when (B) gives none.

Table: {{"type": "table", "caption": "...", "headers": ["..."], "rows": [["cell", "..."], "…"], \
"symbols": [], "note": "..."}}
- for marks inside a table: its caption (e.g. "Table 1. Summary of ..."), its column headers, \
and only the rows the reader marked, in order, as lists of cells copied from (A); put the \
string "…" where rows were left out (also at the start or end). Wrap the marked cells or \
words in **...**; a ring drawn around or next to words marks the nearest group of words. \
A symbol drawn next to a row goes at the end of that row's first cell after "✎", e.g. \
"Questions ✎?".

Rules: quote only text that is in (A); never paraphrase, translate or correct it. Take \
notes only from (C); use the image to decide where each note belongs, not to re-read it. \
Every note in (C) appears exactly once, in an item or in unplaced_notes. Return JSON only.

(A) Printed text:
{blocks}

(B) Marked passages found geometrically:
{marks}

(C) Handwriting on the page:
{handwriting}
"""


def build_compose_prompt(blocks: list, marks: list, handwriting: Optional[str]) -> str:
    block_text = "\n".join(f"[{i + 1}] {b}" for i, b in enumerate(blocks)) or "(none)"
    lines = []
    for m in marks:
        if not isinstance(m, dict):
            lines.append(f"- {m}")
            continue
        pen = m.get("pen") or {}
        pen_text = ", ".join(
            str(v)
            for v in (
                pen.get("tool"),
                pen.get("color"),
                pen.get("width_mm") and f"{pen['width_mm']} mm",
            )
            if v
        )
        lines.append(f"- {m.get('text', '')}" + (f" (pen: {pen_text})" if pen_text else ""))
    mark_text = "\n".join(lines)
    return COMPOSE_PROMPT.format(
        blocks=block_text,
        marks=mark_text or "(none)",
        handwriting=(handwriting or "").strip() or "(none)",
    )


def _parse_json_object(text: str) -> Optional[dict]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def compose_annotations(
    png: bytes,
    blocks: list,
    marks: list,
    handwriting: Optional[str],
    timeout: float = 180.0,
    model: Optional[str] = None,
) -> Optional[dict]:
    """Annotations of a PDF page as {"items", "unplaced_notes"} (cached), or None.

    The model sees the page with its ink, the printed text, the geometric
    marks and the handwriting transcript, and ties notes, symbols and labels
    to the passages they belong to. REMARKABLE_ANNOTATION_MODEL picks the
    model (default: the configured Claude model).
    """
    model = (model or os.environ.get("REMARKABLE_ANNOTATION_MODEL", "")).strip()
    backend = backend_for_model(model) if model else "anthropic"
    if backend is None or not has_credentials(backend):
        return None
    model = model or _model_for(backend)
    prompt = build_compose_prompt(blocks, marks, handwriting)
    path = None
    if _cache_enabled():
        h = hashlib.sha256()
        h.update(png)
        h.update(f"\0compose\0{COMPOSE_VERSION}\0{backend}\0{model}\0".encode())
        h.update(prompt.encode())
        digest = h.hexdigest()
        path = _cache_dir() / "composed" / digest[:2] / f"{digest}.json"
        cached = _cache_get(path)
        if cached is not None:
            return _parse_json_object(cached)
    try:
        text = _CALLERS[backend](png, model, prompt, timeout)
    except Exception as e:
        logger.warning("Composing annotations failed: %s", e)
        return None
    if not text:
        return None
    data = _parse_json_object(text)
    if data is not None and path:
        _cache_put(path, json.dumps(data, ensure_ascii=False))
    return data


# ---------------------------------------------------------------------------
# Blind vote on what the final pass left open
# ---------------------------------------------------------------------------

VOTE_VERSION = "v1"

# {likelier|other} or {a|b|c}: options the final pass could not settle.
ALTERNATIVES_RE = re.compile(r"\{([^{}|\n]+(?:\|[^{}|\n]+){1,2})\}")

VOTE_PROMPT = """You see a page of handwriting and its transcription. At some spots \
the transcription could not decide between readings; each spot is marked ⟦n⟧ in the \
text and its options are listed below. For each spot, look at the strokes at that place \
and choose the option they show.

- Judge by the letters: their number, height, loops, dots and crossings.
- A real word or a name fits better than a string of letters only when the strokes fit \
it at least as well.
- Answer "?" when the strokes do not clearly favour one option.

Return JSON only, one entry per spot: {{"1": "A", "2": "?"}}

Transcription:
{text}

Options:
{options}{hint}
"""


def _vote_slots(text: str, rng) -> tuple:
    """The text with each {a|b} replaced by ⟦n⟧, and per spot the shuffled options."""
    spots = []

    def slot(m):
        options = [o.strip() for o in m.group(1).split("|")]
        order = list(range(len(options)))
        rng.shuffle(order)
        spots.append((options, order))
        return f"⟦{len(spots)}⟧"

    return ALTERNATIVES_RE.sub(slot, text), spots


def _vote_once(png, backend, model, text, printed, timeout) -> Optional[list]:
    """One reader's choice per spot (index into the options, or None), cached."""
    import random

    seed = hashlib.sha256(f"{backend}\0{model}\0{text}".encode()).hexdigest()
    slotted, spots = _vote_slots(text, random.Random(seed))
    if not spots:
        return []
    letters = "ABC"
    options = "\n".join(
        f"{i}: " + "  ".join(f"{letters[k]} = {opts[j]}" for k, j in enumerate(order))
        for i, (opts, order) in enumerate(spots, 1)
    )
    hint = _printed_block(printed)
    prompt = VOTE_PROMPT.format(text=slotted, options=options, hint=hint)
    path = None
    if _cache_enabled():
        h = hashlib.sha256()
        h.update(png)
        h.update(f"\0vote\0{VOTE_VERSION}\0{backend}\0{model}\0".encode())
        h.update(prompt.encode())
        digest = h.hexdigest()
        path = _cache_dir() / "votes" / digest[:2] / f"{digest}.json"
        raw = _cache_get(path)
    else:
        raw = None
    if raw is None:
        try:
            raw = _CALLERS[backend](png, model, prompt, timeout)
        except Exception as e:
            logger.warning("Vote by %s failed: %s", backend, e)
            return None
        if raw is None:
            return None
        if path:
            _cache_put(path, raw)
    data = _parse_json_object(raw)
    if data is None:
        return None
    picks = []
    for i, (opts, order) in enumerate(spots, 1):
        answer = str(data.get(str(i), "?")).strip().upper()[:1]
        k = letters.find(answer) if answer else -1
        picks.append(order[k] if 0 <= k < len(order) else None)
    return picks


def vote_alternatives(
    png: bytes,
    text: str,
    models: Optional[dict] = None,
    printed: Optional[str] = None,
    timeout: float = 120.0,
) -> str:
    """Settle {a|b} spots that both readers, voting blind, read the same way.

    Each reader sees the image, the text with the open spots numbered and the
    options in a shuffled order, without knowing who proposed which. A spot is
    replaced by an option only when every reader chose it; otherwise it stays
    open. ``models`` maps each backend to its model (None: configured model).
    REMARKABLE_VOTE=0 turns the vote off.
    """
    if os.environ.get("REMARKABLE_VOTE", "1").strip().lower() in ("0", "false", "no", ""):
        return text
    if not text or not ALTERNATIVES_RE.search(text):
        return text
    models = models or {b: None for b in LLM_OCR_BACKENDS}
    voters = [b for b in models if b in _CALLERS and has_credentials(b)]
    if len(voters) < 2:
        return text
    picks = [
        _vote_once(png, b, (models.get(b) or "").strip() or _model_for(b), text, printed, timeout)
        for b in voters
    ]
    if any(p is None for p in picks):
        return text
    counter = iter(range(len(picks[0])))

    def settle(m):
        i = next(counter)
        choices = {p[i] for p in picks}
        if len(choices) == 1 and None not in choices:
            return [o.strip() for o in m.group(1).split("|")][choices.pop()]
        return m.group(0)

    return ALTERNATIVES_RE.sub(settle, text)
