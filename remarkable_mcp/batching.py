"""Anthropic Message Batches for library syncs: the same requests at half price.

A sync does not need its answers within seconds, so its Claude requests are
gathered and sent as one Message Batch instead of one call each. Nothing else
changes: the page pipeline still calls ``llm_ocr._call_anthropic``, which
hands the request body to the batcher of the current context and blocks
until the batch has ended. Requests the batch could not answer are then made
as ordinary calls, so a failed batch costs time, never a page.

When a batch is sent
    As soon as every page that is being worked on is waiting for an answer
    (nothing more can join), when MAX_REQUESTS or MAX_BYTES are reached, or
    when the oldest waiting request is older than REMARKABLE_BATCH_MAX_WAIT
    seconds (default 120).

Interactive tools (remarkable_transcribe, remarkable_read) never use it.
REMARKABLE_BATCH=0 turns it off for syncs as well.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

logger = logging.getLogger(__name__)

API = "https://api.anthropic.com/v1/messages/batches"
MAX_REQUESTS = 2000
MAX_BYTES = 150 * 1024 * 1024  # the API takes up to 256 MB per batch
POLL_FIRST = 15.0
POLL_MAX = 60.0

BATCHER: contextvars.ContextVar[Optional["AnthropicBatcher"]] = contextvars.ContextVar(
    "remarkable_anthropic_batcher", default=None
)


class BatchCancelled(Exception):
    """The job waiting for this answer was cancelled."""


def enabled() -> bool:
    return os.environ.get("REMARKABLE_BATCH", "1").strip().lower() not in ("0", "false", "no", "")


def pages_in_flight() -> int:
    """Pages of one document worked on at once in a batched job (default 48)."""
    try:
        return max(1, int(os.environ.get("REMARKABLE_BATCH_PAGES", "48")))
    except ValueError:
        return 48


def _max_wait() -> float:
    try:
        return max(5.0, float(os.environ.get("REMARKABLE_BATCH_MAX_WAIT", "120")))
    except ValueError:
        return 120.0


def _headers() -> dict:
    headers = {
        "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    if workspace:
        headers["anthropic-workspace-id"] = workspace
    return headers


class _Request:
    __slots__ = ("id", "body", "size", "created", "event", "result", "cancelled")

    def __init__(self, body: dict):
        self.id = "r" + uuid.uuid4().hex[:20]
        self.body = body
        self.size = len(json.dumps(body))
        self.created = time.monotonic()
        self.event = threading.Event()
        self.result: Optional[dict] = None  # the Message, or None: call it directly
        self.cancelled = False


class AnthropicBatcher:
    """Collects Messages API request bodies from many threads into batches.

    ``active`` counts the units of work (pages) that may still add a request.
    A page is entered when its work starts and left when it ends; while its
    thread waits for an answer it does not count, so once nothing is active
    the batch is complete and goes out.
    """

    def __init__(self, session=None, poll_first: float = POLL_FIRST, poll_max: float = POLL_MAX):
        self._lock = threading.Lock()
        self._pending: List[_Request] = []
        self._waiting: set = set()
        self._active = 0
        self._session = session
        self._poll_first = poll_first
        self._poll_max = poll_max
        self._timer: Optional[threading.Thread] = None
        self.batches_sent = 0

    # -- page accounting (called from the event loop) -------------------------

    def enter(self) -> None:
        with self._lock:
            self._active += 1

    def leave(self) -> None:
        with self._lock:
            self._active -= 1
            self._maybe_flush()

    # -- requests (called from worker threads) --------------------------------

    def request(self, body: dict) -> Optional[dict]:
        """The Message for ``body`` once its batch has ended, or None on failure."""
        req = _Request(body)
        with self._lock:
            self._pending.append(req)
            self._waiting.add(req)
            self._active -= 1
            self._ensure_timer()
            self._maybe_flush()
        req.event.wait()
        with self._lock:
            self._waiting.discard(req)
            self._active += 1
        if req.cancelled:
            raise BatchCancelled()
        return req.result

    def cancel_all(self) -> None:
        """Release every waiting thread (server shutdown or a cancelled sync)."""
        with self._lock:
            waiting, self._pending = list(self._waiting), []
        for req in waiting:
            req.cancelled = True
            req.event.set()

    # -- sending ---------------------------------------------------------------

    def _maybe_flush(self, force: bool = False) -> None:
        """Send the pending requests if the batch is complete (lock held)."""
        if not self._pending:
            return
        size = sum(r.size for r in self._pending)
        if force or self._active <= 0 or len(self._pending) >= MAX_REQUESTS or size >= MAX_BYTES:
            batch, self._pending = self._pending, []
            threading.Thread(target=self._run, args=(batch,), daemon=True).start()

    def _ensure_timer(self) -> None:
        if self._timer is not None and self._timer.is_alive():
            return

        def tick():
            while True:
                time.sleep(5)
                with self._lock:
                    if not self._pending and not self._waiting:
                        self._timer = None
                        return
                    if self._pending and time.monotonic() - self._pending[0].created > _max_wait():
                        self._maybe_flush(force=True)

        self._timer = threading.Thread(target=tick, daemon=True)
        self._timer.start()

    def _http(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _run(self, batch: List[_Request]) -> None:
        results = {}
        try:
            results = self._send_and_wait(batch)
        except Exception as e:  # the requests are made directly instead
            logger.warning("Message batch failed (%d requests): %s", len(batch), e)
        for req in batch:
            req.result = results.get(req.id)
            req.event.set()

    def _send_and_wait(self, batch: List[_Request]) -> dict:
        http = self._http()
        body = {"requests": [{"custom_id": r.id, "params": r.body} for r in batch]}
        resp = http.post(API, json=body, headers=_headers(), timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {' '.join(resp.text[:300].split())}")
        info = resp.json()
        batch_id = info["id"]
        self.batches_sent += 1
        logger.info("Message batch %s sent with %d requests", batch_id, len(batch))
        delay = self._poll_first
        while info.get("processing_status") != "ended":
            if all(r.cancelled for r in batch):
                http.post(f"{API}/{batch_id}/cancel", headers=_headers(), timeout=60)
                return {}
            time.sleep(delay)
            delay = min(self._poll_max, delay * 1.5)
            resp = http.get(f"{API}/{batch_id}", headers=_headers(), timeout=60)
            if resp.status_code == 200:
                info = resp.json()
        url = info.get("results_url") or f"{API}/{batch_id}/results"
        resp = http.get(url, headers=_headers(), timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(f"results HTTP {resp.status_code}")
        results = {}
        for line in resp.text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            result = item.get("result") or {}
            if result.get("type") == "succeeded":
                results[item.get("custom_id")] = result.get("message")
            else:
                logger.info("Batch request %s: %s", item.get("custom_id"), result.get("type"))
        logger.info("Message batch %s ended: %d/%d answered", batch_id, len(results), len(batch))
        return results


_SHARED: Optional[AnthropicBatcher] = None
_SHARED_LOCK = threading.Lock()
_EXECUTOR: Optional[ThreadPoolExecutor] = None


def shared() -> AnthropicBatcher:
    """One batcher for the whole server, so parallel documents share batches."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = AnthropicBatcher()
        return _SHARED


def executor() -> ThreadPoolExecutor:
    """Threads for batched calls: they mostly wait, so there are many of them,
    kept apart from the default pool that renders pages."""
    global _EXECUTOR
    with _SHARED_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=256, thread_name_prefix="rm-batch")
        return _EXECUTOR
