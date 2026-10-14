"""Message Batches for library syncs."""

import json
import threading

import pytest

from remarkable_mcp import batching, llm_ocr


class FakeResp:
    def __init__(self, status=200, data=None, text=""):
        self.status_code, self._data, self.text = status, data, text

    def json(self):
        return self._data


class FakeSession:
    """Answers every request of a batch with its own custom_id."""

    def __init__(self, fail=False, skip=()):
        self.fail, self.skip, self.batches = fail, set(skip), []

    def post(self, url, json=None, headers=None, timeout=None):
        if self.fail:
            return FakeResp(500, text="down")
        self.batches.append(json["requests"])
        return FakeResp(200, {"id": "b1", "processing_status": "in_progress"})

    def get(self, url, headers=None, timeout=None):
        if url.endswith("/results") or "results" in url:
            lines = []
            for i, r in enumerate(self.batches[-1]):
                if i in self.skip:
                    lines.append({"custom_id": r["custom_id"], "result": {"type": "errored"}})
                    continue
                msg = {
                    "content": [{"type": "text", "text": "answer " + r["params"]["tag"]}],
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                }
                lines.append(
                    {"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": msg}}
                )
            return FakeResp(200, text="\n".join(json.dumps(x) for x in lines))
        return FakeResp(200, {"id": "b1", "processing_status": "ended", "results_url": "x/results"})


def _pages(batcher, n):
    """n pages at once, each entering, asking and leaving like a job's page task."""
    out = {}

    def page(i):
        batcher.enter()
        try:
            out[i] = batcher.request({"tag": str(i)})
        finally:
            batcher.leave()

    threads = [threading.Thread(target=page, args=(i,)) for i in range(n)]
    for i in range(n):  # all pages enter before any asks, as gather does
        batcher.enter()
    for t in threads:
        t.start()
    for i in range(n):
        batcher.leave()
    for t in threads:
        t.join(10)
    return out


def test_pages_waiting_together_go_out_as_one_batch():
    session = FakeSession()
    b = batching.AnthropicBatcher(session=session, poll_first=0.01)
    out = _pages(b, 5)
    assert len(session.batches) == 1 and len(session.batches[0]) == 5
    assert {i: m["content"][0]["text"] for i, m in out.items()} == {
        i: f"answer {i}" for i in range(5)
    }


def test_a_failed_batch_leaves_the_requests_to_direct_calls():
    b = batching.AnthropicBatcher(session=FakeSession(fail=True), poll_first=0.01)
    assert _pages(b, 3) == {0: None, 1: None, 2: None}


def test_anthropic_call_uses_the_batch_and_falls_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    session = FakeSession(skip={0})
    b = batching.AnthropicBatcher(session=session, poll_first=0.01)
    monkeypatch.setattr(b, "request", lambda body: None)  # an errored request
    direct = []
    monkeypatch.setattr(
        llm_ocr,
        "_post_with_retry",
        lambda *a, **k: (
            direct.append(1)
            or FakeResp(200, {"content": [{"type": "text", "text": "live"}], "usage": {}})
        ),
    )
    usage = {}
    token_b, token_u = batching.BATCHER.set(b), llm_ocr.USAGE_SINK.set(usage)
    try:
        assert llm_ocr._call_anthropic(b"png", "claude-opus-5-5", "p", 5) == "live"
        assert direct == [1] and "claude-opus-5-5" in usage
        monkeypatch.setattr(
            b,
            "request",
            lambda body: {
                "content": [{"type": "text", "text": "batched"}],
                "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
            },
        )
        assert llm_ocr._call_anthropic(b"png", "claude-opus-5-5", "p", 5) == "batched"
        assert direct == [1]
    finally:
        batching.BATCHER.reset(token_b)
        llm_ocr.USAGE_SINK.reset(token_u)
    batched = usage["claude-opus-5-5" + llm_ocr.BATCH_SUFFIX]
    assert batched["input_tokens"] == 1_000_000
    # a million input tokens at half of the list price
    assert llm_ocr.estimate_cost({"claude-opus-5-5 (batch)": batched}) == pytest.approx(2.0)


def test_cancel_releases_waiting_pages():
    b = batching.AnthropicBatcher(session=FakeSession(), poll_first=0.01)
    b.enter()
    b.enter()  # a second page still at work: the batch is not whole yet
    errors = []

    def page():
        try:
            b.request({"tag": "x"})
        except batching.BatchCancelled:
            errors.append("cancelled")

    t = threading.Thread(target=page)
    t.start()
    import time

    time.sleep(0.1)
    b.cancel_all()
    t.join(5)
    assert errors == ["cancelled"]


def test_sync_jobs_batch_their_claude_calls(tmp_path, monkeypatch):
    import asyncio
    from unittest.mock import patch

    from remarkable_mcp import transcribe
    from test_transcribe import INK

    monkeypatch.setenv("REMARKABLE_TRANSCRIPT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("REMARKABLE_OCR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    seen = []

    def ocr(png, backend, *a):
        seen.append(batching.BATCHER.get() is not None)
        return "text"

    async def go():
        job = transcribe.start_job(
            document="N",
            path="/N",
            mode="anthropic",
            pages=[1, 2, 3],
            total_pages=3,
            raw_doc=b"zip",
            render=lambda z, p: INK,
            workers=1,
            batch=True,
        )
        await job.task
        return job

    with patch("remarkable_mcp.llm_ocr.ocr_png", side_effect=ocr):
        job = asyncio.run(go())
    assert job.state == "done" and seen == [True, True, True]
