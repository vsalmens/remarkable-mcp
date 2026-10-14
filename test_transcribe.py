"""Tests for background transcription jobs (dual reading + adjudication)."""

import asyncio
import json
from unittest.mock import Mock, patch

import pymupdf
import pytest

from remarkable_mcp import transcribe


def _png(ink: bool) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    pix.clear_with(255)
    if ink:
        for x in range(10, 50):
            for y in range(28, 32):
                pix.set_pixel(x, y, (0, 0, 0))
    return pix.tobytes("png")


INK, BLANK = _png(True), _png(False)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("REMARKABLE_TRANSCRIPT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("REMARKABLE_OCR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("REMARKABLE_OCR_VOCABULARY_FILE", str(tmp_path / "no-vocab.txt"))
    monkeypatch.delenv("REMARKABLE_OCR_VOCABULARY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("GEMINI_API_KEY", "g")


def test_parse_pages():
    assert transcribe.parse_pages("all", 5) == [1, 2, 3, 4, 5]
    assert transcribe.parse_pages("1-2,4,9", 5) == [1, 2, 4]
    assert transcribe.parse_pages("4-", 6) == [4, 5, 6]
    assert transcribe.parse_pages("-2", 6) == [1, 2]
    with pytest.raises(ValueError):
        transcribe.parse_pages("x", 3)


def test_has_ink():
    assert transcribe.has_ink(INK)
    assert not transcribe.has_ink(BLANK)


def test_agreement():
    assert transcribe.agreement("a b c", "a b c") == 1.0
    assert transcribe.agreement("a b c", "a x c") == pytest.approx(2 / 3)
    assert transcribe.agreement(None, "a") is None


def _run(job_kwargs, renders, ocr, adjudicate=None):
    async def go():
        job = transcribe.start_job(raw_doc=b"zip", **job_kwargs, render=lambda z, p: renders[p])
        await job.task
        return job

    with (
        patch(
            "remarkable_mcp.llm_ocr.ocr_png",
            side_effect=lambda png, b, *a: ocr(png, b),
        ) as ocr_mock,
        patch("remarkable_mcp.llm_ocr.adjudicate_png", side_effect=adjudicate) as adj,
    ):
        job = asyncio.run(go())
    _run.last_ocr_calls = ocr_mock.call_args_list
    return job, adj


def test_dual_job_end_to_end(tmp_path):
    renders = {1: INK, 2: BLANK, 3: None, 4: INK, 5: INK}

    def ocr(png, backend):
        # page identity is not visible here; vary by call order instead
        return {"anthropic": "surprisingly it worked", "gemini": "surprisingly it works"}[backend]

    job, adj = _run(
        dict(
            document="Sketchbook",
            path="/Sketchbook",
            mode="dual",
            pages=[1, 2, 3, 4, 5],
            total_pages=5,
            workers=2,
        ),
        renders,
        ocr,
        adjudicate=lambda png, readings, *a: "surprisingly it worked",
    )
    s = transcribe.summarize(job)
    assert s["state"] == "done"
    assert s["done"] == 3 and s["blank"] == 2 and s["pending"] == 0
    assert adj.call_count == 3
    assert job.results[1].final == "surprisingly it worked"
    assert 0 < job.results[1].agreement < 1
    md = open(s["output_markdown"], encoding="utf-8").read()
    assert "## Page 1" in md and "## Page 2" not in md and "Claude" in md and "Gemini" in md
    assert "·" not in md
    data = json.load(open(s["output_json"], encoding="utf-8"))
    assert data["pages"]["2"]["status"] == "blank"


def test_identical_readings_skip_adjudication():
    job, adj = _run(
        dict(document="D", path="/D", mode="dual", pages=[1], total_pages=1, workers=1),
        {1: INK},
        lambda png, b: "same text",
    )
    assert job.results[1].final == "same text"
    assert job.results[1].agreement == 1.0
    adj.assert_not_called()


def test_one_reader_fails_uses_other():
    job, adj = _run(
        dict(document="D", path="/D", mode="dual", pages=[1], total_pages=1, workers=1),
        {1: INK},
        lambda png, b: None if b == "gemini" else "vain claude",
    )
    assert job.results[1].status == "done"
    assert job.results[1].final == "vain claude"
    adj.assert_not_called()


def test_all_readers_fail_marks_page_failed():
    job, _ = _run(
        dict(document="D", path="/D", mode="anthropic", pages=[1], total_pages=1, workers=1),
        {1: INK},
        lambda png, b: None,
    )
    s = transcribe.summarize(job)
    assert s["failed"] == 1 and s["failed_pages"] == [1]


def test_model_reports_blank():
    job, _ = _run(
        dict(document="D", path="/D", mode="dual", pages=[1], total_pages=1, workers=1),
        {1: INK},
        lambda png, b: "",
    )
    assert job.results[1].status == "blank"


@pytest.mark.asyncio
async def test_status_tool_text_and_filters(monkeypatch):
    from test_server import _call_tool

    job = transcribe.Job(
        id="abc12345", document="D", path="/D", mode="dual", pages=[1, 2], total_pages=2
    )
    job.results = {
        1: transcribe.PageResult(1, "done", {"anthropic": "a", "gemini": "a"}, "a", 1.0),
        2: transcribe.PageResult(2, "done", {"anthropic": "x y", "gemini": "x z"}, "x y", 0.5),
    }
    job.state = "done"
    transcribe._JOBS[job.id] = job
    res = await _call_tool(
        "remarkable_transcribe_status",
        {"job_id": "abc12345", "include_text": True, "disagreements_only": True},
    )
    data = json.loads(res.content[0].text)
    assert [p["page"] for p in data["pages"]] == [2]
    assert data["pages_below_80pct_agreement"] == [2]


@pytest.mark.asyncio
async def test_transcribe_tool_requires_keys(monkeypatch):
    from test_server import _call_tool

    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    res = await _call_tool("remarkable_transcribe", {"document": "X", "mode": "dual"})
    assert "missing_credentials" in res.content[0].text


@pytest.mark.asyncio
@patch("remarkable_mcp.tools.get_rmapi")
async def test_transcribe_tool_respects_exclusion(mock_get_rmapi, monkeypatch):
    from test_server import _call_tool

    monkeypatch.setenv("REMARKABLE_EXCLUDE_PATHS", "/Private")
    folder = Mock(
        ID="f", VissibleName="Private", Parent="", is_folder=True, is_cloud_archived=False
    )
    doc = Mock(ID="d", VissibleName="Ledger", Parent="f", is_folder=False, is_cloud_archived=False)
    mock_get_rmapi.return_value = Mock(get_meta_items=Mock(return_value=[folder, doc]))
    res = await _call_tool("remarkable_transcribe", {"document": "/Private/Ledger"})
    assert "document_not_found" in res.content[0].text


@pytest.mark.asyncio
@patch("remarkable_mcp.tools.get_rmapi")
async def test_transcribe_tool_real_render_end_to_end(mock_get_rmapi, tmp_path):
    from remarkable_mcp import notebooks
    from test_server import _call_tool, _make_notebook_zip

    doc = Mock(
        ID="doc-1",
        VissibleName="Notes",
        Parent="",
        is_folder=False,
        is_cloud_archived=False,
        ModifiedClient="2026-10-13T10:00:00Z",
    )
    client = Mock(get_meta_items=Mock(return_value=[doc]))
    client.download.return_value = _make_notebook_zip(notebooks.page_rm_bytes("Hello ink"))
    mock_get_rmapi.return_value = client

    with (
        patch("remarkable_mcp.llm_ocr.ocr_png", side_effect=lambda png, b, *a: f"{b} text"),
        patch("remarkable_mcp.llm_ocr.adjudicate_png", return_value="final text"),
    ):
        res = await _call_tool("remarkable_transcribe", {"document": "Notes"})
        data = json.loads(res.content[0].text)
        job = transcribe.get_job(data["job_id"])
        await job.task
    s = transcribe.summarize(job)
    assert s["state"] == "done", s
    assert s["done"] + s["blank"] == s["pages_requested"]
    assert s["done"] >= 1
    assert open(s["output_markdown"], encoding="utf-8").read().count("final text") == s["done"]


def test_uncertainty_markers_counted_and_written():
    final = "Anna wrote → {Anna|anne} notes\nvan (?) and [?]\n{a|b|c} ok"
    assert transcribe.uncertainty(final) == {"alternatives": 2, "doubtful": 1, "illegible": 1}
    assert transcribe.alternatives(final) == [["Anna", "anne"], ["a", "b", "c"]]
    assert transcribe.uncertainty("set {x} and (note)") == {
        "alternatives": 0,
        "doubtful": 0,
        "illegible": 0,
    }
    assert transcribe.certainty_level(0.95, transcribe.uncertainty("clear")) == "high"
    assert transcribe.certainty_level(0.5, transcribe.uncertainty(final)) == "low"

    job, _ = _run(
        dict(document="D", path="/D", mode="dual", pages=[1], total_pages=1, workers=1),
        {1: INK},
        lambda png, b: "anne wrote" if b == "anthropic" else "Anna wrote",
        adjudicate=lambda png, readings, *a: "{Anna|anne} wrote notes",
    )
    s = transcribe.summarize(job)
    assert s["pages_with_uncertainty_markers"] == [1]
    md = open(s["output_markdown"], encoding="utf-8").read()
    assert "Markers:" in md and "1 alternative readings" in md and "certainty " in md
    assert "·" not in md
    data = json.load(open(s["output_json"], encoding="utf-8"))
    assert data["pages"]["1"]["alternatives"] == [["Anna", "anne"]]


def test_model_override_reaches_reader_and_names_output():
    from remarkable_mcp import llm_ocr

    assert llm_ocr.backend_for_model("gemini-3.8-flash") == "gemini"
    assert llm_ocr.backend_for_model("claude-fable-5-1") == "anthropic"
    assert llm_ocr.backend_for_model("gpt-x") is None

    job, _ = _run(
        dict(
            document="D",
            path="/J/D",
            mode="gemini",
            pages=[1],
            total_pages=1,
            workers=1,
            models={"gemini": "gemini-3.8-flash"},
        ),
        {1: INK},
        lambda png, b: "text",
    )
    assert _run.last_ocr_calls[0].args[3] == "gemini-3.8-flash"
    s = transcribe.summarize(job)
    assert s["models"] == {"gemini": "gemini-3.8-flash"}
    assert s["output_markdown"].endswith("J - D [gemini-3.8-flash].md")
    md = open(s["output_markdown"], encoding="utf-8").read()
    assert "Gemini (gemini-3.8-flash)" in md


def test_model_override_changes_cache_key(tmp_path, monkeypatch):
    from remarkable_mcp import llm_ocr

    monkeypatch.setenv("REMARKABLE_OCR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls = []

    def fake(png, model, prompt, timeout):
        calls.append(model)
        return f"reading {model}"

    with patch.dict(llm_ocr._CALLERS, {"gemini": fake}):
        assert llm_ocr.ocr_png(b"png", "gemini") == f"reading {llm_ocr.DEFAULT_MODELS['gemini']}"
        assert llm_ocr.ocr_png(b"png", "gemini", model="gemini-3.8-flash") == (
            "reading gemini-3.8-flash"
        )
        llm_ocr.ocr_png(b"png", "gemini", model="gemini-3.8-flash")
    assert calls == [llm_ocr.DEFAULT_MODELS["gemini"], "gemini-3.8-flash"]


def test_finnish_transcript_labels(monkeypatch):
    monkeypatch.setenv("REMARKABLE_TRANSCRIPT_LANG", "fi")
    job, _ = _run(
        dict(document="D", path="/D", mode="dual", pages=[1], total_pages=1, workers=1),
        {1: INK},
        lambda png, b: "a" if b == "anthropic" else "b",
        adjudicate=lambda png, readings, *a: "{a|b}",
    )
    md = open(transcribe.summarize(job)["output_markdown"], encoding="utf-8").read()
    assert "## Sivu 1" in md and "Merkinnät:" in md and "1 vaihtoehtolukua" in md


def test_finished_jobs_are_pruned(monkeypatch):
    monkeypatch.setattr(transcribe, "MAX_FINISHED_JOBS", 2)
    for _ in range(4):
        _run(
            dict(document="D", path="/D", mode="gemini", pages=[1], total_pages=1, workers=1),
            {1: INK},
            lambda png, b: "text",
        )
    assert sum(1 for j in transcribe.list_jobs() if j.finished is not None) <= 3


def test_page_context_uses_nearest_read_pages():
    job = transcribe.Job(
        id="x", document="D", path="/D", mode="dual", pages=[1, 2, 3, 4], total_pages=4
    )
    job.results = {p: transcribe.PageResult(page=p) for p in job.pages}
    job.results[1].status, job.results[1].final = "done", "Yours, Anna"
    job.results[1].readings = {"anthropic": "Yours, Arna", "gemini": "Yours, Anna"}
    job.results[2].status = "blank"
    job.results[4].status, job.results[4].final = "pending", "after"
    job.results[4].readings = {"anthropic": "after"}
    assert transcribe.page_context(job, 3) == {"previous": "Yours, Anna", "next": "after"}
    assert transcribe.page_context(job, 1) == {"previous": None, "next": "after"}


def test_adjudication_gets_neighbour_context_and_judge():
    seen = []

    def adjudicate(png, readings, timeout, context, judge):
        seen.append((context, judge))
        return "final"

    texts = {1: ("Alpha", "Alpha"), 2: ("Beta x", "Beta y"), 3: ("Gamma", "Gamma")}
    calls = {"n": 0}

    def ocr(png, backend):
        page = calls["n"] // 2 + 1
        calls["n"] += 1
        return texts[page][0 if backend == "anthropic" else 1]

    job, adj = _run(
        dict(
            document="D",
            path="/D",
            mode="dual",
            pages=[1, 2, 3],
            total_pages=3,
            workers=1,
            models={"judge": "gemini-4-argon"},
        ),
        {1: INK, 2: INK, 3: INK},
        ocr,
        adjudicate=adjudicate,
    )
    assert adj.call_count == 1
    assert seen == [({"previous": "Alpha", "next": "Gamma"}, "gemini-4-argon")]
    s = transcribe.summarize(job)
    assert s["output_markdown"].endswith("D [judge gemini-4-argon].md")


def test_failed_adjudication_keeps_first_reading():
    def boom(*a):
        raise RuntimeError("judge down")

    job, _ = _run(
        dict(document="D", path="/D", mode="dual", pages=[1], total_pages=1, workers=1),
        {1: INK},
        lambda png, b: "claude text" if b == "anthropic" else "gemini text",
        adjudicate=boom,
    )
    assert job.results[1].status == "done" and job.results[1].final == "claude text"
