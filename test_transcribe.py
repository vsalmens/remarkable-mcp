"""Tests for background transcription jobs (dual reading + adjudication)."""

import asyncio
import json
import os
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
    assert transcribe.uncertainty(final) == {
        "alternatives": 2,
        "doubtful": 1,
        "suggestions": 0,
        "illegible": 1,
    }
    assert transcribe.alternatives(final) == [["Anna", "anne"], ["a", "b", "c"]]
    assert transcribe.uncertainty("set {x} and (note)") == {
        "alternatives": 0,
        "doubtful": 0,
        "suggestions": 0,
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


def test_shared_notation_runs_and_suggestions():
    text = "met anne→Anna (?) at [?: the old harbour] and {Bruno|∅} said (?) [?]"
    assert transcribe.uncertainty(text) == {
        "alternatives": 1,
        "doubtful": 2,
        "suggestions": 1,
        "illegible": 1,
    }
    assert transcribe.suggestions(text) == [["anne", "Anna"]]
    assert transcribe.alternatives(text) == [["Bruno", "∅"]]
    job, _ = _run(
        dict(document="D", path="/D", mode="dual", pages=[1], total_pages=1, workers=1),
        {1: INK},
        lambda png, b: "anne" if b == "anthropic" else "arne",
        adjudicate=lambda png, readings, *a: "anne→Anna (?)",
    )
    data = json.load(open(transcribe.summarize(job)["output_json"], encoding="utf-8"))
    assert data["pages"]["1"]["suggestions"] == [["anne", "Anna"]]
    assert "word→suggestion (?)" in data["notation"]


def test_document_without_handwriting_writes_no_files(tmp_path):
    job, adj = _run(
        dict(
            document="Paper.pdf",
            path="/R/Paper.pdf",
            mode="dual",
            pages=[1, 2],
            total_pages=2,
            workers=1,
        ),
        {1: BLANK, 2: None},
        lambda png, b: "should not be called",
    )
    s = transcribe.summarize(job)
    assert s["state"] == "done" and s["no_handwriting"] is True
    assert "output_markdown" not in s and adj.call_count == 0
    out = tmp_path / "out"
    assert not out.exists() or not any(out.iterdir())


def test_usage_reaches_the_job_through_worker_threads():
    from remarkable_mcp import llm_ocr

    def ocr(png, backend):
        llm_ocr._record_usage("claude-opus-5-5" if backend == "anthropic" else "g", 1000, 100)
        return "same"

    job, _ = _run(
        dict(document="D", path="/D", mode="dual", pages=[1, 2], total_pages=2, workers=2),
        {1: INK, 2: INK},
        ocr,
    )
    assert job.usage["claude-opus-5-5"] == {"calls": 2, "input_tokens": 2000, "output_tokens": 200}
    assert job.usage["g"]["calls"] == 2
    s = transcribe.summarize(job)
    assert s["estimated_cost_usd"] == pytest.approx((2000 * 4 + 200 * 20) / 1e6)
    data = json.load(open(s["output_json"], encoding="utf-8"))
    assert data["usage"]["g"]["input_tokens"] == 2000
    assert transcribe.brief(job)["estimated_cost_usd"] == s["estimated_cost_usd"]


def test_price_override(monkeypatch):
    from remarkable_mcp import llm_ocr

    usage = {"gemini-4-argon": {"calls": 1, "input_tokens": 1_000_000, "output_tokens": 0}}
    assert llm_ocr.estimate_cost(usage) is None
    monkeypatch.setenv("REMARKABLE_PRICES", '{"gemini-4-argon": [1.5, 9]}')
    assert llm_ocr.estimate_cost(usage) == 1.5


def test_page_ids_and_document_id_in_json(tmp_path):
    job, _ = _run(
        dict(
            document="D",
            path="/D",
            mode="gemini",
            pages=[1, 2],
            total_pages=2,
            workers=1,
            document_id="doc-9",
            modified="2026-10-14T10:00:00",
            page_ids=["p-a", "p-b"],
        ),
        {1: INK, 2: INK},
        lambda png, b: "text",
    )
    data = json.load(open(transcribe.summarize(job)["output_json"], encoding="utf-8"))
    assert data["document_id"] == "doc-9" and data["modified"] == "2026-10-14T10:00:00"
    assert [data["pages"][k]["page_id"] for k in ("1", "2")] == ["p-a", "p-b"]


def test_page_ids_from_zip(tmp_path):
    import zipfile

    from remarkable_mcp.extract import get_page_ids_from_zip

    z = tmp_path / "d.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(
            "x.content", json.dumps({"cPages": {"pages": [{"id": "a"}, {"idx": 1}, {"id": "c"}]}})
        )
    assert get_page_ids_from_zip(z) == ["a", None, "c"]
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("x.content", json.dumps({"pages": ["p1", "p2"]}))
    assert get_page_ids_from_zip(z) == ["p1", "p2"]


def test_index_records_only_main_full_runs():
    base = dict(document="D", path="/D", mode="dual", total_pages=1, workers=1)
    _run(dict(base, pages=[1], document_id="a", modified="m1"), {1: INK}, lambda png, b: "x")
    _run(
        dict(base, pages=[1], document_id="b", modified="m1", models={"judge": "gemini-x"}),
        {1: INK},
        lambda png, b: "x",
    )
    _run(
        dict(base, pages=[1], total_pages=2, document_id="c", modified="m1"),
        {1: INK},
        lambda png, b: "x",
    )
    _run(dict(base, pages=[1], document_id="e", modified="m1"), {1: BLANK}, lambda png, b: "x")
    docs = transcribe.load_index()["documents"]
    assert set(docs) == {"a", "e"}
    assert docs["a"]["modified"] == "m1" and docs["a"]["status"] == "on_device"
    assert docs["e"]["no_handwriting"] is True and docs["e"]["output_markdown"] is None


def test_plan_sync_and_mark_deleted():
    index = {
        "documents": {
            "same": {"modified": "1", "status": "on_device", "path": "/S"},
            "edited": {"modified": "1", "status": "on_device", "path": "/E"},
            "gone": {"modified": "1", "status": "on_device", "path": "/G"},
            "back": {"modified": "1", "status": "deleted", "path": "/B"},
        }
    }
    docs = [
        {"id": "same", "path": "/S", "modified": "1"},
        {"id": "edited", "path": "/E", "modified": "2"},
        {"id": "back", "path": "/B", "modified": "1"},
        {"id": "fresh", "path": "/F", "modified": "1"},
    ]
    plan = transcribe.plan_sync(docs, index)
    assert [d["id"] for d in plan["new"]] == ["fresh"]
    assert [d["id"] for d in plan["changed"]] == ["edited", "back"]
    assert plan["unchanged"] == 1 and plan["deleted"] == [{"id": "gone", "path": "/G"}]
    transcribe.save_index(index)
    transcribe.mark_deleted(["gone"])
    assert transcribe.load_index()["documents"]["gone"]["status"] == "deleted"


@pytest.mark.asyncio
@patch("remarkable_mcp.tools.get_rmapi")
async def test_sync_tool_transcribes_only_changes(mock_get_rmapi, monkeypatch):
    from remarkable_mcp import notebooks
    from test_server import _call_tool, _make_notebook_zip

    monkeypatch.setenv("REMARKABLE_EXCLUDE_PATHS", "/Private")
    folder = Mock(
        ID="f", VissibleName="Private", Parent="", is_folder=True, is_cloud_archived=False
    )
    secret = Mock(
        ID="s",
        VissibleName="Ledger",
        Parent="f",
        is_folder=False,
        is_cloud_archived=False,
        ModifiedClient="2026-10-01T00:00:00",
    )
    doc = Mock(
        ID="doc-1",
        VissibleName="Notes",
        Parent="",
        is_folder=False,
        is_cloud_archived=False,
        ModifiedClient="2026-10-13T10:00:00",
    )
    binned = Mock(ID="b", VissibleName="Old", Parent="trash", is_folder=True)
    binned.is_cloud_archived = None  # cloud items carry no flag; the parent says "trash"
    in_bin = Mock(
        ID="ib",
        VissibleName="Draft",
        Parent="b",
        is_folder=False,
        is_cloud_archived=None,
        ModifiedClient="2026-10-01T00:00:00",
    )
    client = Mock(get_meta_items=Mock(return_value=[folder, secret, doc, binned, in_bin]))
    client.download.return_value = _make_notebook_zip(notebooks.page_rm_bytes("Hello ink"))
    mock_get_rmapi.return_value = client

    async def call(args):
        res = await _call_tool("remarkable_transcribe_sync", args)
        return json.loads(res.content[0].text)

    with (
        patch("remarkable_mcp.llm_ocr.ocr_png", side_effect=lambda png, b, *a: f"{b} text"),
        patch("remarkable_mcp.llm_ocr.adjudicate_png", return_value="final text"),
    ):
        plan = await call({"dry_run": True})
        assert plan["new"] == ["/Notes"] and plan["unchanged"] == 0
        started = await call({})
        run = transcribe.get_sync(started["sync_id"])
        await run.task
        assert client.download.call_count == 1  # the excluded document is never downloaded
        status = await _call_tool("remarkable_transcribe_status", {"job_id": run.id})
        s = json.loads(status.content[0].text)
        assert s["state"] == "done" and s["documents_finished"] == 1 and s["pages_transcribed"] >= 1

        again = await call({"dry_run": True})
        assert again["new"] == [] and again["changed"] == [] and again["unchanged"] == 1

        doc.ModifiedClient = "2026-10-14T09:00:00"
        assert (await call({"dry_run": True}))["changed"] == ["/Notes"]

        client.get_meta_items.return_value = [folder, secret, binned, in_bin]
        gone = await call({})
        assert gone["deleted_from_tablet"] == ["/Notes"]
    entry = transcribe.load_index()["documents"]["doc-1"]
    assert entry["status"] == "deleted" and entry["output_markdown"]
    assert "s" not in transcribe.load_index()["documents"]
    assert "ib" not in transcribe.load_index()["documents"]  # inside a deleted folder


def _annotated_pdf_zip(tmp_path, scribble=True):
    import io
    import zipfile

    from remarkable_mcp import notebooks, strokes

    doc = pymupdf.open()
    page = doc.new_page(width=445, height=594)
    page.insert_text((50, 100), "Self efficacy predicts adherence strongly", fontsize=14)
    page.insert_text((50, 200), "An unmarked line of ordinary text", fontsize=14)
    pdf = doc.tobytes()
    rm = notebooks.page_rm_bytes("")
    # underline below "efficacy predicts"
    rm = strokes.append_strokes(rm, [{"points": [[82 / 445, 104 / 594], [205 / 445, 104 / 594]]}])
    if scribble:
        zigzag = [[0.1, 0.6], [0.2, 0.65], [0.3, 0.6], [0.4, 0.68], [0.5, 0.6]]
        rm = strokes.append_strokes(rm, [{"points": zigzag}])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        content = {"cPages": {"pages": [{"id": "page-1", "redir": {"value": 0}}]}}
        z.writestr("doc.content", json.dumps(content))
        z.writestr("page-1.rm", rm)
        z.writestr("doc.pdf", pdf)
    path = tmp_path / "doc.zip"
    path.write_bytes(buf.getvalue())
    return path


def test_find_marked_words_groups_phrases():
    from PIL import Image, ImageDraw

    from remarkable_mcp.extract import find_marked_words

    mask = Image.new("L", (400, 100), 0)
    ImageDraw.Draw(mask).line((10, 32, 150, 32), fill=255, width=2)  # under words 0-1
    words = [
        (5, 10, 60, 30, "first"),
        (70, 10, 140, 30, "second"),
        (150, 10, 220, 30, "third"),
        (5, 60, 60, 80, "next"),
    ]
    marked = find_marked_words(mask, words, 1.0)
    assert marked == [
        {"text": "**first second** third next", "marked": "first second", "kind": "underline"}
    ]


def test_strike_through_marks_only_its_own_line():
    from PIL import Image, ImageDraw

    from remarkable_mcp.extract import find_marked_words

    mask = Image.new("L", (200, 60), 0)
    draw = ImageDraw.Draw(mask)
    draw.line((5, 15, 120, 15), fill=255, width=2)  # through the middle of line 2
    draw.line((5, 33, 120, 33), fill=255, width=2)  # just under line 3
    words = [
        (5, 0, 60, 10, "above"),
        (65, 0, 120, 10, "it"),
        (5, 11, 60, 21, "struck"),
        (65, 11, 120, 21, "words"),
        (5, 22, 60, 32, "underlined"),
        (65, 22, 120, 32, "too"),
        (5, 44, 60, 54, "below"),
    ]
    marked = find_marked_words(mask, words, 1.0)
    assert [m["marked"] for m in marked] == ["struck words underlined too"]
    assert marked[0]["kind"] in ("strike", "underline")


def test_marked_phrases_rejoin_hyphenated_words_and_ligatures():
    from PIL import Image, ImageDraw

    from remarkable_mcp.extract import find_marked_words

    mask = Image.new("L", (200, 100), 0)
    draw = ImageDraw.Draw(mask)
    draw.line((5, 32, 60, 32), fill=255, width=2)
    draw.line((5, 82, 60, 82), fill=255, width=2)
    words = [(5, 10, 60, 30, "promi-"), (5, 60, 60, 80, "nent ﬁndings")]
    assert [m["marked"] for m in find_marked_words(mask, words, 1.0)] == ["prominent findings"]


def test_pdf_marks_come_from_the_text_layer(tmp_path):
    from remarkable_mcp.extract import analyze_pdf_page_annotations, document_has_pdf

    path = _annotated_pdf_zip(tmp_path)
    assert document_has_pdf(path)
    result = analyze_pdf_page_annotations(path, 1)
    assert [(m["marked"], m["kind"]) for m in result["marked"]] == [
        ("efficacy predicts", "underline")
    ]
    assert "**efficacy predicts**" in result["marked"][0]["text"]
    assert result["text_layer"]
    assert transcribe.has_ink(result["residual_png"])  # the zigzag, not the underline
    clean = analyze_pdf_page_annotations(_annotated_pdf_zip(tmp_path, scribble=False), 1)
    assert [m["marked"] for m in clean["marked"]] == ["efficacy predicts"]
    assert clean["residual_png"] is None


def test_pdf_job_reads_only_handwriting(tmp_path):
    from remarkable_mcp.extract import analyze_pdf_page_annotations

    path = _annotated_pdf_zip(tmp_path, scribble=False)
    calls = []

    async def go():
        job = transcribe.start_job(
            document="Paper.pdf",
            path="/R/Paper.pdf",
            mode="dual",
            pages=[1],
            total_pages=1,
            raw_doc=path.read_bytes(),
            render=lambda z, p: INK,
            workers=1,
            annotate=analyze_pdf_page_annotations,
        )
        await job.task
        return job

    with patch("remarkable_mcp.llm_ocr.ocr_png", side_effect=lambda *a: calls.append(a) or "x"):
        job = asyncio.run(go())
    assert calls == []  # an underline alone needs no OCR
    r = job.results[1]
    assert (
        r.status == "done"
        and [m["marked"] for m in r.marked] == ["efficacy predicts"]
        and r.final == ""
    )
    s = transcribe.summarize(job)
    assert s["pages_with_marks"] == [1]
    md = open(s["output_markdown"], encoding="utf-8").read()
    assert "Marked in the text" in md and "**efficacy predicts**" in md
    data = json.load(open(s["output_json"], encoding="utf-8"))
    assert [m["marked"] for m in data["pages"]["1"]["marked"]] == ["efficacy predicts"]


def test_drawing_descriptions_are_not_handwriting():
    assert not transcribe._has_handwriting("[two horizontal lines; oval on the right]\n?\n?")
    assert not transcribe._has_handwriting("! ~ —")
    assert transcribe._has_handwriting("[arrow] see p. 4")
    assert transcribe._has_handwriting("[?: word word]")


@pytest.mark.parametrize("reading", ["! ~ —", "[horizontal pen strokes, no legible writing]"])
def test_pdf_symbols_only_are_not_handwriting(tmp_path, reading):
    from remarkable_mcp.extract import analyze_pdf_page_annotations

    path = _annotated_pdf_zip(tmp_path)

    async def go():
        job = transcribe.start_job(
            document="Paper.pdf",
            path="/R/Paper.pdf",
            mode="dual",
            pages=[1],
            total_pages=1,
            raw_doc=path.read_bytes(),
            render=lambda z, p: INK,
            workers=1,
            annotate=analyze_pdf_page_annotations,
        )
        await job.task
        return job

    with (
        patch("remarkable_mcp.llm_ocr.ocr_png", side_effect=lambda png, b, *a: reading),
        patch("remarkable_mcp.llm_ocr.adjudicate_png") as adj,
    ):
        job = asyncio.run(go())
    assert job.results[1].final == "" and [m["marked"] for m in job.results[1].marked] == [
        "efficacy predicts"
    ]
    adj.assert_not_called()


def test_cancel_job_keeps_finished_pages():
    import threading

    gate = threading.Event()

    def ocr(png, backend):
        if not gate.is_set():
            gate.set()
            return "first page"
        threading.Event().wait(5)  # the second page hangs until cancelled
        return "late"

    async def go():
        job = transcribe.start_job(
            document="D",
            path="/D",
            mode="gemini",
            pages=[1, 2],
            total_pages=2,
            raw_doc=b"zip",
            render=lambda z, p: INK,
            workers=1,
        )
        while job.results[1].status != "done":
            await asyncio.sleep(0.01)
        transcribe.cancel_job(job.id)
        await job.task
        return job

    with patch("remarkable_mcp.llm_ocr.ocr_png", side_effect=lambda png, b, *a: ocr(png, b)):
        job = asyncio.run(go())
    assert job.state == "cancelled"
    assert job.results[1].final == "first page" and job.results[2].status != "done"
    assert open(transcribe.summarize(job)["output_markdown"], encoding="utf-8").read()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_outputs_replace_files_that_cannot_be_opened_for_writing(tmp_path):
    target = tmp_path / "old.md"
    target.write_text("old")
    target.chmod(0o444)  # stands in for a file macOS will not open for writing
    transcribe._write_atomic(target, "new")
    assert target.read_text() == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["old.md"]


def test_pen_marks_are_told_apart_from_handwriting_over_the_print():
    from PIL import Image, ImageDraw, ImageFont

    from remarkable_mcp.extract import split_pen_marks

    # scale 1: three lines of 20 px high words
    words = [
        (10, 10, 90, 30, "struck", 0, 0, 0),
        (100, 10, 190, 30, "phrase", 0, 0, 1),
        (10, 60, 90, 80, "plain", 1, 0, 0),
        (100, 60, 190, 80, "circled", 1, 0, 1),
        (10, 110, 90, 130, "noted", 2, 0, 0),
        (100, 110, 190, 130, "glowing", 2, 0, 1),
    ]
    mask = Image.new("L", (400, 160), 0)
    draw = ImageDraw.Draw(mask)
    draw.line((10, 20, 190, 22), fill=255, width=3)  # strike-through, slightly sloped
    draw.ellipse((92, 52, 198, 88), outline=255, width=2)  # ring around "circled"
    draw.rectangle((100, 112, 190, 128), fill=255)  # highlighter band over "glowing"
    font = ImageFont.load_default(size=28)
    draw.text((20, 98), "Orch OR", fill=255, font=font)  # note written over "noted"
    phrases, residual = split_pen_marks(mask, words, 1.0)
    phrases = [{k: v for k, v in p.items() if not k.startswith("_")} for p in phrases]
    assert phrases == [
        {"text": "**struck phrase**", "marked": "struck phrase", "kind": "strike"},
        {"text": "plain **circled**", "marked": "circled", "kind": "circle"},
        {"text": "noted **glowing**", "marked": "glowing", "kind": "highlight"},
    ]
    left = residual.getbbox()
    assert left is not None and left[1] >= 95 and left[3] <= 135  # only the note is left


@pytest.mark.asyncio
@patch("remarkable_mcp.tools.get_rmapi")
async def test_auto_sync_waits_until_writing_has_stopped(mock_get_rmapi, monkeypatch):
    from datetime import datetime, timedelta, timezone

    from remarkable_mcp import notebooks, tools
    from test_server import _make_notebook_zip

    def stamp(minutes_ago):
        return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()

    fresh = Mock(
        ID="fresh", VissibleName="Now", Parent="", is_folder=False, is_cloud_archived=False
    )
    fresh.ModifiedClient = stamp(2)
    settled = Mock(
        ID="settled", VissibleName="Earlier", Parent="", is_folder=False, is_cloud_archived=False
    )
    settled.ModifiedClient = stamp(30)
    client = Mock(get_meta_items=Mock(return_value=[fresh, settled]))
    client.download.return_value = _make_notebook_zip(notebooks.page_rm_bytes("Hello ink"))
    mock_get_rmapi.return_value = client

    with (
        patch("remarkable_mcp.llm_ocr.ocr_png", side_effect=lambda png, b, *a: f"{b} text"),
        patch("remarkable_mcp.llm_ocr.adjudicate_png", return_value="final text"),
    ):
        refused, run = await tools._sync_library(settle_seconds=600, max_documents=0)
        assert run is None and "by hand" in json.loads(refused)["_hint"]
        response, run = await tools._sync_library(settle_seconds=600, max_documents=5)
        await run.task
    data = json.loads(response)
    assert data["new"] == ["/Earlier"] and data["still_being_written"] == ["/Now"]
    assert set(transcribe.load_index()["documents"]) == {"settled"}


def test_auto_sync_is_off_unless_configured(monkeypatch):
    from remarkable_mcp import tools

    monkeypatch.delenv("REMARKABLE_AUTO_SYNC_MINUTES", raising=False)
    assert tools.start_auto_sync() is None


def test_auto_sync_times_of_day():
    from datetime import datetime

    from remarkable_mcp.tools import _parse_times, _seconds_until_next

    times = _parse_times("23, 7, 16:00, nonsense, 25")
    assert times == [(7, 0), (16, 0), (23, 0)]
    assert _seconds_until_next(times, datetime(2026, 10, 14, 16, 28)) == (6 * 60 + 32) * 60
    assert _seconds_until_next(times, datetime(2026, 10, 14, 23, 0, 1)) == 8 * 3600 - 1


def test_one_stroke_ring_through_the_text_marks_the_word():
    from PIL import Image, ImageDraw

    from remarkable_mcp.extract import split_pen_marks

    words = [
        (10, 60, 90, 80, "Homework", 0, 0, 0),
        (100, 60, 190, 80, "Demonstrably", 1, 0, 0),
        (200, 60, 260, 80, "effective", 1, 0, 1),
    ]
    mask = Image.new("L", (300, 140), 0)
    # One stroke: through the words from the left, round the right end, back
    # over the top, down the left side and under them (as drawn on a table)
    path = [
        (98, 70),
        (262, 70),
        (272, 60),
        (262, 52),
        (98, 52),
        (90, 62),
        (92, 80),
        (100, 88),
        (262, 88),
        (270, 80),
    ]
    ImageDraw.Draw(mask).line(path, fill=255, width=2, joint="curve")
    phrases, residual = split_pen_marks(mask, words, 1.0)
    assert [(p["marked"], p["kind"]) for p in phrases] == [("Demonstrably effective", "circle")]
    assert residual.getbbox() is None


def test_sentence_context_fills_small_gaps_and_knows_abbreviations():
    from PIL import Image, ImageDraw

    from remarkable_mcp.extract import find_marked_words

    text = "As noted e.g. by Smith the effect was large and stable. Next sentence here."
    words = [(10 + i * 60, 10, 60 + i * 60, 30, w, 0, 0, i) for i, w in enumerate(text.split())]
    mask = Image.new("L", (60 * len(words) + 20, 40), 0)
    draw = ImageDraw.Draw(mask)
    for i in (5, 6, 8, 9):  # "the effect" … "large and"; "was" between is a gap of one
        draw.line((10 + i * 60, 20, 60 + i * 60, 20), fill=255, width=2)
    marked = find_marked_words(mask, words, 1.0)
    assert marked == [
        {
            "text": "As noted e.g. by Smith **the effect was large and** stable.",
            "marked": "the effect was large and",
            "kind": "strike",
        }
    ]


def test_soft_hyphen_and_stray_short_words():
    from PIL import Image, ImageDraw

    from remarkable_mcp.extract import find_marked_words

    words = [
        (10, 10, 60, 30, "conscious", 0, 0, 0),
        (70, 10, 120, 30, "con­", 0, 0, 1),
        (10, 40, 60, 60, "tents.", 0, 1, 0),
        (10, 70, 60, 90, "They", 1, 0, 0),
        (70, 70, 120, 90, "can", 1, 0, 1),
        (130, 70, 180, 90, "go.", 1, 0, 2),
    ]
    mask = Image.new("L", (200, 100), 0)
    draw = ImageDraw.Draw(mask)
    draw.line((70, 31, 120, 31), fill=255, width=2)
    draw.line((10, 61, 60, 61), fill=255, width=2)
    draw.line((70, 91, 120, 91), fill=255, width=2)  # only "can": a stray touch
    assert find_marked_words(mask, words, 1.0) == [
        {"text": "conscious **contents.**", "marked": "contents.", "kind": "underline"}
    ]


def test_composed_annotations_keep_only_text_that_is_on_the_page():
    from remarkable_mcp.extract import checked_annotations

    blocks = [
        "Because of some of these reactions, PIs are often considered potentially "
        "unethical because they can be deceptive or manipulative.",
        "Table 1. Summary of Associations",
        "Skills or methods Distal end-of-treatment outcomes Questions Neutral/mixed",
    ]
    data = {
        "items": [
            {
                "type": "passage",
                "quote": "PIs are **often considered potentially unethical** because they can be "
                "deceptive or manipulative.",
                "note": "manipulaatiota?",
                "symbols": ["?"],
            },
            {"type": "passage", "quote": "PIs are **always harmful.**"},  # not on the page
            {"type": "passage", "quote": "Because of some of these reactions"},  # no mark
            {
                "type": "table",
                "caption": "Table 1. Summary of Associations",
                "headers": ["Skills or methods"],
                "rows": ["…", ["Questions ✎?", "**Neutral/mixed**"], "…"],
            },
        ],
        "unplaced_notes": ["orch OR"],
    }
    kept = checked_annotations(data, blocks)
    assert [i["type"] for i in kept["items"]] == ["passage", "table"]
    assert kept["unplaced_notes"] == ["orch OR"]
    assert checked_annotations({"items": [], "unplaced_notes": []}, blocks) is None


def test_annotation_markdown_sets_the_readers_own_marks_apart(monkeypatch):
    monkeypatch.setenv("REMARKABLE_TRANSCRIPT_LANG", "en")
    ann = {
        "items": [
            {
                "type": "passage",
                "quote": "States of consciousness can be grouped into two classes: "
                "**global states and local states.**",
                "label": "I",
                "symbols": ["!"],
                "note": "tietoisuus kokemuksena",
            },
            {
                "type": "table",
                "caption": "Table 1. Summary",
                "headers": ["Skills or methods", "Distal"],
                "rows": ["…", ["**Paradoxical interventions**", "**Demonstrably effective**"], "…"],
            },
        ],
        "unplaced_notes": ["Ha ff"],
    }
    md = "\n".join(transcribe.annotation_lines(ann))
    assert (
        "- **(I)** States of consciousness can be grouped into two classes: "
        "**global states and local states.** ✎ **!**" in md
    )
    assert "  - ✍ tietoisuus kokemuksena" in md
    assert "| Skills or methods | Distal |" in md and "| … |  |" in md
    assert "| **Paradoxical interventions** | **Demonstrably effective** |" in md
    assert "- ✍ Ha ff" in md


def test_pdf_job_composes_annotations(tmp_path, monkeypatch):
    from remarkable_mcp.extract import analyze_pdf_page_annotations

    monkeypatch.setenv("REMARKABLE_TRANSCRIPT_LANG", "en")
    path = _annotated_pdf_zip(tmp_path)
    seen = {}

    def compose(png, blocks, marks, handwriting, *a):
        seen["blocks"], seen["marks"], seen["handwriting"] = blocks, marks, handwriting
        return {
            "items": [
                {
                    "type": "passage",
                    "quote": "Self **efficacy predicts** adherence strongly",
                    "note": "see ch. 2",
                }
            ],
            "unplaced_notes": [],
        }

    async def go():
        job = transcribe.start_job(
            document="Paper.pdf",
            path="/R/Paper.pdf",
            mode="dual",
            pages=[1],
            total_pages=1,
            raw_doc=path.read_bytes(),
            render=lambda z, p: INK,
            workers=1,
            annotate=analyze_pdf_page_annotations,
        )
        await job.task
        return job

    with (
        patch("remarkable_mcp.llm_ocr.ocr_png", side_effect=lambda png, b, *a: "see ch. 2"),
        patch("remarkable_mcp.llm_ocr.compose_annotations", side_effect=compose),
    ):
        job = asyncio.run(go())
    r = job.results[1]
    assert r.annotations["items"][0]["note"] == "see ch. 2"
    assert any("efficacy predicts" in b for b in seen["blocks"])
    assert seen["handwriting"] == "see ch. 2"
    md = open(transcribe.summarize(job)["output_markdown"], encoding="utf-8").read()
    assert "- Self **efficacy predicts** adherence strongly" in md and "  - ✍ see ch. 2" in md


def test_marks_name_the_pen_that_drew_them(tmp_path, monkeypatch):
    from remarkable_mcp.extract import analyze_pdf_page_annotations

    result = analyze_pdf_page_annotations(_annotated_pdf_zip(tmp_path, scribble=False), 1)
    pen = result["marked"][0]["pen"]
    assert pen["tool"] == "fineliner" and pen["color"] == "black" and 0.1 < pen["width_mm"] < 1.5
    monkeypatch.setenv("REMARKABLE_TRANSCRIPT_LANG", "fi")
    assert transcribe.pen_text({"tool": "calligraphy pen", "color": "black", "width_mm": 0.26}) == (
        " _(kalligrafiakynä, musta, 0,26 mm)_"
    )
    monkeypatch.setenv("REMARKABLE_TRANSCRIPT_LANG", "en")
    assert transcribe.pen_text("ballpoint, red, 0.5 mm") == " _(ballpoint, red, 0.5 mm)_"
    assert transcribe.pen_text(None) == ""


def test_calligraphy_lines_are_drawn_thin():
    from types import SimpleNamespace

    from remarkable_mcp.extract import _v6_stroke_width

    points = [SimpleNamespace(width=8, pressure=100)] * 5  # as stored by a Paper Pro
    assert _v6_stroke_width(21, points, 1.16) < 4  # stroke units; the old rule gave 5
    assert 5 < _v6_stroke_width(17, points, 2.0) < 6.5  # fineliner, thickness setting 2


def test_composing_runs_only_where_geometry_cannot_place_the_marks():
    r = transcribe.PageResult(page=1, status="done")
    r.marked = [{"text": "**a b**", "marked": "a b", "kind": "underline"}]
    assert not transcribe.needs_composing(r)  # lines in running text: already whole sentences
    r.final = "?"
    assert transcribe.needs_composing(r)  # a drawn symbol or a note must be placed
    r.final = ""
    r.marked = [{"text": "x", "marked": "x", "kind": "circle"}]
    assert transcribe.needs_composing(r)
    r.marked = [{"text": "x", "marked": "x", "kind": "underline", "table": True}]
    assert transcribe.needs_composing(r)


def test_table_blocks_are_recognised():
    from remarkable_mcp.extract import _table_blocks

    row = lambda y, line: [  # noqa: E731
        (10, y, 60, y + 10, "Method", 3, line, 0),
        (200, y, 260, y + 10, "Probably", 3, line, 1),
        (400, y, 460, y + 10, "Neutral", 3, line, 2),
    ]
    words = row(10, 0) + row(30, 1) + row(50, 2)
    words += [(10, 100, 50, 110, "Plain", 4, 0, 0), (55, 100, 90, 110, "prose.", 4, 0, 1)]
    assert _table_blocks(words) == [(10, 10, 460, 60)]


def test_pen_is_named_once_and_again_only_when_it_changes(monkeypatch):
    monkeypatch.setenv("REMARKABLE_TRANSCRIPT_LANG", "fi")
    cal = {"tool": "calligraphy pen", "color": "black", "width_mm": 0.24}
    ann = {
        "items": [
            {"type": "passage", "quote": "one", "pen": cal},
            {"type": "passage", "quote": "two", "pen": dict(cal, width_mm=0.27)},
            {"type": "passage", "quote": "three", "pen": dict(cal, color="red")},
            {"type": "passage", "quote": "four", "pen": dict(cal, color="red", width_mm=0.6)},
        ]
    }
    pens = transcribe.PenTracker()
    md = "\n".join(transcribe.annotation_lines(ann, pens=pens))
    assert md.count("✒") == 3
    assert "_✒ Kynä: kalligrafiakynä, musta, 0,24 mm_" in md
    assert "(kalligrafiakynä" not in md  # no pen after each passage
    # the same pen on the next page is not named again
    md2 = "\n".join(transcribe.annotation_lines({"items": ann["items"][3:]}, pens=pens))
    assert "✒" not in md2
    marks = [{"text": "a **b**", "pen": cal}, {"text": "c **d**", "pen": cal}]
    lines = transcribe.marked_lines(marks)
    assert sum("✒" in line for line in lines) == 1 and lines[-1] == "- c **d**"
