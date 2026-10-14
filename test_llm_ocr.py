"""Tests for the Gemini / Anthropic OCR backends and their disk cache."""

import json
from unittest.mock import Mock, patch

import pytest

from remarkable_mcp import extract, llm_ocr

PNG = b"\x89PNG\r\n\x1a\nfake-page"


def _resp(status, payload):
    r = Mock(status_code=status)
    r.json.return_value = payload
    r.text = json.dumps(payload)
    return r


GEMINI_OK = {"candidates": [{"content": {"parts": [{"text": "Hello\nworrier → warrior"}]}}]}
GEMINI_THOUGHT = {
    "candidates": [
        {"content": {"parts": [{"text": "thinking…", "thought": True}, {"text": "Hello"}]}}
    ]
}
ANTHROPIC_OK = {"content": [{"type": "text", "text": "Run! ♡"}]}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("REMARKABLE_OCR_CACHE_DIR", str(tmp_path / "ocr"))
    monkeypatch.setenv("REMARKABLE_OCR_LANGUAGES", "fi,en")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    monkeypatch.setenv("REMARKABLE_OCR_RETRIES", "1")
    monkeypatch.setenv("REMARKABLE_OCR_VOCABULARY_FILE", str(tmp_path / "no-vocab.txt"))
    monkeypatch.delenv("REMARKABLE_OCR_VOCABULARY", raising=False)
    for var in ("REMARKABLE_GEMINI_MODEL", "REMARKABLE_ANTHROPIC_MODEL", "REMARKABLE_OCR_CACHE"):
        monkeypatch.delenv(var, raising=False)


def test_prompt_names_languages():
    prompt = llm_ocr.build_prompt()
    assert "Finnish and English" in prompt
    assert llm_ocr.BLANK_MARKER in prompt


def test_gemini_request_and_cache(tmp_path):
    with patch("requests.post", return_value=_resp(200, GEMINI_OK)) as post:
        assert llm_ocr.ocr_png(PNG, "gemini") == "Hello\nworrier → warrior"
        assert llm_ocr.ocr_png(PNG, "gemini") == "Hello\nworrier → warrior"
    assert post.call_count == 1  # second call served from disk cache
    url = post.call_args.args[0]
    assert url.endswith("/models/gemini-3.1-pro-preview:generateContent")
    assert post.call_args.kwargs["headers"] == {"x-goog-api-key": "g-key"}
    parts = post.call_args.kwargs["json"]["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "image/png"
    assert "Transcribe" in parts[1]["text"]
    assert list((tmp_path / "ocr" / "gemini").rglob("*.txt"))


def test_gemini_skips_thought_parts():
    with patch("requests.post", return_value=_resp(200, GEMINI_THOUGHT)):
        assert llm_ocr.ocr_png(PNG, "gemini") == "Hello"


def test_anthropic_request(monkeypatch):
    monkeypatch.setenv("REMARKABLE_ANTHROPIC_MODEL", "claude-test")
    with patch("requests.post", return_value=_resp(200, ANTHROPIC_OK)) as post:
        assert llm_ocr.ocr_png(PNG, "anthropic") == "Run! ♡"
    assert post.call_args.args[0] == "https://api.anthropic.com/v1/messages"
    headers = post.call_args.kwargs["headers"]
    assert headers["x-api-key"] == "a-key" and headers["anthropic-version"]
    body = post.call_args.kwargs["json"]
    assert body["model"] == "claude-test"
    assert body["messages"][0]["content"][0]["source"]["media_type"] == "image/png"


def test_backends_cached_separately():
    with patch("requests.post", side_effect=[_resp(200, GEMINI_OK), _resp(200, ANTHROPIC_OK)]):
        assert llm_ocr.ocr_png(PNG, "gemini").startswith("Hello")
        assert llm_ocr.ocr_png(PNG, "anthropic") == "Run! ♡"


def test_blank_page_is_empty_string_and_cached():
    blank = {"candidates": [{"content": {"parts": [{"text": "[blank]"}]}}]}
    with patch("requests.post", return_value=_resp(200, blank)) as post:
        assert llm_ocr.ocr_png(PNG, "gemini") == ""
        assert llm_ocr.ocr_png(PNG, "gemini") == ""
    assert post.call_count == 1


def test_http_error_and_missing_key_return_none(monkeypatch):
    with patch("requests.post", return_value=_resp(429, {"error": "rate"})):
        assert llm_ocr.ocr_png(PNG, "anthropic") is None
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with patch("requests.post") as post:
        assert llm_ocr.ocr_png(PNG, "gemini") is None
    post.assert_not_called()


def test_failures_are_not_cached():
    with patch("requests.post", side_effect=[_resp(500, {}), _resp(200, GEMINI_OK)]) as post:
        assert llm_ocr.ocr_png(PNG, "gemini") is None
        assert llm_ocr.ocr_png(PNG, "gemini").startswith("Hello")
    assert post.call_count == 2


def test_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("REMARKABLE_OCR_CACHE", "0")
    with patch("requests.post", return_value=_resp(200, GEMINI_OK)) as post:
        llm_ocr.ocr_png(PNG, "gemini")
        llm_ocr.ocr_png(PNG, "gemini")
    assert post.call_count == 2


def test_backend_env_accepted(monkeypatch):
    for backend in ("gemini", "anthropic"):
        monkeypatch.setenv("REMARKABLE_OCR_BACKEND", backend)
        assert extract.get_ocr_backend() == backend


def test_extract_handwriting_uses_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("REMARKABLE_OCR_BACKEND", "anthropic")
    rm = tmp_path / "p.rm"
    rm.write_bytes(b"x")
    with (
        patch.object(extract, "_rm_to_png_bytes", return_value=PNG),
        patch("requests.post", return_value=_resp(200, ANTHROPIC_OK)),
    ):
        assert extract.extract_handwriting_ocr([rm]) == (["Run! ♡"], "anthropic")


def test_extract_handwriting_falls_back_when_llm_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("REMARKABLE_OCR_BACKEND", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_VISION_API_KEY", raising=False)
    rm = tmp_path / "p.rm"
    rm.write_bytes(b"x")
    with (
        patch.object(extract, "_rm_to_png_bytes", return_value=PNG),
        patch.object(extract, "_ocr_tesseract", return_value=["tess"]) as tess,
    ):
        assert extract.extract_handwriting_ocr([rm]) == (["tess"], "tesseract")
    tess.assert_called_once()


def test_extract_blank_pages_do_not_fall_back(monkeypatch, tmp_path):
    monkeypatch.setenv("REMARKABLE_OCR_BACKEND", "gemini")
    rm = tmp_path / "p.rm"
    rm.write_bytes(b"x")
    blank = {"candidates": [{"content": {"parts": [{"text": "[blank]"}]}}]}
    with (
        patch.object(extract, "_rm_to_png_bytes", return_value=PNG),
        patch("requests.post", return_value=_resp(200, blank)),
        patch.object(extract, "_ocr_tesseract") as tess,
    ):
        assert extract.extract_handwriting_ocr([rm]) == (None, "gemini")
    tess.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override", "llm_answer", "expected_backend", "expected_text"),
    [
        ("anthropic", "handwritten", "anthropic", "handwritten"),
        ("gemini", "", "gemini", None),  # blank page: no Google/Tesseract fallback
        ("gemini", None, "tesseract", "tesseract text"),  # backend failed: fall back
    ],
)
@patch("remarkable_mcp.tools.get_rmapi")
async def test_image_ocr_backend_override(
    mock_get_rmapi,
    mock_document,
    monkeypatch,
    override,
    llm_answer,
    expected_backend,
    expected_text,
):
    from remarkable_mcp import notebooks
    from test_server import _call_tool, _make_notebook_zip

    monkeypatch.setenv("REMARKABLE_OCR_BACKEND", "google")
    monkeypatch.delenv("GOOGLE_VISION_API_KEY", raising=False)
    mock_client = Mock()
    mock_get_rmapi.return_value = mock_client
    mock_document.is_folder = False
    mock_client.get_meta_items.return_value = [mock_document]
    mock_client.download.return_value = _make_notebook_zip(notebooks.page_rm_bytes("x"))

    with (
        patch("remarkable_mcp.llm_ocr.ocr_png", return_value=llm_answer) as llm,
        patch("remarkable_mcp.tools._ocr_png_google_vision", return_value=None) as google,
        patch("remarkable_mcp.tools._ocr_png_tesseract", return_value="tesseract text") as tess,
    ):
        result = await _call_tool(
            "remarkable_image",
            {
                "document": "Test Document",
                "include_ocr": True,
                "compatibility": True,
                "ocr_backend": override,
            },
        )
    data = json.loads(result.content[0].text)
    assert data["ocr_backend"] == expected_backend
    assert data["ocr_text"] == expected_text
    assert llm.call_args.args[1] == override
    if llm_answer is not None:
        google.assert_not_called()
        tess.assert_not_called()


@pytest.fixture
def mock_document():
    doc = Mock()
    doc.VissibleName = "Test Document"
    doc.ID = "doc-123"
    doc.Parent = ""
    doc.ModifiedClient = "2024-01-15T10:30:00Z"
    doc.is_cloud_archived = False
    return doc


def test_anthropic_workspace_header(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    with patch("requests.post", return_value=_resp(200, ANTHROPIC_OK)) as post:
        llm_ocr.ocr_png(PNG, "anthropic")
    assert post.call_args.kwargs["headers"]["anthropic-workspace-id"] == "wrkspc_test"


def test_anthropic_no_workspace_header_by_default(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    with patch("requests.post", return_value=_resp(200, ANTHROPIC_OK)) as post:
        llm_ocr.ocr_png(PNG, "anthropic")
    assert "anthropic-workspace-id" not in post.call_args.kwargs["headers"]


def test_retry_on_rate_limit(monkeypatch):
    monkeypatch.setenv("REMARKABLE_OCR_RETRIES", "3")
    monkeypatch.setattr("time.sleep", lambda s: None)
    with patch(
        "requests.post", side_effect=[_resp(429, {}), _resp(503, {}), _resp(200, GEMINI_OK)]
    ) as post:
        assert llm_ocr.ocr_png(PNG, "gemini").startswith("Hello")
    assert post.call_count == 3


def test_adjudication_prompt_and_cache():
    readings = {"anthropic": "maintaining", "gemini": "unexpectedly"}
    prompt = llm_ocr.build_adjudication_prompt(readings)
    assert "Reader A" in prompt and "maintaining" in prompt and "unexpectedly" in prompt
    assert "{likelier|other}" in prompt and "[?]" in prompt
    with patch(
        "requests.post",
        return_value=_resp(200, {"content": [{"type": "text", "text": "unexpectedly"}]}),
    ) as post:
        assert llm_ocr.adjudicate_png(PNG, readings) == "unexpectedly"
        assert llm_ocr.adjudicate_png(PNG, readings) == "unexpectedly"
    assert post.call_count == 1
    assert post.call_args.args[0] == "https://api.anthropic.com/v1/messages"


def test_vocabulary_from_env_and_file_reaches_adjudication(tmp_path, monkeypatch):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("# names\nAnna, Bruno\nCarla\n\nAnna\n", encoding="utf-8")
    monkeypatch.setenv("REMARKABLE_OCR_VOCABULARY_FILE", str(vocab))
    monkeypatch.setenv("REMARKABLE_OCR_VOCABULARY", "Dana,Eli")
    assert llm_ocr.get_vocabulary() == ["Dana", "Eli", "Anna", "Bruno", "Carla"]
    prompt = llm_ocr.build_adjudication_prompt({"anthropic": "anne", "gemini": "arne"})
    assert "Anna" in prompt and "hint, not a rule" in prompt
    # readers' prompt stays unchanged so cached readings remain valid
    assert "Anna" not in llm_ocr.build_prompt()


def test_no_vocabulary_no_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("REMARKABLE_OCR_VOCABULARY_FILE", str(tmp_path / "missing.txt"))
    monkeypatch.delenv("REMARKABLE_OCR_VOCABULARY", raising=False)
    assert llm_ocr.get_vocabulary() == []
    assert "hint, not a rule" not in llm_ocr.build_adjudication_prompt({"gemini": "a"})
