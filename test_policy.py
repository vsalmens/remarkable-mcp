"""Tests for REMARKABLE_EXCLUDE_PATHS and REMARKABLE_OCR_LANGUAGES."""

import base64
import json
from unittest.mock import Mock, patch

import pytest
from mcp import Client

from remarkable_mcp import policy
from remarkable_mcp.server import mcp


async def _call_tool(name: str, arguments: dict):
    async with Client(mcp) as client:
        return await client.call_tool(name, arguments)


def _item(item_id, name, parent="", is_folder=False):
    item = Mock()
    item.ID = item_id
    item.VissibleName = name
    item.Parent = parent
    item.is_folder = is_folder
    item.is_cloud_archived = False
    item.ModifiedClient = "2026-10-13T10:00:00Z"
    item.tags = []
    return item


@pytest.fixture
def library():
    return [
        _item("f-private", "Private", is_folder=True),
        _item("f-diary", "Diary", is_folder=True),
        _item("d-ah", "Ledger", parent="f-private"),
        _item("d-secret", "Diary", parent="f-private"),
        _item("d-diary", "Diary", parent="f-diary"),
        _item("d-top", "Sketchbook"),
    ]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("REMARKABLE_ROOT_PATH", raising=False)
    monkeypatch.setenv("REMARKABLE_EXCLUDE_PATHS", "/private/")
    monkeypatch.delenv("REMARKABLE_OCR_LANGUAGES", raising=False)


class TestExcludePaths:
    def test_parsing_and_matching(self, monkeypatch):
        monkeypatch.setenv("REMARKABLE_EXCLUDE_PATHS", " Private/ , /Work/Clients,,")
        assert policy.get_excluded_paths() == ["/private", "/work/clients"]
        assert policy.is_excluded("/Private")
        assert policy.is_excluded("/PRIVATE/Ledger")
        assert policy.is_excluded("/Work/Clients/X")
        assert not policy.is_excluded("/Privateer")
        assert not policy.is_excluded("/Work")
        assert not policy.is_excluded("/")

    def test_root_slash_is_ignored(self, monkeypatch):
        monkeypatch.setenv("REMARKABLE_EXCLUDE_PATHS", "/")
        assert policy.get_excluded_paths() == []
        assert not policy.is_excluded("/Anything")

    def test_unset_excludes_nothing(self, monkeypatch):
        monkeypatch.delenv("REMARKABLE_EXCLUDE_PATHS")
        assert not policy.is_excluded("/Private/Ledger")

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_browse_root_hides_excluded_folder(self, mock_get_rmapi, library):
        mock_get_rmapi.return_value = Mock(get_meta_items=Mock(return_value=library))
        data = json.loads((await _call_tool("remarkable_browse", {"path": "/"})).content[0].text)
        assert [f["name"] for f in data["folders"]] == ["Diary"]

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_browse_into_excluded_folder_denied(self, mock_get_rmapi, library):
        mock_get_rmapi.return_value = Mock(get_meta_items=Mock(return_value=library))
        for path in ("/Private", "/private/Ledger"):
            text = (await _call_tool("remarkable_browse", {"path": path})).content[0].text
            assert "access_denied" in text
            assert "Ledger" not in json.loads(text).get("documents", [])

    @pytest.mark.asyncio
    @patch("remarkable_mcp.tools.get_rmapi")
    async def test_query_and_recent_skip_excluded(self, mock_get_rmapi, library):
        mock_get_rmapi.return_value = Mock(get_meta_items=Mock(return_value=library))
        q = json.loads((await _call_tool("remarkable_browse", {"query": "a"})).content[0].text)
        assert all(not r["path"].lower().startswith("/private") for r in q["results"])
        r = json.loads((await _call_tool("remarkable_recent", {"limit": 10})).content[0].text)
        paths = [d["path"] for d in r["documents"]]
        assert paths and all(not p.lower().startswith("/private") for p in paths)

    def test_find_target_never_resolves_excluded(self, library):
        from remarkable_mcp.api import get_items_by_id
        from remarkable_mcp.tools import _find_target_document

        by_id = get_items_by_id(library)
        assert _find_target_document(library, by_id, "Ledger") is None
        assert _find_target_document(library, by_id, "/Private/Ledger") is None
        # Same display name outside the excluded folder still resolves
        found = _find_target_document(library, by_id, "Diary")
        assert found is not None and found.ID == "d-diary"

    def test_resources_not_registered_for_excluded(self, library):
        from remarkable_mcp.resources import _is_within_root

        assert not _is_within_root("/Private/Ledger", "/")
        assert _is_within_root("/Diary/Diary", "/")


class TestOcrLanguageHints:
    def test_no_hints_by_default(self):
        req = policy.vision_request("abc")
        assert "imageContext" not in req
        assert req["features"] == [{"type": "DOCUMENT_TEXT_DETECTION"}]

    def test_hints_parsed(self, monkeypatch):
        monkeypatch.setenv("REMARKABLE_OCR_LANGUAGES", "fi, en ,fi")
        assert policy.get_ocr_language_hints() == ["fi", "en"]
        req = policy.vision_request("abc")
        assert req["imageContext"] == {"languageHints": ["fi", "en"]}
        assert req["image"] == {"content": "abc"}

    def test_tools_png_ocr_sends_hints(self, monkeypatch, tmp_path):
        from remarkable_mcp import tools

        monkeypatch.setenv("GOOGLE_VISION_API_KEY", "AIzaTEST")
        monkeypatch.setenv("REMARKABLE_OCR_LANGUAGES", "fi,en")
        png = tmp_path / "p.png"
        png.write_bytes(b"\x89PNG fake")

        response = Mock(status_code=200)
        response.json.return_value = {"responses": [{"fullTextAnnotation": {"text": "Hei"}}]}
        with patch("requests.post", return_value=response) as post:
            assert tools._ocr_png_google_vision(png) == "Hei"
        sent = post.call_args.kwargs["json"]["requests"][0]
        assert sent["imageContext"] == {"languageHints": ["fi", "en"]}
        assert sent["image"]["content"] == base64.b64encode(b"\x89PNG fake").decode()
