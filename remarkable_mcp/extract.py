"""
Text extraction helpers for reMarkable documents.
"""

import json
import logging
import os
import re
import tempfile
import threading
import time
import zipfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as _xml_escape

from remarkable_mcp import llm_ocr
from remarkable_mcp.llm_ocr import LLM_OCR_BACKENDS
from remarkable_mcp.policy import get_ocr_language_hints, vision_request

logger = logging.getLogger(__name__)


def _rm_to_svg(rm_file_path: Path, output_svg_path: Path) -> bool:
    """Convert a .rm file to SVG using the built-in rmscene renderers.

    The ``rmc`` dependency has been dropped: it transitively pinned
    ``rmscene<0.7.0``, which cannot parse current-firmware ``.rm`` scene
    blocks (e.g. ``SceneInfo`` ``block_type=13``). We render via ``rmscene``
    directly using the in-repo v6/v5 renderers. v6 is tried first since it is
    the current firmware format.

    Writes the SVG content to output_svg_path and returns True on success.
    Returns False if no renderer could handle the file.

    Credit: khalid-hasan (PR #97), ljdutel (#95)
    """
    svg_content = _render_rm_v6_to_svg(rm_file_path) or _render_rm_v5_to_svg(rm_file_path)
    if svg_content is not None:
        output_svg_path.write_text(svg_content)
        return True

    return False


# reMarkable tablet screen dimensions (in pixels) - used as fallback
REMARKABLE_WIDTH = 1404
REMARKABLE_HEIGHT = 1872

# PDF points per stroke unit, keyed by the per-page SceneInfo.paper_size grid.
# The 1404x1872 value was calibrated against a device PDF export. Paper Pro
# scenes (1620x2160 grid) use the same scale: underlines drawn on a Paper Pro
# land under their words with 0.3177, while the panel-DPI estimate (0.3144)
# drifted up to half a line too high near the bottom of a Letter page.
_DEVICE_POINTS_PER_UNIT = {
    (1404, 1872): 0.3177,
    (1620, 2160): 0.3177,
}
REMARKABLE_PDF_POINTS_PER_UNIT = 0.3177


def _points_per_unit(paper_size: Optional[Tuple[float, float]]) -> float:
    """Return PDF points per stroke unit for a SceneInfo paper grid."""
    if paper_size:
        key = (int(round(paper_size[0])), int(round(paper_size[1])))
        if key in _DEVICE_POINTS_PER_UNIT:
            return _DEVICE_POINTS_PER_UNIT[key]
    return REMARKABLE_PDF_POINTS_PER_UNIT


def _annotation_page_viewbox(
    pdf_w_pt: float,
    pdf_h_pt: float,
    points_per_unit: float,
    *,
    centered_x: bool = True,
) -> Tuple[float, float, float, float]:
    """Return the stroke-space rectangle occupied by a PDF page."""
    view_w = pdf_w_pt / points_per_unit
    view_h = pdf_h_pt / points_per_unit
    view_x = -view_w / 2.0 if centered_x else 0.0
    return view_x, 0.0, view_w, view_h


def _pdf_ink_viewbox(pdf_w_pt: float, pdf_h_pt: float, blocks) -> Tuple[float, float, float, float]:
    """The stroke-space rectangle of a PDF page for one .rm page.

    v6 pages (with blocks) place the page at a fixed scale, centred on x = 0.
    Older v5 pages (blocks is None) store screen pixels of a 1404x1872 display
    on which the page was fitted to the screen and centred horizontally.
    """
    if blocks is None:
        fit = min(REMARKABLE_WIDTH / pdf_w_pt, REMARKABLE_HEIGHT / pdf_h_pt)
        width, height = pdf_w_pt * fit, pdf_h_pt * fit
        return -(REMARKABLE_WIDTH - width) / 2.0, 0.0, width, height
    paper_size = _v6_paper_size(blocks)
    return _annotation_page_viewbox(pdf_w_pt, pdf_h_pt, _points_per_unit(paper_size))


def _rewrite_svg_root(
    svg_content: str,
    *,
    viewbox: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    """Rewrite only root SVG geometry, leaving path stroke widths untouched."""
    import re

    root_match = re.search(r"<svg\b[^>]*>", svg_content, re.IGNORECASE)
    if root_match is None:
        return svg_content

    root = root_match.group(0)
    values = {
        "viewBox": " ".join(f"{value:.2f}" for value in viewbox),
        "width": str(width),
        "height": str(height),
    }
    for name, value in values.items():
        pattern = rf'(?<![-\w]){name}="[^"]*"'
        if re.search(pattern, root):
            root = re.sub(pattern, f'{name}="{value}"', root, count=1)
        else:
            root = root[:-1] + f' {name}="{value}">'

    return svg_content[: root_match.start()] + root + svg_content[root_match.end() :]


# Standard reMarkable background color (light cream/gray)
# Can be overridden via REMARKABLE_BACKGROUND_COLOR environment variable
_DEFAULT_BACKGROUND_COLOR = "#FBFBFB"


def get_background_color() -> str:
    """Get the background color, checking env var for override."""
    return os.environ.get("REMARKABLE_BACKGROUND_COLOR", _DEFAULT_BACKGROUND_COLOR)


OCR_BACKENDS = frozenset({"auto", "google", "tesseract", "gemini", "anthropic"})


def get_ocr_backend() -> str:
    """Return a supported OCR backend, falling back to ``auto`` safely."""
    backend = os.environ.get("REMARKABLE_OCR_BACKEND", "auto").strip().lower()
    if backend not in OCR_BACKENDS:
        supported = ", ".join(sorted(OCR_BACKENDS))
        if backend == "sampling":
            logger.warning(
                "The 'sampling' OCR backend has been removed. "
                "Falling back to 'auto'. Set REMARKABLE_OCR_BACKEND to one of: %s.",
                supported,
            )
        else:
            logger.warning(
                "Unsupported REMARKABLE_OCR_BACKEND value %r. "
                "Falling back to 'auto'. Supported values are: %s.",
                backend,
                supported,
            )
        return "auto"
    return backend


# For backwards compatibility, expose as module constant (evaluated at import)
# Use get_background_color() for runtime evaluation of env var
REMARKABLE_BACKGROUND_COLOR = get_background_color()

# Margin around content when using content-based bounding box (in pixels)
CONTENT_MARGIN = 50

# Target long-edge resolution (px) for a full-page canvas render. The page is
# rasterised at this resolution preserving the page aspect; the displayed image
# is then scaled by the host, so this only sets crispness, not layout.
FULL_PAGE_TARGET_LONG_EDGE = 1872

# Cache TTL in seconds (5 minutes)
CACHE_TTL_SECONDS = 300

# Module-level cache for OCR results (full document)
# Key: doc_id
# Value: {"result": extraction_result, "include_ocr": bool, "timestamp": float}
_extraction_cache: Dict[str, Dict[str, Any]] = {}

_cache_lock = threading.RLock()
_cache_generation: Dict[str, int] = {}
_global_cache_generation = 0


def _is_cache_valid(cached: Dict[str, Any]) -> bool:
    """Check if a cached entry is still valid based on TTL."""
    if "timestamp" not in cached:
        return True  # Old cache entries without timestamp are valid
    return (time.time() - cached["timestamp"]) < CACHE_TTL_SECONDS


def clear_extraction_cache(doc_id: Optional[str] = None) -> None:
    """
    Clear the extraction cache.

    Args:
        doc_id: If provided, only clear cache for this document.
                If None, clear the entire cache.
    """
    global _global_cache_generation

    with _cache_lock:
        if doc_id:
            _cache_generation[doc_id] = _cache_generation.get(doc_id, 0) + 1
            _extraction_cache.pop(doc_id, None)
        else:
            _global_cache_generation += 1
            _extraction_cache.clear()
            _cache_generation.clear()


def _cache_token(doc_id: str) -> tuple[int, int]:
    with _cache_lock:
        return _global_cache_generation, _cache_generation.get(doc_id, 0)


def _cache_extraction_result_if_current(
    doc_id: str,
    result: Dict[str, Any],
    include_ocr: bool,
    token: tuple[int, int],
) -> bool:
    """Cache only if no mutation invalidated this document during extraction."""
    with _cache_lock:
        if token != (
            _global_cache_generation,
            _cache_generation.get(doc_id, 0),
        ):
            return False
        _extraction_cache[doc_id] = {
            "result": result,
            "include_ocr": include_ocr,
            "timestamp": time.time(),
        }
        return True


def get_cached_ocr_result(
    doc_id: str,
    include_ocr: bool = True,
    ocr_backend: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get cached OCR result for a document if available and valid.

    Args:
        doc_id: Document ID to look up
        include_ocr: Whether OCR content is required
        ocr_backend: If specified, only return cache if it was produced by this backend.
                     Use "google" or "tesseract". None accepts any backend.

    Returns:
        Cached result dict or None if not cached/expired/wrong backend
    """
    with _cache_lock:
        if doc_id in _extraction_cache:
            cached = _extraction_cache[doc_id]
            if (cached["include_ocr"] or not include_ocr) and _is_cache_valid(cached):
                if ocr_backend is not None:
                    cached_backend = cached["result"].get("ocr_backend")
                    if cached_backend != ocr_backend:
                        return None
                return cached["result"]
    return None


def cache_ocr_result(
    doc_id: str,
    result: Dict[str, Any],
    include_ocr: bool = True,
) -> None:
    """
    Cache an OCR result for a document.

    Args:
        doc_id: Document ID
        result: Extraction result dict with keys: typed_text, highlights,
                handwritten_text, pages, page_ids, ocr_backend
        include_ocr: Whether this result includes OCR content
    """
    with _cache_lock:
        _extraction_cache[doc_id] = {
            "result": result,
            "include_ocr": include_ocr,
            "timestamp": time.time(),
        }


def find_similar_documents(query: str, documents: List, limit: int = 5) -> List[str]:
    """Find documents with similar names for 'did you mean' suggestions."""
    query_lower = query.lower()
    scored = []
    for doc in documents:
        name = doc.VissibleName
        # Use sequence matcher for fuzzy matching
        ratio = SequenceMatcher(None, query_lower, name.lower()).ratio()
        # Boost partial matches
        if query_lower in name.lower():
            ratio += 0.3
        scored.append((name, ratio))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, score in scored[:limit] if score > 0.3]


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file using PyMuPDF.

    Returns the full text content of the PDF.
    """
    try:
        import fitz  # PyMuPDF

        text_parts = []
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{page_text.strip()}")

        return "\n\n".join(text_parts) if text_parts else ""
    except ImportError:
        return ""
    except Exception:
        return ""


def extract_text_from_epub(epub_path: Path) -> str:
    """
    Extract text from an EPUB file.

    Returns the full text content of the EPUB.
    """
    try:
        from bs4 import BeautifulSoup
        from ebooklib import ITEM_DOCUMENT, epub

        book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
        text_parts = []

        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                # Get text, preserving some structure
                text = soup.get_text(separator="\n", strip=True)
                if text:
                    text_parts.append(text)

        return "\n\n".join(text_parts) if text_parts else ""
    except ImportError:
        return ""
    except Exception:
        return ""


def extract_text_from_rm_file(rm_file_path: Path) -> List[str]:
    """
    Extract typed text from a .rm file using rmscene.

    This extracts text that was typed via Type Folio or on-screen keyboard.
    Does NOT require OCR - text is stored natively in v6 .rm files.
    """
    try:
        from rmscene import read_blocks
        from rmscene.scene_items import Text
        from rmscene.text import TextDocument

        with open(rm_file_path, "rb") as f:
            blocks = list(read_blocks(f))

        # The page's typed text lives on the RootTextBlock, not among the
        # root group's children (those are the ink layers), so scan the raw
        # blocks the same way the renderer does.
        text_item = next(
            (b.value for b in blocks if isinstance(getattr(b, "value", None), Text)),
            None,
        )
        if text_item is None:
            return []

        # TextDocument resolves the CRDT character sequence into ordered,
        # styled paragraphs; iterating Text.items directly yields CRDT ids
        # and fragments in insertion order, not readable lines.
        doc = TextDocument.from_scene_item(text_item)
        return [line for line in (str(para).strip() for para in doc.contents) if line]

    except ImportError:
        return []  # rmscene not available
    except Exception:
        # Log but don't fail - file might be older format
        return []


def _parse_hex_color(hex_color: str) -> tuple:
    """Parse a hex color string to RGBA tuple.

    Supports #RRGGBB (RGB) and #RRGGBBAA (RGBA) formats.

    Args:
        hex_color: Hex color string (e.g., "#FFFFFF" or "#FFFFFF80")

    Returns:
        Tuple of (r, g, b, a) values (0-255)
    """
    if not hex_color.startswith("#"):
        return (255, 255, 255, 255)

    hex_str = hex_color.lstrip("#")
    if len(hex_str) == 6:
        r, g, b = tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
        return (r, g, b, 255)
    elif len(hex_str) == 8:
        r, g, b, a = tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4, 6))
        return (r, g, b, a)
    else:
        return (255, 255, 255, 255)


def _get_svg_content_bounds(svg_path: Path) -> Optional[tuple]:
    """
    Parse SVG file to get the content bounding box from viewBox.

    Args:
        svg_path: Path to the SVG file

    Returns:
        Tuple of (min_x, min_y, width, height) or None if not determinable
    """
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Try to get viewBox attribute
        viewbox = root.get("viewBox")
        if viewbox:
            parts = viewbox.split()
            if len(parts) == 4:
                return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))

        # Fallback to width/height attributes
        width = root.get("width")
        height = root.get("height")
        if width and height:
            # Remove 'px' suffix if present
            w = float(width.replace("px", ""))
            h = float(height.replace("px", ""))
            return (0, 0, w, h)

        return None
    except Exception:
        return None


CONTENT_PADDING = 20  # Padding around content bounds for SVG viewBox


def _v6_stroke_width(tool, points, thickness_scale) -> float:
    """Average drawn width of a v6 stroke in stroke units, per pen type.

    v6 points store width x4 and pressure 0-255. The formulas follow how each
    tablet pen draws (after the rmc exporter's pen models): a fineliner has a
    fixed width set by the thickness setting, the others follow point width
    and pressure. The old flat rule (0.8 x stored width) drew calligraphy and
    ballpoint lines about twice as thick as on the tablet.
    """
    widths = [getattr(p, "width", 0) for p in points]
    pressures = [getattr(p, "pressure", 0) for p in points]
    w = (sum(widths) / len(widths)) / 4.0 if widths else 2.0
    pr = (sum(pressures) / len(pressures)) / 255.0 if pressures else 0.5
    return _pen_width(tool, w, pr, thickness_scale)


def _pen_width(tool, w: float, pr: float, thickness_scale=None) -> float:
    """Drawn width in stroke units from point width ``w`` and pressure ``pr`` (0-1)."""
    ts = float(thickness_scale or 2.0)
    if tool in (4, 17):  # fineliner
        base = 1.3 * ts**2.1
    elif tool in (2, 15):  # ballpoint
        base = 0.5 + pr + w
    elif tool == 21:  # calligraphy
        base = 0.9 * (1 + 1.4 * pr) * w
    elif tool in (3, 16):  # marker
        base = 0.9 * w
    elif tool in (1, 14):  # pencil
        base = 0.7 * (0.8 * ts + 0.5 * pr) * w
    elif tool in (7, 13):  # mechanical pencil
        base = ts**2
    elif tool in (0, 12):  # paintbrush
        base = 0.7 * (1 + 1.4 * pr) * w
    else:
        base = w
    return max(0.4, min(base, 12.0))


def _svg_from_paths(paths: list, all_coords: list) -> Optional[str]:
    """Build SVG string with viewBox computed from actual content bounds."""
    if not paths or not all_coords:
        return None

    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    min_x = min(xs) - CONTENT_PADDING
    min_y = min(ys) - CONTENT_PADDING
    w = max(xs) - min_x + CONTENT_PADDING
    h = max(ys) - min_y + CONTENT_PADDING

    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{min_x:.0f} {min_y:.0f} {w:.0f} {h:.0f}" '
        f'width="{w:.0f}" height="{h:.0f}">'
        f"{''.join(paths)}</svg>"
    )


def _render_rm_v5_to_svg(rm_file_path: Path) -> Optional[str]:
    """
    Render a v5 .rm file (reMarkable .lines format) to SVG.

    The v5 binary format stores layers of strokes, where each stroke
    has pen/color metadata and a sequence of (x, y, speed, tilt, width,
    pressure) segments.
    """
    import struct

    # Pen IDs that are highlighters (both old and v5 mappings)
    HIGHLIGHTER_PENS = {5, 18}
    # Pen IDs that are erasers — skip rendering
    ERASER_PENS = {6, 7, 8}

    # Color mapping by pen type
    STROKE_COLORS = {0: "black", 1: "gray", 2: "white"}
    HIGHLIGHT_COLORS = {0: "#FFD700", 1: "#FFD700", 2: "#FFD700"}

    try:
        with open(rm_file_path, "rb") as f:
            header = f.read(43)
            if b"version=5" not in header:
                return None

            nlayers = struct.unpack("<I", f.read(4))[0]
            paths = []
            all_coords = []

            for _ in range(nlayers):
                nstrokes = struct.unpack("<I", f.read(4))[0]

                for _ in range(nstrokes):
                    pen, color, _pad, _w, _unk, nsegments = struct.unpack("<IIIIfI", f.read(24))

                    segments = []
                    for _ in range(nsegments):
                        x, y, speed, tilt, width, pressure = struct.unpack("<ffffff", f.read(24))
                        segments.append((x, y, width, pressure))

                    if not segments:
                        continue

                    # Skip eraser strokes
                    if pen in ERASER_PENS:
                        continue

                    is_highlighter = pen in HIGHLIGHTER_PENS
                    if is_highlighter:
                        stroke_color = HIGHLIGHT_COLORS.get(color, "#FFD700")
                        avg_width = sum(s[2] for s in segments) / len(segments)
                        stroke_width = max(10.0, min(avg_width * 2.0, 40.0))
                        opacity = ' opacity="0.35"'
                    else:
                        stroke_color = STROKE_COLORS.get(color, "black")
                        avg_width = sum(s[2] for s in segments) / len(segments)
                        avg_pressure = sum(s[3] for s in segments) / len(segments)
                        # v5 stores width in pixels and pressure as 0-1
                        stroke_width = _pen_width(pen, avg_width, avg_pressure)
                        opacity = ""

                    d = f"M {segments[0][0]:.1f} {segments[0][1]:.1f}"
                    d += "".join(f" L {s[0]:.1f} {s[1]:.1f}" for s in segments[1:])
                    all_coords.extend((s[0], s[1]) for s in segments)

                    paths.append(
                        f'<path d="{d}" stroke="{stroke_color}" '
                        f'stroke-width="{stroke_width:.1f}" '
                        f'fill="none" stroke-linecap="round" '
                        f'stroke-linejoin="round"{opacity} '
                        f'data-tool="{pen}" data-color="{color}"/>'
                    )

        return _svg_from_paths(paths, all_coords)
    except Exception:
        return None


def _v6_blocks(rm_file_path: Path) -> Optional[list]:
    """Read v6 .rm scene blocks, or None if the file is not v6 / unreadable."""
    try:
        from rmscene import read_blocks
    except ImportError:
        return None
    try:
        with open(rm_file_path, "rb") as f:
            header = f.read(43)
            if b"version=6" not in header:
                return None
            f.seek(0)
            return list(read_blocks(f))
    except Exception:
        return None


def _v6_paper_size(blocks: list) -> Tuple[float, float]:
    """Page extent (W, H) in stroke units, read from SceneInfo.paper_size.

    This is the authoritative stroke-coordinate extent for the page and the
    same value the write path (``strokes.page_geometry``) maps normalized
    coordinates into. Building a full-page render from it therefore makes the
    displayed image share ONE coordinate space with written strokes (so the
    drawing overlay and the write tool land ink in the same place). Falls back
    to the standard reMarkable page when no SceneInfo is present.
    """
    for b in blocks:
        ps = getattr(b, "paper_size", None)
        if ps and len(ps) == 2 and ps[0] and ps[1]:
            try:
                return float(ps[0]), float(ps[1])
            except (TypeError, ValueError):
                continue
    return float(REMARKABLE_WIDTH), float(REMARKABLE_HEIGHT)


def _v6_group_offsets(blocks: list, anchor_pos: dict) -> dict:
    """Map group node ``CrdtId``s to cumulative ``(dx, dy)`` anchor offsets.

    In the v6 format a group of strokes can be anchored to a typed-text
    character (``Group.anchor_id``): its ink is stored relative to that anchor
    and the device translates it to the anchored line's Y as text reflows.
    ``anchor_pos`` comes from :func:`_v6_text_elements_with_bounds`. Offsets
    accumulate up the parent chain so groups nested under an anchored group
    inherit its translation. Returns ``{}`` when there is nothing to anchor.
    """
    if not anchor_pos:
        return {}

    parents: dict = {}
    own: dict = {}
    for block in blocks:
        # SceneTreeBlock links a node to its parent.
        tree_id = getattr(block, "tree_id", None)
        if tree_id is not None and hasattr(block, "parent_id"):
            parents[tree_id] = block.parent_id
        # TreeNodeBlock carries the group's anchor attributes.
        group = getattr(block, "group", None)
        if group is None:
            continue
        anchor_id = getattr(group, "anchor_id", None)
        if anchor_id is None:
            continue
        anchor_key = getattr(anchor_id, "value", None)
        if anchor_key not in anchor_pos:
            continue
        origin_x = getattr(group, "anchor_origin_x", None)
        dx = float(origin_x.value) if origin_x is not None else 0.0
        own[group.node_id] = (dx, float(anchor_pos[anchor_key]))

    if not own:
        return {}

    offsets: dict = {}

    def resolve(node_id) -> Tuple[float, float]:
        if node_id in offsets:
            return offsets[node_id]
        dx = dy = 0.0
        cur = node_id
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            if cur in own:
                odx, ody = own[cur]
                dx += odx
                dy += ody
            cur = parents.get(cur)
        offsets[node_id] = (dx, dy)
        return offsets[node_id]

    for node_id in set(parents) | set(own):
        resolve(node_id)
    return offsets


def _v6_paths_from_blocks(blocks: list, anchor_pos: Optional[dict] = None) -> Tuple[list, list]:
    """Build SVG ``<path>`` strings + a flat coordinate list from v6 blocks.

    ``anchor_pos`` (from :func:`_v6_text_elements_with_bounds`) enables
    text-anchored ink translation: strokes whose parent group is anchored to a
    typed-text line are shifted to that line's position, matching where the
    device draws them. Without it strokes render at their stored coordinates,
    which for anchored groups is on top of the typed text block.
    """
    # Integer pen/color values (rmscene exposes ints on blocks).
    HIGHLIGHTER_PENS = {5, 18}  # HIGHLIGHTER_1, HIGHLIGHTER_2
    ERASER_PENS = {6, 8}  # ERASER, ERASER_AREA
    COLOR_MAP = {
        0: "black",  # BLACK
        1: "#808080",  # GRAY
        2: "white",  # WHITE
        3: "#FFD700",  # YELLOW
        4: "#00A000",  # GREEN
        5: "#FF69B4",  # PINK
        6: "#4169E1",  # BLUE
        7: "#E00000",  # RED
        8: "#A0A0A0",  # GRAY_OVERLAP
        9: "#FFD700",  # HIGHLIGHT
        10: "#00C000",  # GREEN_2
        11: "#00CED1",  # CYAN
        12: "#FF00FF",  # MAGENTA
        13: "#FFD700",  # YELLOW_2
    }

    group_offsets = _v6_group_offsets(blocks, anchor_pos) if anchor_pos else {}

    paths: list = []
    all_coords: list = []
    for block in blocks:
        if not hasattr(block, "item") or not hasattr(block.item, "value"):
            continue
        line = block.item.value
        dx, dy = group_offsets.get(getattr(block, "parent_id", None), (0.0, 0.0))

        # Text highlights are GlyphRange items: they carry `rectangles`
        # (highlight boxes over selected text) rather than stroke `points`.
        # Emit a translucent filled rect per rectangle in the highlight colour.
        if getattr(line, "rectangles", None):
            hl_color = getattr(line, "color", 9)
            hl_color = hl_color.value if hasattr(hl_color, "value") else hl_color
            hl_fill = COLOR_MAP.get(hl_color, "#FFD700")
            for r in line.rectangles:
                paths.append(
                    f'<rect x="{r.x + dx:.1f}" y="{r.y + dy:.1f}" '
                    f'width="{r.w:.1f}" height="{r.h:.1f}" '
                    f'fill="{hl_fill}" opacity="0.35" stroke="none"/>'
                )
                all_coords.append((r.x + dx, r.y + dy))
                all_coords.append((r.x + dx + r.w, r.y + dy + r.h))
            continue

        if not hasattr(line, "points") or not line.points:
            continue

        tool = line.tool if hasattr(line, "tool") else None
        color = line.color if hasattr(line, "color") else 0
        # Convert enums to int if needed
        tool = tool.value if hasattr(tool, "value") else tool
        color = color.value if hasattr(color, "value") else color

        if tool in ERASER_PENS:
            continue

        is_highlighter = tool in HIGHLIGHTER_PENS
        stroke_color = COLOR_MAP.get(color, "black")

        if is_highlighter:
            avg_width = (
                sum(p.width for p in line.points) / len(line.points)
                if all(hasattr(p, "width") for p in line.points)
                else 20.0
            )
            stroke_width = max(10.0, min(avg_width * 2.0, 40.0))
            opacity = ' opacity="0.35"'
        else:
            stroke_width = _v6_stroke_width(
                tool, line.points, getattr(line, "thickness_scale", None)
            )
            opacity = ""

        d = f"M {line.points[0].x + dx:.1f} {line.points[0].y + dy:.1f}"
        d += "".join(f" L {p.x + dx:.1f} {p.y + dy:.1f}" for p in line.points[1:])
        all_coords.extend((p.x + dx, p.y + dy) for p in line.points)

        paths.append(
            f'<path d="{d}" stroke="{stroke_color}" '
            f'stroke-width="{stroke_width:.1f}" '
            f'fill="none" stroke-linecap="round" '
            f'stroke-linejoin="round"{opacity} data-tool="{tool}" data-color="{color}"/>'
        )
    return paths, all_coords


# reMarkable typed-text layout, in raw stroke units. Calibrated against
# device-rendered page thumbnails (a reMarkable Paper Pro / "Ferrari", which
# normalizes typed text into the same 1404x1872 coordinate space as the rM1/2).
# These differ from the rmc exporter's constants (github.com/ricklupton/rmc):
# rmc lays text out in a 72/226-DPI-scaled space and its plain font (7pt) and
# top offset (-88) render text noticeably smaller and higher than the device
# actually draws it. Values here are in the reference 1404x1872 space and are
# scaled by the page's real height (see ``_TEXT_REF_PAGE_HEIGHT``) so they adapt
# to other geometries (e.g. reMarkable Move) that normalize differently.
_TEXT_REF_PAGE_HEIGHT = 1872.0
_TEXT_TOP_Y = -39.0
_TEXT_DEFAULT_LINE_HEIGHT = 70.0
_TEXT_DEFAULT_FONT_SIZE = 30.0
# Continuation lines of a wrapped paragraph are spaced tighter than the gap
# between paragraphs (device uses ~44 vs ~70 units for plain text).
_TEXT_WRAP_LINE_RATIO = 44.0 / 70.0
# Average glyph advance as a fraction of font size, used only to wrap a
# paragraph to the text-box width (SVG <text> does not wrap on its own).
_TEXT_CHAR_ADVANCE = 0.5


def _wrap_text(text: str, max_width: float, char_advance: float) -> List[str]:
    """Greedily word-wrap ``text`` to ``max_width`` (stroke units).

    SVG ``<text>`` does not wrap, so paragraphs wider than the text box are
    split here the way the device wraps them. Line width is approximated as
    ``len(line) * char_advance`` -- good enough for a faithful preview without
    embedding the device font. A single word longer than the box is left on its
    own line rather than split. Returns ``[text]`` when no wrapping applies.
    """
    if max_width <= 0 or char_advance <= 0:
        return [text]
    lines: List[str] = []
    cur = ""
    for word in text.split(" "):
        trial = word if not cur else f"{cur} {word}"
        if not cur or len(trial) * char_advance <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _v6_text_svg_elements(blocks: list) -> list:
    """Build SVG ``<text>`` strings for typed text (a RootTextBlock) on a page.

    Thin wrapper around :func:`_v6_text_elements_with_bounds` for callers (e.g.
    the full-page render) that fix the viewBox to the page extent and so do not
    need the text's bounding coordinates.
    """
    return _v6_text_elements_with_bounds(blocks)[0]


# Special anchor CrdtIds the device uses for ink groups pinned to the page
# itself rather than to a typed character: (0, 281474976710654) anchors at the
# top of the text block, (0, 281474976710655) below its last line.
_ANCHOR_TEXT_TOP = (0, 281474976710654)
_ANCHOR_TEXT_BOTTOM = (0, 281474976710655)


def _v6_text_elements_with_bounds(blocks: list) -> Tuple[list, list, dict]:
    """Build SVG ``<text>`` strings, bounding coords and anchor map for typed text.

    Returns ``(elements, coords, anchor_pos)``. ``coords`` is a flat list of
    ``(x, y)`` extent points (in the page's stroke/screen units, center-origin
    X) so a content-cropped render can size its viewBox to include the text.
    ``anchor_pos`` maps text-character ``CrdtId``s (plus the special
    top/bottom-of-text ids) to the page-space Y of the line holding that
    character: ink groups anchored to a character must be translated by that Y
    (see :func:`_v6_paths_from_blocks`), which is how the device keeps
    handwriting positioned relative to text as the typed block reflows.
    Returns ``([], [], {})`` when the page has no typed text (the common case
    for handwritten notebooks) or when rmscene's text helpers are unavailable.
    """
    try:
        from rmscene.scene_items import ParagraphStyle, Text
        from rmscene.tagged_block_common import CrdtId
        from rmscene.text import TextDocument
    except ImportError:
        return [], [], {}

    text_item = next(
        (b.value for b in blocks if isinstance(getattr(b, "value", None), Text)),
        None,
    )
    if text_item is None:
        return [], [], {}

    # Blank pages we synthesize carry an empty RootTextBlock; skip them so we
    # neither emit empty <text> nodes nor trigger rmscene's empty-item warning.
    try:
        if not any(isinstance(v, str) and v.strip() for v in text_item.items.values()):
            return [], [], {}
    except Exception:
        pass

    line_heights = {
        ParagraphStyle.PLAIN: 70.0,
        ParagraphStyle.HEADING: 150.0,
        ParagraphStyle.BOLD: 70.0,
        ParagraphStyle.BULLET: 35.0,
        ParagraphStyle.BULLET2: 35.0,
        ParagraphStyle.CHECKBOX: 35.0,
        ParagraphStyle.CHECKBOX_CHECKED: 35.0,
    }
    font_sizes = {
        ParagraphStyle.HEADING: 60.0,
        ParagraphStyle.BOLD: 34.0,
    }

    try:
        doc = TextDocument.from_scene_item(text_item)
    except Exception:
        return [], [], {}

    pos_x = float(getattr(text_item, "pos_x", 0.0) or 0.0)
    pos_y = float(getattr(text_item, "pos_y", 0.0) or 0.0)
    box_width = float(getattr(text_item, "width", 0.0) or 0.0)

    # Scale the reference metrics to the page's real height so text on devices
    # that normalize to a different coordinate space (e.g. reMarkable Move)
    # stays proportional. Identity for the standard 1404x1872 page.
    _, paper_h = _v6_paper_size(blocks)
    scale = paper_h / _TEXT_REF_PAGE_HEIGHT if paper_h else 1.0

    elements: list = []
    coords: list = []
    anchor_pos: dict = {}
    y_offset = _TEXT_TOP_Y * scale
    anchor_pos[CrdtId(*_ANCHOR_TEXT_TOP)] = pos_y + y_offset
    for para in doc.contents:
        style = para.style.value if getattr(para, "style", None) is not None else None
        line_height = line_heights.get(style, _TEXT_DEFAULT_LINE_HEIGHT) * scale
        size = font_sizes.get(style, _TEXT_DEFAULT_FONT_SIZE) * scale
        text = str(para).strip()
        # Every character in this paragraph anchors ink at the paragraph's own
        # Y (the running offset before this paragraph consumes its lines), the
        # same reference point the device and the rmc exporter use.
        para_anchor_y = pos_y + y_offset
        try:
            anchor_pos[para.start_id] = para_anchor_y
            for subpara in para.contents:
                for char_id in getattr(subpara, "i", []) or []:
                    anchor_pos[char_id] = para_anchor_y
        except Exception:
            pass
        if not text:
            # A blank paragraph still consumes a line (e.g. spacing under a title).
            y_offset += line_height
            continue
        family = "serif" if style == ParagraphStyle.HEADING else "sans-serif"
        weight = (
            ' font-weight="bold"' if style in (ParagraphStyle.BOLD, ParagraphStyle.HEADING) else ""
        )
        # Wrap long paragraphs to the text box; each wrapped line takes a line,
        # with continuation lines spaced tighter than a paragraph break.
        for idx, line in enumerate(_wrap_text(text, box_width, size * _TEXT_CHAR_ADVANCE)):
            y_offset += line_height if idx == 0 else line_height * _TEXT_WRAP_LINE_RATIO
            baseline = pos_y + y_offset
            elements.append(
                f'<text x="{pos_x:.1f}" y="{baseline:.1f}" '
                f'font-family="{family}" font-size="{size:.1f}"{weight} '
                f'fill="black" xml:space="preserve">{_xml_escape(line)}</text>'
            )
            # Estimate the line's extent for crop sizing, capped at the box width.
            est_width = len(line) * size * _TEXT_CHAR_ADVANCE
            if box_width > 0:
                est_width = min(est_width, box_width)
            coords.append((pos_x, baseline - size))
            coords.append((pos_x + est_width, baseline + size * 0.3))
    anchor_pos[CrdtId(*_ANCHOR_TEXT_BOTTOM)] = pos_y + y_offset
    return elements, coords, anchor_pos


def _render_rm_v6_to_svg(rm_file_path: Path) -> Optional[str]:
    """
    Render a v6 .rm file to a content-cropped SVG (read-only viewing).

    This handles newer reMarkable firmware that uses the v6 format with
    scene tree blocks, including proper highlighter and color support. The
    viewBox is cropped to the ink bounding box; for a full-page render that
    shares the page's own coordinate space, see ``render_rm_file_full_page_png``.
    """
    blocks = _v6_blocks(rm_file_path)
    if blocks is None:
        return None
    try:
        text_elements, text_coords, anchor_pos = _v6_text_elements_with_bounds(blocks)
        paths, all_coords = _v6_paths_from_blocks(blocks, anchor_pos)
        # Typed text is drawn under strokes (handwriting layers on top), and its
        # extent is folded into the crop so a text-only page still renders.
        return _svg_from_paths(text_elements + paths, all_coords + text_coords)
    except Exception:
        return None


def render_rm_file_to_png(
    rm_file_path: Path, background_color: Optional[str] = None
) -> Optional[bytes]:
    """
    Render a .rm file to PNG image bytes.

    Uses the rmscene renderers to convert .rm to SVG, then PyMuPDF to convert to PNG.
    The output is sized based on the SVG content bounds with a margin.

    Args:
        rm_file_path: Path to the .rm file
        background_color: Background color (e.g., "#FFFFFF", "transparent", None).
                         None means transparent. Use REMARKABLE_BACKGROUND_COLOR
                         for the standard reMarkable paper color.

    Returns:
        PNG image bytes, or None if rendering failed
    """
    tmp_svg_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
            tmp_svg_path = Path(tmp_svg.name)

        # Convert .rm to SVG via the rmscene renderers
        if not _rm_to_svg(rm_file_path, tmp_svg_path):
            return None

        # Get content bounds from SVG
        bounds = _get_svg_content_bounds(tmp_svg_path)
        if bounds:
            # Use content bounds with margin
            _, _, content_width, content_height = bounds
            output_width = int(content_width) + 2 * CONTENT_MARGIN
            output_height = int(content_height) + 2 * CONTENT_MARGIN
        else:
            # Fallback to standard reMarkable dimensions
            output_width = REMARKABLE_WIDTH
            output_height = REMARKABLE_HEIGHT

        return _svg_string_to_png(
            tmp_svg_path.read_text(),
            output_width,
            output_height,
            background_color,
        )
    except Exception:
        return None
    finally:
        if tmp_svg_path:
            tmp_svg_path.unlink(missing_ok=True)


def _svg_full_page(paths: list, paper_w: float, paper_h: float) -> str:
    """Wrap paths in an SVG whose viewBox spans the WHOLE page in stroke units.

    The page coordinate system is center-origin in X (x in [-W/2, W/2]) and
    top-origin in Y (y in [0, H]) — the same mapping ``strokes._map_point``
    uses (``rm_x = (nx-0.5)*W``, ``rm_y = ny*H``). So a point drawn at
    normalized (nx, ny) over the rendered image lands at exactly the stroke
    coordinate the write tool will use. Empty ``paths`` yields a blank page.
    """
    min_x = -paper_w / 2.0
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{min_x:.0f} 0 {paper_w:.0f} {paper_h:.0f}" '
        f'width="{paper_w:.0f}" height="{paper_h:.0f}">'
        f"{''.join(paths)}</svg>"
    )


def _svg_string_to_png(
    svg: str, output_width: int, output_height: int, background_color: Optional[str]
) -> Optional[bytes]:
    """Rasterize an SVG string to exact-size PNG bytes without system Cairo."""
    import io

    try:
        import fitz
    except ImportError:
        logger.error(
            "SVG rasterization requires PyMuPDF. Reinstall remarkable-mcp so its "
            "runtime dependencies are available."
        )
        return None

    if output_width < 1 or output_height < 1:
        logger.error(
            "SVG rasterization requires positive output dimensions, got %sx%s.",
            output_width,
            output_height,
        )
        return None

    doc = None
    try:
        doc = fitz.open(stream=svg.encode("utf-8"), filetype="svg")
        if len(doc) != 1:
            raise ValueError(f"expected one SVG page, found {len(doc)}")

        page = doc[0]
        if page.rect.width <= 0 or page.rect.height <= 0:
            raise ValueError(f"SVG has invalid dimensions: {page.rect}")

        matrix = fitz.Matrix(
            output_width / page.rect.width,
            output_height / page.rect.height,
        )
        pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=True)
        png = pixmap.tobytes("png")

        from PIL import Image as PILImage

        image = PILImage.open(io.BytesIO(png)).convert("RGBA")
        if image.size != (output_width, output_height):
            image = image.resize((output_width, output_height), PILImage.Resampling.LANCZOS)

        if background_color is not None:
            red, green, blue, alpha = _parse_hex_color(background_color)
            background = PILImage.new("RGBA", image.size, (red, green, blue, alpha))
            image = PILImage.alpha_composite(background, image)
            if alpha == 255:
                image = image.convert("RGB")

        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
    except Exception as exc:
        logger.error(
            "PyMuPDF could not rasterize SVG to PNG (%sx%s): %s",
            output_width,
            output_height,
            exc,
        )
        return None
    finally:
        if doc is not None:
            doc.close()


def render_rm_file_full_page_png(
    rm_file_path: Path, background_color: Optional[str] = None
) -> Optional[Tuple[bytes, Tuple[float, float]]]:
    """Render a v6 .rm page to a FULL-PAGE PNG (not cropped to ink).

    Unlike ``render_rm_file_to_png`` (which crops the viewBox to the ink
    bounding box), this renders the whole page using a viewBox derived from the
    page's own ``SceneInfo.paper_size``. That makes the displayed image map
    linearly to the page's coordinate system, so the interactive drawing
    overlay places strokes exactly where the write tool will, and blank pages
    render as a blank page instead of returning ``None``.

    Returns ``(png_bytes, (paper_w, paper_h))`` or ``None`` if the file is not
    v6 or rendering dependencies are unavailable (callers should fall back).
    """
    blocks = _v6_blocks(rm_file_path)
    if blocks is None:
        return None
    try:
        text_elements, _, anchor_pos = _v6_text_elements_with_bounds(blocks)
        paths, _ = _v6_paths_from_blocks(blocks, anchor_pos)
        paper_w, paper_h = _v6_paper_size(blocks)
    except Exception:
        return None

    # Typed text is drawn first so handwritten strokes layer on top of it,
    # matching the device's compositing order.
    svg = _svg_full_page(text_elements + paths, paper_w, paper_h)
    scale = FULL_PAGE_TARGET_LONG_EDGE / max(paper_w, paper_h)
    output_width = max(1, round(paper_w * scale))
    output_height = max(1, round(paper_h * scale))

    png = _svg_string_to_png(svg, output_width, output_height, background_color)
    if png is None:
        return None
    return png, (paper_w, paper_h)


def render_rm_file_to_svg(
    rm_file_path: Path, background_color: Optional[str] = None
) -> Optional[str]:
    """
    Render a .rm file to SVG string.

    Uses the rmscene renderers to convert .rm to SVG, optionally adding a background.

    Args:
        rm_file_path: Path to the .rm file
        background_color: Background color (e.g., "#FFFFFF", None for transparent).
                         Use REMARKABLE_BACKGROUND_COLOR for the standard paper color.

    Returns:
        SVG content as string, or None if rendering failed
    """
    import subprocess
    import tempfile

    tmp_svg_path = None

    try:
        # Create temp file for SVG output
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
            tmp_svg_path = Path(tmp_svg.name)

        # Convert .rm to SVG via the rmscene renderers
        if not _rm_to_svg(rm_file_path, tmp_svg_path):
            return None

        # Read SVG content
        svg_content = tmp_svg_path.read_text()

        # Add background rectangle if color specified
        if background_color:
            svg_content = _add_svg_background(svg_content, background_color)

        return svg_content

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    finally:
        if tmp_svg_path:
            tmp_svg_path.unlink(missing_ok=True)


def _add_svg_background(svg_content: str, background_color: str) -> str:
    """Add a background rectangle to an SVG.

    Inserts a rect element as the first child of the SVG to act as background.

    Args:
        svg_content: Original SVG content
        background_color: Background color (e.g., "#FFFFFF")

    Returns:
        SVG content with background added
    """
    import re

    # Find the opening <svg> tag and its attributes
    svg_match = re.search(r"(<svg[^>]*>)", svg_content, re.IGNORECASE)
    if not svg_match:
        return svg_content

    svg_tag = svg_match.group(1)

    # Extract viewBox or width/height for the background rect dimensions
    viewbox_match = re.search(r'viewBox="([^"]*)"', svg_tag)
    if viewbox_match:
        viewbox = viewbox_match.group(1)
        parts = viewbox.split()
        if len(parts) == 4:
            x, y, width, height = parts
            bg_rect = (
                f'<rect x="{x}" y="{y}" width="{width}" '
                f'height="{height}" fill="{background_color}"/>'
            )
        else:
            # Fallback to full page
            bg_rect = f'<rect x="0" y="0" width="100%" height="100%" fill="{background_color}"/>'
    else:
        # No viewBox, use 100% dimensions
        bg_rect = f'<rect x="0" y="0" width="100%" height="100%" fill="{background_color}"/>'

    # Insert background rect right after the opening svg tag
    insert_pos = svg_match.end()
    return svg_content[:insert_pos] + bg_rect + svg_content[insert_pos:]


def _get_ordered_rm_files(tmpdir_path: Path) -> List[Path]:
    """Extract and order .rm files from an extracted document directory.

    Reads the .content file to determine page order and returns .rm files
    sorted accordingly. Falls back to filesystem order if no page order found.

    Args:
        tmpdir_path: Path to the extracted document directory

    Returns:
        List of .rm file paths in correct page order
    """
    # Get page order from .content file
    page_order = []
    for content_file in tmpdir_path.glob("*.content"):
        try:
            data = json.loads(content_file.read_text())
            # New format: cPages.pages array
            if "cPages" in data and "pages" in data["cPages"]:
                page_order = [p["id"] for p in data["cPages"]["pages"]]
            # Fallback: pages array directly
            elif "pages" in data and isinstance(data["pages"], list):
                page_order = data["pages"]
        except Exception:
            # Ignore errors reading/parsing .content file; fallback to default page order
            pass
        break

    rm_files = list(tmpdir_path.glob("**/*.rm"))

    # Sort rm_files by page order if available
    if page_order:
        rm_by_id = {}
        for rm_file in rm_files:
            page_id = rm_file.stem
            rm_by_id[page_id] = rm_file

        ordered_rm_files = []
        for page_id in page_order:
            if page_id in rm_by_id:
                ordered_rm_files.append(rm_by_id[page_id])
        # Add any remaining files not in page order
        for rm_file in rm_files:
            if rm_file not in ordered_rm_files:
                ordered_rm_files.append(rm_file)
        return ordered_rm_files

    return rm_files


def _get_page_order(tmpdir_path: Path) -> List[str]:
    """Return the ordered list of page ids from the .content file.

    formatVersion 2 stores them under ``cPages.pages[].id``; formatVersion 1 as
    a flat ``pages`` list of ids. Returns an empty list if unavailable. This is
    the full page order (including pages that have no strokes), unlike
    ``_get_ordered_rm_files`` which is compacted to pages that do.
    """
    for content_file in tmpdir_path.glob("*.content"):
        try:
            data = json.loads(content_file.read_text())
            if "cPages" in data and "pages" in data["cPages"]:
                return [p.get("id") for p in data["cPages"]["pages"] if p.get("id")]
            if isinstance(data.get("pages"), list):
                return [p for p in data["pages"] if isinstance(p, str)]
        except Exception:
            pass
        break
    return []


def _select_rm_file_for_page(tmpdir_path: Path, rm_files: List[Path], page: int) -> Optional[Path]:
    """Return the .rm stroke file for a 1-based page, matched by page id.

    ``_get_ordered_rm_files`` is compacted (only pages that actually have
    strokes), so indexing it by page number pairs the wrong ink with a page
    whenever an earlier page is un-annotated. Map through the full ``.content``
    page order and match by id instead. Returns None when the page has no
    strokes (a page with a PDF underlay but no annotation), or when the page is
    out of range; callers should treat None as "no annotation layer".
    """
    page_order = _get_page_order(tmpdir_path)
    if page_order:
        if 1 <= page <= len(page_order):
            page_id = page_order[page - 1]
            return next((p for p in rm_files if p.stem == page_id), None)
        return None
    # No page order available (unusual): fall back to positional selection.
    if 1 <= page <= len(rm_files):
        return rm_files[page - 1]
    return None


def render_page_from_document_zip_svg(
    zip_path: Path, page: int = 1, background_color: Optional[str] = None
) -> Optional[str]:
    """
    Render a specific page from a reMarkable document zip to SVG.

    Args:
        zip_path: Path to the document zip file
        page: Page number (1-indexed)
        background_color: Background color (e.g., "#FFFFFF", None for transparent).
                         Use REMARKABLE_BACKGROUND_COLOR for the standard paper color.

    Returns:
        SVG content as string, or None if rendering failed or page doesn't exist
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)

        rm_files = _get_ordered_rm_files(tmpdir_path)

        # Select the page's .rm by id (see _select_rm_file_for_page); None means
        # the page has no annotation layer.
        target_rm_file = _select_rm_file_for_page(tmpdir_path, rm_files, page)
        if target_rm_file is None:
            return None

        return render_rm_file_to_svg(target_rm_file, background_color=background_color)


def render_page_from_document_zip(
    zip_path: Path, page: int = 1, background_color: Optional[str] = None
) -> Optional[bytes]:
    """
    Render a specific page from a reMarkable document zip to PNG.

    Args:
        zip_path: Path to the document zip file
        page: Page number (1-indexed)
        background_color: Background color (e.g., "#FFFFFF", None for transparent).
                         Use REMARKABLE_BACKGROUND_COLOR for the standard paper color.

    Returns:
        PNG image bytes, or None if rendering failed or page doesn't exist
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)

        rm_files = _get_ordered_rm_files(tmpdir_path)

        # Select the page's .rm by id (see _select_rm_file_for_page); None means
        # the page has no annotation layer.
        target_rm_file = _select_rm_file_for_page(tmpdir_path, rm_files, page)
        if target_rm_file is None:
            return None

        return render_rm_file_to_png(target_rm_file, background_color=background_color)


def _document_paper_size(rm_files: List[Path]) -> Tuple[float, float]:
    """Best-effort page extent for a document by scanning its .rm files.

    Used to render a blank page (one with no .rm of its own) at the same size
    as the rest of the document. Falls back to the standard reMarkable page.
    """
    for p in rm_files:
        blocks = _v6_blocks(p)
        if blocks:
            return _v6_paper_size(blocks)
    return float(REMARKABLE_WIDTH), float(REMARKABLE_HEIGHT)


def render_page_full_page_from_document_zip(
    zip_path: Path, page: int = 1, background_color: Optional[str] = None
) -> Optional[Tuple[bytes, Tuple[float, float]]]:
    """Full-page render of a page, addressed by cPages index.

    This is the render used by the interactive canvas. It differs from
    ``render_page_from_document_zip`` in two ways that matter for write-back:

    1. Pages are addressed by the ``.content`` cPages order — the SAME index
       the write tool (``_page_ids_from_content``) uses — so "page N" means the
       same page in the viewer and the writer even when blank pages (which may
       have no ``.rm`` file) are present.
    2. The page is rendered full-bleed using its own ``SceneInfo.paper_size``,
       so the overlay's normalized coordinates map exactly onto stroke space.

    Returns ``(png_bytes, (paper_w, paper_h))`` or ``None`` (caller falls back
    to PDF rasterization or an error).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)

        return render_page_full_page_from_extracted_document(
            tmpdir_path,
            page=page,
            background_color=background_color,
        )


def render_page_full_page_from_extracted_document(
    tmpdir_path: Path, page: int = 1, background_color: Optional[str] = None
) -> Optional[Tuple[bytes, Tuple[float, float]]]:
    """Render a full page from an already extracted document archive."""
    rm_files = _get_ordered_rm_files(tmpdir_path)
    page_order = _get_page_order(tmpdir_path)

    if page_order:
        if page < 1 or page > len(page_order):
            return None
        rm_file = _select_rm_file_for_page(tmpdir_path, rm_files, page)
    else:
        # No page metadata: fall back to filesystem order of .rm files.
        if page < 1 or page > len(rm_files):
            return None
        rm_file = rm_files[page - 1]

    if rm_file is not None and rm_file.exists():
        return render_rm_file_full_page_png(rm_file, background_color=background_color)

    # The page exists in cPages but has no .rm layer yet (a blank page):
    # render a blank full page at the document's paper size so the viewer
    # still shows it. (The write tool will return no_page_layer until the
    # page has a drawable layer / has been added via remarkable_add_page.)
    paper_w, paper_h = _document_paper_size(rm_files)
    svg = _svg_full_page([], paper_w, paper_h)
    scale = FULL_PAGE_TARGET_LONG_EDGE / max(paper_w, paper_h)
    png = _svg_string_to_png(
        svg, max(1, round(paper_w * scale)), max(1, round(paper_h * scale)), background_color
    )
    if png is None:
        return None
    return png, (paper_w, paper_h)


def document_zip_has_pdf_underlay(zip_path: Path) -> bool:
    """Check if a reMarkable document zip contains a PDF underlay.

    Args:
        zip_path: Path to the document zip file

    Returns:
        True if the zip contains a .pdf file
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return any(name.endswith(".pdf") for name in zf.namelist())
    except Exception:
        return False


def _read_cpages_entries(tmpdir_path: Path) -> List[Dict[str, Any]]:
    """Read cPages.pages entries from the .content metadata file.

    Args:
        tmpdir_path: Path to the extracted document directory

    Returns:
        List of cPages page entries, or empty list if not found
    """
    content_file = next(tmpdir_path.glob("*.content"), None)
    if content_file is None:
        return []
    try:
        data = json.loads(content_file.read_text())
        if "cPages" in data and "pages" in data["cPages"]:
            return data["cPages"]["pages"]
    except Exception:
        pass
    return []


def _pdf_page_index_for_cpages_entry(entry: Dict[str, Any]) -> Optional[int]:
    """Get the 0-based PDF page index from a cPages entry's redir field.

    The redir.value field in a cPages entry maps the reMarkable page to
    the original PDF page number (0-based).

    Args:
        entry: A single cPages page entry dict

    Returns:
        0-based PDF page index, or None if no redirect exists
    """
    redir = entry.get("redir", {})
    if isinstance(redir, dict) and "value" in redir:
        try:
            return int(redir["value"])
        except (ValueError, TypeError):
            return None
    return None


def _read_redirection_page_map(tmpdir_path: Path) -> List[Any]:
    """Read the legacy (formatVersion 1) redirectionPageMap from .content.

    formatVersion 1 documents have no cPages structure; instead the reMarkable
    page -> original PDF page mapping is a flat list where the index is the
    0-based reMarkable page and the value is the 0-based PDF page index (-1 for
    user-added pages). Many cloud-imported PDFs use this layout.

    Returns the list, or an empty list if not found.
    """
    content_file = next(tmpdir_path.glob("*.content"), None)
    if content_file is None:
        return []
    try:
        data = json.loads(content_file.read_text())
        rpm = data.get("redirectionPageMap")
        if isinstance(rpm, list):
            return rpm
    except Exception:
        pass
    return []


def _resolve_pdf_page_index(tmpdir_path: Path, page: int) -> Optional[int]:
    """Resolve the 0-based PDF page index for a 1-based reMarkable page.

    Handles both document layouts:
    - formatVersion 2: ``cPages.pages[i].redir.value``
    - formatVersion 1: ``redirectionPageMap[i]`` (legacy; used by many
      cloud-imported PDFs)

    Returns None only when the page genuinely has no PDF underlay (a user-added
    page). This lets render_merged composite annotations onto imported PDFs that
    use either layout, instead of falling back to an annotation-only render.

    When cPages entries exist they are authoritative: an entry without ``redir``
    is a user-added page, and we must NOT fall through to a redirectionPageMap a
    v1->v2 migrated document may still carry — its stale, order-shifted indices
    could composite the wrong PDF page under a user-added page.
    """
    # formatVersion 2 (cPages) — authoritative when present
    entries = _read_cpages_entries(tmpdir_path)
    if entries:
        if 1 <= page <= len(entries):
            return _pdf_page_index_for_cpages_entry(entries[page - 1])
        return None

    # formatVersion 1 (redirectionPageMap)
    rpm = _read_redirection_page_map(tmpdir_path)
    if rpm:
        if 1 <= page <= len(rpm):
            val = rpm[page - 1]
            if isinstance(val, int) and val >= 0:
                return val
        return None

    # Some formatVersion 1 documents (including files uploaded by this server)
    # contain a flat page-id list but omit redirectionPageMap. With no explicit
    # user-added-page markers, the source PDF order is the only authoritative
    # mapping and is therefore identity.
    page_order = _get_page_order(tmpdir_path)
    if page_order and 1 <= page <= len(page_order):
        return page - 1

    return None


def _render_pdf_page_to_png(
    pdf_bytes: bytes, page_index: int, width: int, height: int
) -> Optional[bytes]:
    """Rasterize a single PDF page to PNG bytes using PyMuPDF (fitz).

    Args:
        pdf_bytes: Raw PDF file bytes
        page_index: 0-based page index
        width: Target output width in pixels
        height: Target output height in pixels

    Returns:
        PNG image bytes, or None on failure
    """
    try:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            if page_index < 0 or page_index >= len(doc):
                return None

            pdf_page = doc[page_index]
            # Scale to fill the target dimensions
            mat = fitz.Matrix(width / pdf_page.rect.width, height / pdf_page.rect.height)
            pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        return None


def render_tablet_pdf_page_to_png(
    pdf_bytes: bytes, page: int = 1, target_long_edge: int = 2048
) -> Optional[bytes]:
    """Rasterize one page of the tablet's native PDF export to PNG bytes.

    This is the portable fallback used when the local stroke renderer cannot
    produce an image. The reMarkable's own firmware renders every notebook (and
    annotated PDF/EPUB) to a PDF served at ``/download/<uuid>/pdf``, so this path
    works regardless of the .rm block format, handles empty pages, and—crucially
    for portability—does not depend on a system graphics library (PyMuPDF
    bundles its own renderer). Credit: ljdutel (#95).

    Args:
        pdf_bytes: Raw bytes of the tablet-exported PDF.
        page: 1-based page number (matches notebook page ordering 1:1).
        target_long_edge: Target pixel size for the longest page edge.

    Returns:
        PNG image bytes, or None on failure / out-of-range page.
    """
    try:
        import fitz
    except ImportError:
        return None

    # The tablet's PDFs sometimes reference graphics-state resources MuPDF
    # considers malformed; it still rasterizes them correctly, so suppress the
    # noisy non-fatal error output.
    try:
        fitz.TOOLS.mupdf_display_errors(False)
    except Exception:
        pass

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None

    try:
        index = page - 1
        if index < 0 or index >= len(doc):
            return None
        pdf_page = doc[index]
        longest_edge = max(pdf_page.rect.width, pdf_page.rect.height) or 1.0
        zoom = target_long_edge / longest_edge
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")
    except Exception:
        return None
    finally:
        doc.close()


def render_mapped_pdf_page_from_document_zip(
    zip_path: Path, page: int = 1, target_long_edge: int = 2048
) -> Tuple[Optional[bytes], bool]:
    """Render a source-PDF underlay using the document's authoritative page map.

    Returns ``(png, has_source_pdf)``. A source PDF with ``png=None`` means the
    requested page has no mapped underlay (for example, a user-added page), so
    callers must not substitute the same ordinal from a native PDF export.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir_path)
        except Exception:
            return None, False

        pdf_files = list(tmpdir_path.glob("**/*.pdf"))
        if not pdf_files:
            return None, False

        pdf_page_index = _resolve_pdf_page_index(tmpdir_path, page)
        if pdf_page_index is None:
            return None, True

        content_stems = {path.stem for path in tmpdir_path.glob("*.content")}
        matching = [path for path in pdf_files if path.stem in content_stems]
        pdf_path = matching[0] if matching else sorted(pdf_files)[0]
        png = render_tablet_pdf_page_to_png(
            pdf_path.read_bytes(),
            page=pdf_page_index + 1,
            target_long_edge=target_long_edge,
        )
        return png, True


def render_merged_page_from_document_zip(
    zip_path: Path,
    page: int = 1,
    background_color: Optional[str] = None,
    canvas_width: Optional[int] = None,
    canvas_height: Optional[int] = None,
) -> tuple[Optional[bytes], Optional[str]]:
    """Render a page with the PDF underlay composited with the annotation layer.

    Extracts the zip, determines which PDF page corresponds to the requested
    reMarkable page, rasterizes the PDF page, renders the annotation layer,
    and alpha-composites them into a single image.

    Credit: Re-implementation inspired by PR #79 from @ColinSha.

    Args:
        zip_path: Path to the document zip file
        page: Page number (1-indexed)
        background_color: Background color for annotation layer
        canvas_width: Output canvas width (default: derived from PDF page)
        canvas_height: Output canvas height (default: derived from PDF page)

    Returns:
        Tuple of (png_bytes, note) where note is an informational message
        or None. Returns (None, error_note) on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir_path)
        except Exception:
            return None, "Could not extract document archive."

        return render_merged_page_from_extracted_document(
            tmpdir_path,
            page=page,
            background_color=background_color,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )


def render_merged_page_from_extracted_document(
    tmpdir_path: Path,
    page: int = 1,
    background_color: Optional[str] = None,
    canvas_width: Optional[int] = None,
    canvas_height: Optional[int] = None,
) -> tuple[Optional[bytes], Optional[str]]:
    """Render a merged page from an already extracted document archive."""
    import io

    import fitz
    from PIL import Image as PILImage

    # Find the PDF file in the extracted directory. If multiple are present
    # (rare), prefer the one whose stem matches the .content document id so
    # selection is deterministic.
    pdf_files = list(tmpdir_path.glob("**/*.pdf"))
    if not pdf_files:
        full_page = render_page_full_page_from_extracted_document(
            tmpdir_path,
            page=page,
            background_color=background_color,
        )
        if full_page is None:
            return None, f"Page {page} has no renderable annotation layer."
        return full_page[0], "No PDF underlay found; returned full-page annotation render."

    content_stems = {p.stem for p in tmpdir_path.glob("*.content")}
    matching = [p for p in pdf_files if p.stem in content_stems]
    pdf_path = matching[0] if matching else sorted(pdf_files)[0]
    pdf_bytes = pdf_path.read_bytes()

    rm_files = _get_ordered_rm_files(tmpdir_path)
    total_pages = len(_get_page_order(tmpdir_path)) or len(rm_files)
    if total_pages == 0:
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
                total_pages = len(pdf_document)
        except Exception:
            total_pages = 0

    if page < 1 or page > total_pages:
        return None, f"Page {page} out of range (document has {total_pages} pages)."

    # Select the .rm for this page by id (see _select_rm_file_for_page). A
    # page with no strokes yields None and composites to the bare PDF page.
    target_rm_file: Optional[Path] = _select_rm_file_for_page(tmpdir_path, rm_files, page)

    def annotation_only(reason: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Return a complete annotation page when the PDF side cannot be used."""
        full_page = render_page_full_page_from_extracted_document(
            tmpdir_path,
            page=page,
            background_color=background_color,
        )
        if full_page is None:
            return None, f"{reason}; page has no renderable annotation layer."
        return full_page[0], f"{reason}; returned full-page annotation render."

    # Determine which PDF page this reMarkable page maps to. Handles both the
    # formatVersion 2 (cPages.redir) and formatVersion 1 (redirectionPageMap)
    # layouts; the latter is used by many cloud-imported PDFs, which would
    # otherwise be misread as user-added pages and rendered annotation-only.
    pdf_page_index: Optional[int] = _resolve_pdf_page_index(tmpdir_path, page)

    if pdf_page_index is None:
        return annotation_only("Page has no PDF underlay (user-added page)")

    # Get PDF page dimensions to set annotation viewBox correctly
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            if pdf_page_index >= len(doc):
                return annotation_only("PDF page index out of range")

            pdf_page = doc[pdf_page_index]
            pdf_w_pt = pdf_page.rect.width
            pdf_h_pt = pdf_page.rect.height
        finally:
            doc.close()
    except Exception:
        return annotation_only("Could not read PDF dimensions")

    # Determine output canvas size
    out_w = canvas_width or int(pdf_w_pt * 2)
    out_h = canvas_height or int(pdf_h_pt * 2)

    # 1. Rasterize the PDF page
    pdf_png = _render_pdf_page_to_png(pdf_bytes, pdf_page_index, out_w, out_h)
    if pdf_png is None:
        return annotation_only("PDF rasterization failed")

    # 2. Render annotation layer to SVG, then to PNG with transparent background
    ann_svg_path = None
    ann_png_bytes = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
            ann_svg_path = Path(tmp_svg.name)

        if target_rm_file is not None and _rm_to_svg(target_rm_file, ann_svg_path):
            # rmscene coordinates use the device paper grid, not PDF points. Set
            # the root viewBox to the calibrated page rectangle before rasterizing.
            svg_content = ann_svg_path.read_text()

            viewbox = _pdf_ink_viewbox(pdf_w_pt, pdf_h_pt, _v6_blocks(target_rm_file))
            svg_content = _rewrite_svg_root(
                svg_content,
                viewbox=viewbox,
                width=out_w,
                height=out_h,
            )

            ann_png_bytes = _svg_string_to_png(
                svg_content,
                out_w,
                out_h,
                None,
            )
            if ann_png_bytes is None:
                raise RuntimeError("PyMuPDF could not rasterize the annotation SVG")
    except Exception as exc:
        # Annotation rendering failed; return the PDF page but expose the loss.
        logger.debug("Annotation overlay rendering failed: %s", exc)
        ann_render_error = exc
    else:
        ann_render_error = None
    finally:
        if ann_svg_path:
            ann_svg_path.unlink(missing_ok=True)

    # 3. Composite: PDF base + annotation overlay
    try:
        pdf_img = PILImage.open(io.BytesIO(pdf_png)).convert("RGBA")

        if ann_png_bytes:
            ann_img = PILImage.open(io.BytesIO(ann_png_bytes)).convert("RGBA")
            if ann_img.size != pdf_img.size:
                ann_img = ann_img.resize(pdf_img.size, PILImage.LANCZOS)
            composite = PILImage.alpha_composite(pdf_img, ann_img)
            merged_note = None
        else:
            composite = pdf_img
            merged_note = (
                "Annotation overlay failed to render; returned PDF page without annotations."
                if ann_render_error is not None
                else None
            )

        composite = composite.convert("RGB")
        buf = io.BytesIO()
        composite.save(buf, format="PNG")
        return buf.getvalue(), merged_note
    except Exception:
        return annotation_only("Compositing failed")


def get_page_ids_from_zip(zip_path: Path) -> List[Optional[str]]:
    """Stable page ids of a document zip, in page order (None where unknown).

    The ids come from the .content file (``cPages.pages[].id`` in format 2, the
    ``pages`` list in format 1). They survive pages being inserted or removed,
    unlike page numbers. The list is aligned with get_document_page_count.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            name = next((n for n in zf.namelist() if n.endswith(".content")), None)
            if name is None:
                return []
            data = json.loads(zf.read(name))
    except Exception:
        return []
    if isinstance(data.get("cPages"), dict) and isinstance(data["cPages"].get("pages"), list):
        return [p.get("id") if isinstance(p, dict) else None for p in data["cPages"]["pages"]]
    if isinstance(data.get("pages"), list):
        return [p if isinstance(p, str) else None for p in data["pages"]]
    return []


def get_document_page_count(zip_path: Path) -> int:
    """
    Get the number of pages in a reMarkable document zip.

    Uses the .content metadata file for accurate page count (includes
    user-added pages in PDFs). Falls back to counting .rm files.

    Args:
        zip_path: Path to the document zip file

    Returns:
        Number of pages (0 if unable to determine)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)

        # Try .content metadata first — it's the authoritative page list
        for content_file in tmpdir_path.glob("*.content"):
            try:
                data = json.loads(content_file.read_text())
                if "cPages" in data and "pages" in data["cPages"]:
                    return len(data["cPages"]["pages"])
                if "pages" in data and isinstance(data["pages"], list):
                    return len(data["pages"])
            except Exception:
                pass
            break

        # Fallback to counting .rm files
        return len(list(tmpdir_path.glob("**/*.rm")))


def get_document_file_type(zip_path: Path) -> str:
    """
    Read the ``fileType`` from a document zip's ``.content`` file.

    Returns one of "notebook", "pdf", "epub", or "" when it cannot be
    determined. Reads only the ``.content`` entry from the zip (no full
    extraction) so it is cheap to call alongside get_document_page_count.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".content"):
                    try:
                        data = json.loads(zf.read(name).decode("utf-8"))
                    except Exception:
                        return ""
                    return str(data.get("fileType", "") or "")
    except Exception:
        pass
    return ""


def _extract_page_annotations(rm_file: Path) -> Tuple[List[str], bool]:
    """Extract a page's text highlights and whether it has pen strokes.

    reMarkable text highlights are stored as ``GlyphRange`` scene items inside the
    page's v6 ``.rm`` file (each carries the selected ``text`` and its bounding
    ``rectangles``); freehand ink is stored as ``Line`` items. The legacy
    ``.highlights`` JSON scanned elsewhere does not cover current firmware, so
    read them here.

    Returns ``(highlighted_texts_in_reading_order, has_pen_strokes)``. Returns
    ``([], False)`` if the file cannot be parsed.
    """
    try:
        import rmscene
    except Exception:
        return [], False
    try:
        with rm_file.open("rb") as fh:
            tree = rmscene.read_tree(fh)
    except Exception:
        return [], False

    highlights: list = []
    has_strokes = False
    try:
        for item in tree.walk():
            kind = type(item).__name__
            if kind == "Line":
                has_strokes = True
            elif kind == "GlyphRange":
                text = getattr(item, "text", None)
                if not text:
                    continue
                rects = getattr(item, "rectangles", None) or []
                y = rects[0].y if rects else 0.0
                x = rects[0].x if rects else 0.0
                highlights.append((y, x, text))
    except Exception:
        pass

    highlights.sort(key=lambda t: (t[0], t[1]))  # top-to-bottom, left-to-right
    return [t[2] for t in highlights], has_strokes


def extract_text_from_document_zip(
    zip_path: Path, include_ocr: bool = False, doc_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract all text content from a reMarkable document zip.

    Args:
        zip_path: Path to the document zip file
        include_ocr: Whether to run OCR on handwritten content
        doc_id: Optional document ID for caching OCR results

    Returns:
        {
            "typed_text": [...],      # From rmscene parsing (list of strings)
            "highlights": [...],       # Highlighted text (GlyphRange + legacy JSON)
            "handwritten_text": [...], # From OCR (if enabled) - one per page, in order
            "pages": int,
            "page_ids": [...],         # Page UUIDs in order
            "annotated_pages": [       # Only pages that carry annotations
                {"page": int,          # 1-based page number
                 "page_id": str,
                 "has_handwriting": bool,   # page has pen strokes
                 "highlights": [str]}, # highlighted text on the page
            ],
            "ocr_backend": str,        # Which OCR backend was used (if any)
        }
    """
    cache_token = None
    if doc_id:
        with _cache_lock:
            cache_token = (
                _global_cache_generation,
                _cache_generation.get(doc_id, 0),
            )
            cached = _extraction_cache.get(doc_id)
            if cached and (cached["include_ocr"] or not include_ocr) and _is_cache_valid(cached):
                return cached["result"]

    result: Dict[str, Any] = {
        "typed_text": [],
        "highlights": [],
        "handwritten_text": None,
        "pages": 0,
        "page_ids": [],
        "annotated_pages": [],
        "ocr_backend": None,
        "tags": [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)

        # Get page order from .content file
        page_order = []
        for content_file in tmpdir_path.glob("*.content"):
            try:
                data = json.loads(content_file.read_text())
                # New format: cPages.pages array
                if "cPages" in data and "pages" in data["cPages"]:
                    page_order = [p["id"] for p in data["cPages"]["pages"]]
                # Fallback: pages array directly
                elif "pages" in data and isinstance(data["pages"], list):
                    page_order = data["pages"]
            except Exception:
                # Malformed .content file - continue without page order
                pass
            break  # Only process first .content file

        rm_files = list(tmpdir_path.glob("**/*.rm"))

        # If we have page order, sort rm_files accordingly
        if page_order:
            # Create mapping of page_id -> rm_file
            rm_by_id = {}
            for rm_file in rm_files:
                page_id = rm_file.stem  # filename without extension
                rm_by_id[page_id] = rm_file

            # Sort rm_files by page order
            ordered_rm_files = []
            for page_id in page_order:
                if page_id in rm_by_id:
                    ordered_rm_files.append(rm_by_id[page_id])
            # Add any remaining files not in page order
            for rm_file in rm_files:
                if rm_file not in ordered_rm_files:
                    ordered_rm_files.append(rm_file)
            rm_files = ordered_rm_files
            result["page_ids"] = page_order
        else:
            result["page_ids"] = [f.stem for f in rm_files]

        result["pages"] = len(page_order) or len(rm_files)

        # Extract typed text from .rm files using rmscene
        for rm_file in rm_files:
            text_lines = extract_text_from_rm_file(rm_file)
            result["typed_text"].extend(text_lines)

        # Extract per-page text highlights (GlyphRange items) and note-presence
        # from the .rm files, indexed by real page number. This powers "show only
        # the annotated pages / the highlighted text" without paging through the
        # whole document. (The legacy .highlights JSON scanned below only covers
        # older firmware exports.)
        page_pos = {pid: i for i, pid in enumerate(page_order)}
        for order_pos, rm_file in enumerate(rm_files):
            page_num = page_pos.get(rm_file.stem, order_pos) + 1
            page_highlights, has_strokes = _extract_page_annotations(rm_file)
            if page_highlights or has_strokes:
                result["annotated_pages"].append(
                    {
                        "page": page_num,
                        "page_id": rm_file.stem,
                        "has_handwriting": has_strokes,
                        "highlights": page_highlights,
                    }
                )
                result["highlights"].extend(page_highlights)
        result["annotated_pages"].sort(key=lambda a: a["page"])

        # Extract text from .txt and .md files
        for txt_file in tmpdir_path.glob("**/*.txt"):
            try:
                content = txt_file.read_text(errors="ignore")
                if content.strip():
                    result["typed_text"].append(content)
            except Exception:
                # File read failed - skip this file and continue
                pass

        for md_file in tmpdir_path.glob("**/*.md"):
            try:
                content = md_file.read_text(errors="ignore")
                if content.strip():
                    result["typed_text"].append(content)
            except Exception:
                # File read failed - skip this file and continue
                pass

        # Extract from .content files (metadata with text and tags)
        for content_file in tmpdir_path.glob("**/*.content"):
            try:
                data = json.loads(content_file.read_text())
                if "text" in data:
                    result["typed_text"].append(data["text"])
                if "tags" in data and data["tags"]:
                    result["tags"] = data["tags"]
            except Exception:
                # Malformed JSON or read error - skip this file
                pass

        # Extract PDF highlights
        for json_file in tmpdir_path.glob("**/*.json"):
            try:
                data = json.loads(json_file.read_text())
                if isinstance(data, dict) and "highlights" in data:
                    for h in data.get("highlights", []):
                        if "text" in h and h["text"]:
                            result["highlights"].append(h["text"])
            except Exception:
                # Malformed JSON - skip this file
                pass

        # OCR for handwritten content (optional)
        if include_ocr and rm_files:
            ocr_result, ocr_backend = extract_handwriting_ocr(rm_files)
            result["handwritten_text"] = ocr_result
            result["ocr_backend"] = ocr_backend

    if doc_id and cache_token is not None:
        _cache_extraction_result_if_current(doc_id, result, include_ocr, cache_token)

    return result


def extract_handwriting_ocr(rm_files: List[Path]) -> tuple[Optional[List[str]], Optional[str]]:
    """
    Extract handwritten text using OCR.

    Supports multiple backends (set REMARKABLE_OCR_BACKEND env var):
    - "google": Google Cloud Vision - best for handwriting
    - "tesseract": pytesseract - basic OCR, rasterized locally with PyMuPDF
    - "gemini" / "anthropic": multimodal LLM transcription, cached on disk
    - "auto" (default): Google if API key provided, else Tesseract

    Returns:
        Tuple of (ocr_results, backend_used) where backend_used is "google" or "tesseract"
    """
    backend = get_ocr_backend()

    # Auto-detect best available backend
    if backend == "auto":
        # Check for Google Vision API key first (simplest auth method)
        if os.environ.get("GOOGLE_VISION_API_KEY"):
            backend = "google"
        else:
            backend = "tesseract"

    if backend in LLM_OCR_BACKENDS:
        result = _ocr_llm(rm_files, backend)
        if result is not None:
            return (result or None, backend)
        logger.warning("%s OCR returned nothing; falling back", backend)
        backend = "google" if os.environ.get("GOOGLE_VISION_API_KEY") else "tesseract"

    if backend == "google":
        result = _ocr_google_vision(rm_files)
        if result:
            return (result, "google")
        result = _ocr_tesseract(rm_files)
        return (result, "tesseract")
    result = _ocr_tesseract(rm_files)
    return (result, "tesseract")


def _rm_to_png_bytes(rm_file: Path) -> Optional[bytes]:
    """Render one .rm page to a white-background PNG for OCR."""
    import tempfile

    tmp_svg_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
            tmp_svg_path = Path(tmp_svg.name)
        if not _rm_to_svg(rm_file, tmp_svg_path):
            return None
        return _svg_string_to_png(
            tmp_svg_path.read_text(),
            REMARKABLE_WIDTH,
            REMARKABLE_HEIGHT,
            "#FFFFFF",
        )
    except Exception:
        return None
    finally:
        if tmp_svg_path:
            tmp_svg_path.unlink(missing_ok=True)


def _ocr_llm(rm_files: List[Path], backend: str) -> Optional[List[str]]:
    """OCR each page with a multimodal LLM (results cached on disk)."""
    results = []
    answered = 0
    for rm_file in rm_files:
        png = _rm_to_png_bytes(rm_file)
        if png is None:
            continue
        text = llm_ocr.ocr_png(png, backend)
        if text is None:
            continue
        answered += 1
        if text:
            results.append(text)
    if answered == 0:
        return None  # backend unavailable: let the caller fall back
    return results


def _ocr_google_vision(rm_files: List[Path]) -> Optional[List[str]]:
    """
    OCR using Google Cloud Vision API.
    Best quality for handwriting recognition.

    Supports two authentication methods:
    1. GOOGLE_VISION_API_KEY env var (simplest - just an API key)
    2. GOOGLE_APPLICATION_CREDENTIALS or default credentials (service account)
    """
    import os

    api_key = os.environ.get("GOOGLE_VISION_API_KEY")

    if api_key:
        # Use REST API with API key (simpler, no SDK needed)
        return _ocr_google_vision_rest(rm_files, api_key)
    else:
        # Use SDK with service account credentials
        return _ocr_google_vision_sdk(rm_files)


def _ocr_google_vision_rest(rm_files: List[Path], api_key: str) -> Optional[List[str]]:
    """
    OCR using Google Cloud Vision REST API with API key.
    """
    import base64
    import tempfile

    import requests

    ocr_results = []

    for rm_file in rm_files:
        tmp_svg_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
                tmp_svg_path = Path(tmp_svg.name)

            # Convert .rm to SVG via the rmscene renderers
            if not _rm_to_svg(rm_file, tmp_svg_path):
                continue

            png = _svg_string_to_png(
                tmp_svg_path.read_text(),
                REMARKABLE_WIDTH,
                REMARKABLE_HEIGHT,
                "#FFFFFF",
            )
            if png is None:
                continue

            # Read and encode image
            image_content = base64.b64encode(png).decode("utf-8")

            # Call Google Vision REST API
            url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
            payload = {"requests": [vision_request(image_content)]}

            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if "responses" in data and data["responses"]:
                    resp = data["responses"][0]
                    if "fullTextAnnotation" in resp:
                        text = resp["fullTextAnnotation"]["text"]
                        if text.strip():
                            ocr_results.append(text.strip())
            elif response.status_code in (401, 403):
                # API key invalid or API not enabled - fall back to Tesseract
                return _ocr_tesseract(rm_files)

        except Exception:
            # API call or rendering failed - skip this page and continue
            pass
        finally:
            if tmp_svg_path:
                tmp_svg_path.unlink(missing_ok=True)

    return ocr_results if ocr_results else None


def _ocr_google_vision_sdk(rm_files: List[Path]) -> Optional[List[str]]:
    """
    OCR using Google Cloud Vision SDK with service account credentials.
    """
    try:
        import tempfile

        from google.cloud import vision

        client = vision.ImageAnnotatorClient()
        ocr_results = []

        for rm_file in rm_files:
            tmp_svg_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
                    tmp_svg_path = Path(tmp_svg.name)

                # Convert .rm to SVG via the rmscene renderers
                if not _rm_to_svg(rm_file, tmp_svg_path):
                    continue

                png = _svg_string_to_png(
                    tmp_svg_path.read_text(),
                    REMARKABLE_WIDTH,
                    REMARKABLE_HEIGHT,
                    "#FFFFFF",
                )
                if png is None:
                    continue

                # Send to Google Vision API
                image = vision.Image(content=png)

                # Use DOCUMENT_TEXT_DETECTION for best handwriting results
                hints = get_ocr_language_hints()
                if hints:
                    response = client.document_text_detection(
                        image=image, image_context={"language_hints": hints}
                    )
                else:
                    response = client.document_text_detection(image=image)

                if response.error.message:
                    continue

                if response.full_text_annotation.text:
                    ocr_results.append(response.full_text_annotation.text.strip())

            except Exception:
                # Rendering or API error - skip this page and continue
                pass
            finally:
                if tmp_svg_path:
                    tmp_svg_path.unlink(missing_ok=True)

        return ocr_results if ocr_results else None

    except ImportError:
        # google-cloud-vision not installed, fall back to tesseract
        return _ocr_tesseract(rm_files)
    except Exception:
        # API error, fall back to tesseract
        return _ocr_tesseract(rm_files)


def _ocr_tesseract(rm_files: List[Path]) -> Optional[List[str]]:
    """
    OCR using Tesseract.
    Basic quality - designed for printed text, not handwriting.

    Requires: pytesseract and PyMuPDF.
    """
    try:
        import io
        import tempfile

        import pytesseract
        from PIL import Image, ImageFilter, ImageOps

        ocr_results = []

        for rm_file in rm_files:
            tmp_svg_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
                    tmp_svg_path = Path(tmp_svg.name)

                # Convert .rm to SVG via the rmscene renderers
                if not _rm_to_svg(rm_file, tmp_svg_path):
                    continue

                # Convert SVG to PNG with higher resolution for better OCR
                png = _svg_string_to_png(
                    tmp_svg_path.read_text(),
                    2106,
                    2808,
                    "#FFFFFF",
                )
                if png is None:
                    continue

                # Preprocess image for better OCR
                img = Image.open(io.BytesIO(png))

                # Convert to grayscale
                img = img.convert("L")

                # Increase contrast
                img = ImageOps.autocontrast(img, cutoff=2)

                # Slight sharpening
                img = img.filter(ImageFilter.SHARPEN)

                # Run OCR with optimized settings for sparse handwriting
                # PSM 11 = Sparse text - find as much text as possible
                # PSM 6 = Uniform block of text (alternative)
                custom_config = r"--psm 11 --oem 3"
                text = pytesseract.image_to_string(img, config=custom_config)

                if text.strip():
                    ocr_results.append(text.strip())

            except Exception:
                # Rendering or OCR error - skip this page and continue
                pass
            finally:
                if tmp_svg_path:
                    tmp_svg_path.unlink(missing_ok=True)

        return ocr_results if ocr_results else None

    except ImportError:
        # OCR dependencies not installed
        return None


# ---------------------------------------------------------------------------
# Pen marks on PDF pages: printed text under the ink, handwriting apart from it
# ---------------------------------------------------------------------------

# A word counts as marked when a pen line or highlight crosses at least this
# share of its width inside the band where a strike-through or an underline of
# that word lies: from MARK_BAND_TOP below the top of the word box to
# MARK_BAND_BELOW below its bottom (fractions of the word height). The band
# stays clear of the lines above and below, so a mark on one line does not mark
# its neighbours. A word inside a drawn loop (circled) is marked too.
MARK_COVERAGE = 0.6
MARK_BAND_TOP = 0.2
MARK_BAND_BELOW = 0.25
_INK_ALPHA = 40
_RESIDUAL_MIN_PIXELS = 120


def _column_coverage(mask, box) -> float:
    """Share of columns in ``box`` (of a 0/255 mask) that contain any ink."""
    from PIL import Image

    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(mask.width, x1), min(mask.height, y1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return 0.0
    band = mask.crop((x0, y0, x1, y1))
    # Average each column down to one pixel: a non-zero average means the
    # column holds ink (word bands are short enough that one inked pixel
    # never rounds down to zero).
    columns = band.resize((x1 - x0, 1), Image.Resampling.BOX)
    inked = sum(1 for v in columns.tobytes() if v > 0)
    return inked / (x1 - x0)


def _word_mark(mask, word, scale, loops, highlights) -> Optional[str]:
    """How the pen marked one printed word, or None."""
    x0, y0, x1, y1 = word[:4]
    height = (y1 - y0) * scale
    left, right = x0 * scale, x1 * scale
    top, bottom = y0 * scale, y1 * scale
    band = (left, top + MARK_BAND_TOP * height, right, bottom + MARK_BAND_BELOW * height)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    if any(lx0 <= cx <= lx1 and ly0 <= cy <= ly1 for lx0, ly0, lx1, ly1 in loops):
        return "circle"
    if highlights is not None and _column_coverage(highlights, band) >= MARK_COVERAGE:
        return "highlight"
    if _column_coverage(mask, band) >= MARK_COVERAGE:
        # through the letters or under the baseline (at about 70 % of the box)
        split = top + 0.7 * height
        through = _column_coverage(mask, (left, band[1], right, split))
        under = _column_coverage(mask, (left, split, right, band[3]))
        return "strike" if through >= under else "underline"
    return None


# Hyphen, soft hyphen and Unicode hyphen at a line break
_HYPHENS = ("-", "\u00ad", "\u2010")
# Unmarked words between two marks of one sentence that are filled in, and
# how far a sentence may reach around its marks before it is cut to a window.
MARK_GAP_FILL = 2
CONTEXT_MAX_WORDS = 60
CONTEXT_WINDOW = 15
_NOT_SENTENCE_END = re.compile(
    r"^(e\.g|i\.e|et al|cf|vs|etc|p|pp|fig|figs|no|vol|ed|eds|dr|mr|ms|st|ca|approx|[a-z])\.$",
    re.IGNORECASE,
)


def _ends_sentence(token: str, following: Optional[str]) -> bool:
    bare = token.rstrip(")]}\"'’”0-9,–-")
    if not bare.endswith((".", "?", "!")) or _NOT_SENTENCE_END.match(bare.lstrip("([{\"'‘“")):
        return False
    return following is None or not following[:1].islower()


def find_marked_words(
    mask,
    words: list,
    scale: float,
    loops: Optional[list] = None,
    highlights=None,
    with_boxes: bool = False,
) -> list:
    """The marked passages of a page, each in the sentence around it.

    ``mask`` is an 'L' image (255 = ink) of the pen lines and ``highlights``
    one of the highlighter bands, aligned with the PDF page at ``scale`` pixels
    per point; ``words`` are PyMuPDF word tuples (x0, y0, x1, y1, text, block,
    line, word) in reading order; ``loops`` are pixel boxes of rings drawn
    around words. Returns ``[{"text", "marked", "kind"}]``: the sentence with
    the marked words in **bold**, the marked words alone, and the mark most of
    them carry ("underline", "strike", "highlight" or "circle"). Gaps of up to
    MARK_GAP_FILL unmarked words inside a sentence count as marked.
    """
    import unicodedata

    loops = loops or []
    # Words hyphenated across a line break become one token first.
    tokens: list = []  # [text, block, kind, boxes]
    for word in words:
        text = unicodedata.normalize("NFKC", word[4])
        block = word[5] if len(word) > 5 else 0
        kind = _word_mark(mask, word, scale, loops, highlights)
        if (
            tokens
            and tokens[-1][0].endswith(_HYPHENS)
            and tokens[-1][1] == block
            and text[:1].islower()
            and len(tokens[-1][0]) > 1
        ):
            tokens[-1][0] = tokens[-1][0][:-1] + text
            tokens[-1][2] = tokens[-1][2] or kind
            tokens[-1][3].append(tuple(word[:4]))
            continue
        tokens.append([text, block, kind, [tuple(word[:4])]])
    if not any(t[2] for t in tokens):
        return []

    # Sentences: split at sentence ends and at text-block changes.
    sentences, start = [], 0
    for i, token in enumerate(tokens):
        following = tokens[i + 1] if i + 1 < len(tokens) else None
        if following is None or following[1] != token[1] or _ends_sentence(token[0], following[0]):
            sentences.append((start, i + 1))
            start = i + 1

    passages = []
    for first, last in sentences:
        marked = [i for i in range(first, last) if tokens[i][2]]
        if len(marked) == 1 and len(tokens[marked[0]][0].strip(".,;:()")) <= 3:
            marked = []  # a stray touch on one short word ("can", "to")
        if not marked:
            continue
        kinds = [tokens[i][2] for i in marked]
        flags = {i: bool(tokens[i][2]) for i in range(first, last)}
        for a, b in zip(marked, marked[1:]):
            if 1 < b - a <= MARK_GAP_FILL + 1:
                for i in range(a + 1, b):
                    flags[i] = True
        lo, hi = first, last
        if hi - lo > CONTEXT_MAX_WORDS:
            lo = max(first, marked[0] - CONTEXT_WINDOW)
            hi = min(last, marked[-1] + CONTEXT_WINDOW + 1)
        parts, run, marked_text = [], [], []
        for i in range(lo, hi):
            if flags[i]:
                run.append(tokens[i][0])
                continue
            if run:
                parts.append("**" + " ".join(run) + "**")
                marked_text.append(" ".join(run))
                run = []
            parts.append(tokens[i][0])
        if run:
            parts.append("**" + " ".join(run) + "**")
            marked_text.append(" ".join(run))
        text = " ".join(parts)
        if lo > first:
            text = "… " + text
        if hi < last:
            text += " …"
        passage = {
            "text": text,
            "marked": " … ".join(marked_text),
            "kind": max(set(kinds), key=kinds.count),
        }
        if with_boxes:  # PDF-point boxes of the marked words (to name the pen)
            passage["_boxes"] = [b for i in range(lo, hi) if flags[i] for b in tokens[i][3]]
        passages.append(passage)
    return passages


def _ink_components(mask) -> list:
    """Connected pieces of ink (8-connected) in a 0/255 'L' mask.

    Each piece is a dict with its pixel indices, its bounding box, the top and
    bottom row of ink in each of its columns and the left and right end of ink
    in each of its rows.
    """
    width, height = mask.size
    data = mask.tobytes()
    seen = bytearray(len(data))
    pieces = []
    pos = data.find(b"\xff")
    while pos != -1:
        if not seen[pos]:
            seen[pos] = 1
            stack = [pos]
            pixels = []
            columns: Dict[int, list] = {}
            rows: Dict[int, list] = {}
            while stack:
                i = stack.pop()
                pixels.append(i)
                y, x = divmod(i, width)
                span = columns.get(x)
                if span is None:
                    columns[x] = [y, y]
                elif y < span[0]:
                    span[0] = y
                elif y > span[1]:
                    span[1] = y
                span = rows.get(y)
                if span is None:
                    rows[y] = [x, x]
                elif x < span[0]:
                    span[0] = x
                elif x > span[1]:
                    span[1] = x
                for ny in (y - 1, y, y + 1):
                    if ny < 0 or ny >= height:
                        continue
                    row = ny * width
                    for nx in (x - 1, x, x + 1):
                        if 0 <= nx < width:
                            j = row + nx
                            if data[j] and not seen[j]:
                                seen[j] = 1
                                stack.append(j)
            spans = columns.values()
            pieces.append(
                {
                    "pixels": pixels,
                    "columns": columns,
                    "rows": rows,
                    "box": (
                        min(columns),
                        min(s[0] for s in spans),
                        max(columns),
                        max(s[1] for s in spans),
                    ),
                }
            )
        pos = data.find(b"\xff", pos + 1)
    return pieces


def _classify_ink(piece: dict, word_height: float, scale: float, width: int) -> Optional[str]:
    """Tell a pen mark on the printed text from handwriting.

    Returns "line" (an underline or strike-through: long, roughly level and
    thin in every column), "highlight" (a long solid band), "loop" (a ring drawn around
    text) or None for handwriting. ``width`` is the mask width in pixels.
    """
    x0, y0, x1, y1 = piece["box"]
    w, h = x1 - x0 + 1, y1 - y0 + 1
    count = len(piece["pixels"])
    if w < 1.5 * word_height:
        return None
    spans = [bottom - top + 1 for top, bottom in piece["columns"].values()]
    thin = max(3 * scale, 0.35 * word_height)
    if h <= 1.2 * word_height and sum(1 for s in spans if s <= thin) >= 0.85 * len(spans):
        return "line"
    fill = count / (w * h)
    if h <= 1.6 * word_height and fill >= 0.6:
        return "highlight"
    tall = sum(1 for s in spans if s >= 0.5 * h)
    if h >= 0.8 * word_height and fill < 0.25 and tall >= 0.6 * len(spans):
        band_x, band_y = max(2.0, 0.2 * w), max(2.0, 0.2 * h)
        near_edge = 0
        for i in piece["pixels"]:
            y, x = divmod(i, width)
            if x < x0 + band_x or x > x1 - band_x or y < y0 + band_y or y > y1 - band_y:
                near_edge += 1
        if near_edge >= 0.85 * count:
            return "loop"
    return None


# Classifying ink is pure Python; keep it to a few pages at a time so the
# server's event loop stays responsive while a large PDF is transcribed.
_INK_ANALYSIS_SLOTS = threading.Semaphore(2)
# Ink is classified on a mask this many times smaller than the rendered page
# (1 = full resolution: slower on dense pages, but no two strokes merge).
_ANALYSIS_REDUCE = int(os.environ.get("REMARKABLE_INK_ANALYSIS_REDUCE", "1") or 1)


def _encircles_words(piece: dict, words: list, scale: float, width: int) -> bool:
    """Whether a piece of ink runs around a printed word on all four sides.

    A ring drawn in one stroke often crosses the words it circles, so its
    shape is no guide; what counts is ink of the same piece above, below,
    left and right of a word's centre. Handwriting written over a word also
    surrounds it, but most of its ink lies on the word itself, so a piece with
    much of its ink inside the word's box is not a ring.
    """
    px0, py0, px1, py1 = piece["box"]
    count = len(piece["pixels"])
    w, h = px1 - px0 + 1, py1 - py0 + 1
    if count / (w * h) >= 0.25:
        return False
    for word in words:
        x0, y0, x1, y1 = (v * scale for v in word[:4])
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        if not (px0 <= x0 + 2 and x1 - 2 <= px1 and py0 <= y0 + 2 and y1 - 2 <= py1):
            continue
        column, row = piece["columns"].get(cx), piece["rows"].get(cy)
        if not column or not row:
            continue
        reach = 0.3 * (y1 - y0)
        if not (column[0] <= cy - reach and column[1] >= cy + reach):
            continue
        if not (row[0] <= x0 + 2 and row[1] >= x1 - 2):
            continue
        inside = 0
        for i in piece["pixels"]:
            y, x = divmod(i, width)
            if x0 <= x <= x1 and y0 <= y <= y1:
                inside += 1
        if inside <= 0.45 * count:
            return True
    return False


_PEN_NAMES = {
    0: "paintbrush",
    12: "paintbrush",
    1: "pencil",
    14: "pencil",
    2: "ballpoint",
    15: "ballpoint",
    3: "marker",
    16: "marker",
    4: "fineliner",
    17: "fineliner",
    5: "highlighter",
    18: "highlighter",
    7: "mechanical pencil",
    13: "mechanical pencil",
    21: "calligraphy pen",
    23: "shader",
}
_COLOR_NAMES = {
    0: "black",
    1: "gray",
    2: "white",
    3: "yellow",
    4: "green",
    5: "pink",
    6: "blue",
    7: "red",
    8: "gray",
    9: "yellow",
    10: "green",
    11: "cyan",
    12: "magenta",
    13: "yellow",
}
_PATH_RE = re.compile(
    r'<path d="([^"]+)"[^>]*?stroke-width="([\d.]+)"[^>]*?data-tool="(\d+)" data-color="(\d+)"'
)


def _svg_strokes(svg_text: str, viewbox, pdf_w: float) -> list:
    """Pen strokes of a page as (box in PDF points, tool, color, width in mm)."""
    vx, vy, vw, _vh = viewbox
    per_unit = pdf_w / vw  # PDF points per stroke unit
    strokes = []
    for d, width, tool, color in _PATH_RE.findall(svg_text):
        numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", d)]
        xs, ys = numbers[0::2], numbers[1::2]
        if not xs or not ys:
            continue
        box = (
            (min(xs) - vx) * per_unit,
            (min(ys) - vy) * per_unit,
            (max(xs) - vx) * per_unit,
            (max(ys) - vy) * per_unit,
        )
        strokes.append((box, int(tool), int(color), float(width) * per_unit * 25.4 / 72))
    return strokes


def _pen_for_boxes(strokes: list, boxes: list) -> Optional[dict]:
    """The pen that drew the marks on these word boxes: tool, colour, width."""
    from collections import Counter

    votes: Counter = Counter()
    widths = []
    for (sx0, sy0, sx1, sy1), tool, color, width in strokes:
        for x0, y0, x1, y1 in boxes:
            pad = 0.6 * (y1 - y0)
            if sx0 <= x1 and sx1 >= x0 and sy0 <= y1 + pad and sy1 >= y0 - pad:
                votes[(tool, color)] += 1
                widths.append(width)
                break
    if not votes:
        return None
    tool, color = votes.most_common(1)[0][0]
    widths.sort()
    return {
        "tool": _PEN_NAMES.get(tool, f"pen {tool}"),
        "color": _COLOR_NAMES.get(color, f"colour {color}"),
        "width_mm": round(widths[len(widths) // 2], 2),
    }


def pen_label(pen: Optional[dict]) -> str:
    """'calligraphy pen, black, 0.3 mm' for a pen dict, '' for none."""
    if not pen:
        return ""
    width = pen.get("width_mm")
    return ", ".join(
        str(x) for x in (pen.get("tool"), pen.get("color"), f"{width} mm" if width else None) if x
    )


def _table_blocks(words: list) -> list:
    """Boxes of table-like regions: runs of three or more rows split into columns.

    Words are grouped into rows by their vertical position (across text
    blocks, since table cells are often blocks of their own); a row is split
    when two or more gaps between its words are wider than twice its height.
    """
    rows: list = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        mid, height = (w[1] + w[3]) / 2, w[3] - w[1]
        if rows and abs(rows[-1]["mid"] - mid) < 0.4 * height:
            rows[-1]["words"].append(w)
        else:
            rows.append({"mid": mid, "words": [w]})
    regions, run = [], []

    def close():
        if len(run) >= 3:
            ws = [w for row in run for w in row]
            regions.append(
                (
                    min(w[0] for w in ws),
                    min(w[1] for w in ws),
                    max(w[2] for w in ws),
                    max(w[3] for w in ws),
                )
            )

    for row in rows:
        ws = sorted(row["words"], key=lambda w: w[0])
        height = max(w[3] - w[1] for w in ws)
        gaps = sum(1 for a, b in zip(ws, ws[1:]) if b[0] - a[2] > 2 * height)
        if gaps >= 2:
            run.append(ws)
        else:
            close()
            run = []
    close()
    return regions


def _in_any(box, regions) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in regions)


def split_pen_marks(mask, words: list, scale: float) -> Tuple[list, Any]:
    """Split the ink on a PDF page into marks on the printed text and handwriting.

    ``mask`` is the page's ink as an 'L' image (255 = ink) at ``scale`` pixels
    per point; ``words`` are the PDF's word tuples. Each connected piece of ink
    is classified as a whole: lines, highlights and rings around printed words
    mark text, everything else is handwriting, even where it is written over
    the print. Returns the marked phrases (see find_marked_words) and a mask
    of the handwriting at full size.
    """
    from PIL import Image, ImageChops

    if not words:
        return [], mask
    # Work on a smaller copy: a pixel there is ink if any pixel it covers is.
    reduce = _ANALYSIS_REDUCE if min(mask.size) >= 64 else 1
    small_size = (max(1, mask.size[0] // reduce), max(1, mask.size[1] // reduce))
    small = mask.resize(small_size, Image.Resampling.BOX).point(lambda v: 255 if v else 0)
    small_scale = scale * small_size[0] / mask.size[0]
    heights = sorted((w[3] - w[1]) * small_scale for w in words)
    word_height = heights[len(heights) // 2]
    marks = bytearray(small_size[0] * small_size[1])
    bands = bytearray(len(marks))
    removed = bytearray(len(marks))
    loops = []
    with _INK_ANALYSIS_SLOTS:
        pieces = _ink_components(small)
        for piece in pieces:
            kind = _classify_ink(piece, word_height, small_scale, small_size[0])
            if kind is None and _encircles_words(piece, words, small_scale, small_size[0]):
                kind = "loop"  # drawn around words, whatever its shape
            if kind is None:
                continue  # handwriting stays in the residual image
            if kind == "loop":
                lx0, ly0, lx1, ly1 = piece["box"]
                if not any(
                    lx0 <= (w[0] + w[2]) / 2 * small_scale <= lx1
                    and ly0 <= (w[1] + w[3]) / 2 * small_scale <= ly1
                    for w in words
                ):
                    continue  # a ring around no printed word: handwriting
                loops.append(piece["box"])
            else:
                target = bands if kind == "highlight" else marks
                for i in piece["pixels"]:
                    target[i] = 255
            for i in piece["pixels"]:
                removed[i] = 255
    phrases = find_marked_words(
        Image.frombytes("L", small_size, bytes(marks)),
        words,
        small_scale,
        loops,
        Image.frombytes("L", small_size, bytes(bands)),
        with_boxes=True,
    )
    # Back to full size: drop every full-size pixel whose small pixel was a mark
    drop = Image.frombytes("L", small_size, bytes(removed)).resize(
        mask.size, Image.Resampling.NEAREST
    )
    return phrases, ImageChops.subtract(mask, drop)


def analyze_pdf_page_annotations(
    source: Path, page: int = 1, scale: float = 2.0
) -> Optional[Dict[str, Any]]:
    """Separate pen marks on a PDF page into marked printed text and handwriting.

    Returns None for pages without a PDF underlay (notebook pages, pages added
    to a PDF), so callers can treat them as handwriting. Otherwise returns
    ``{"marked": [phrases], "residual_png": png | None, "text_layer": bool}``:
    the printed words the pen underlined, circled, struck or highlighted (read
    from the PDF's text layer, no OCR needed) and an image of the remaining ink
    (margin notes and other handwriting), or None when nothing else is written.
    Without a text layer (a scanned PDF) all ink is returned as residual.

    ``source`` is the document zip or a directory it was extracted to; pass the
    directory when analysing many pages of a large document.
    """
    source = Path(source)
    if source.is_dir():
        return _analyze_extracted(source, page, scale)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(source, "r") as zf:
                zf.extractall(tmpdir)
        except Exception:
            return None
        return _analyze_extracted(Path(tmpdir), page, scale)


def document_has_pdf(zip_path: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return any(name.lower().endswith(".pdf") for name in zf.namelist())
    except Exception:
        return False


def _analyze_extracted(tmpdir_path: Path, page: int, scale: float) -> Optional[Dict[str, Any]]:
    import io

    import fitz
    from PIL import Image

    pdf_files = list(tmpdir_path.glob("**/*.pdf"))
    if not pdf_files:
        return None
    pdf_page_index = _resolve_pdf_page_index(tmpdir_path, page)
    if pdf_page_index is None:
        return None
    content_stems = {p.stem for p in tmpdir_path.glob("*.content")}
    matching = [p for p in pdf_files if p.stem in content_stems]
    pdf_path = matching[0] if matching else sorted(pdf_files)[0]
    rm_file = _select_rm_file_for_page(tmpdir_path, _get_ordered_rm_files(tmpdir_path), page)

    try:
        with fitz.open(pdf_path) as doc:
            if pdf_page_index >= len(doc):
                return None
            pdf_page = doc[pdf_page_index]
            pdf_w, pdf_h = pdf_page.rect.width, pdf_page.rect.height
            # Native order follows the text blocks, so the columns of a
            # two-column page stay apart (sort=True would interleave them).
            words = sorted(pdf_page.get_text("words"), key=lambda w: w[5:8])
    except Exception:
        return None

    out_w, out_h = max(1, int(pdf_w * scale)), max(1, int(pdf_h * scale))
    result: Dict[str, Any] = {"marked": [], "residual_png": None, "text_layer": bool(words)}
    if rm_file is None:
        return result

    with tempfile.TemporaryDirectory() as svg_dir:
        svg_path = Path(svg_dir) / "ink.svg"
        if not _rm_to_svg(rm_file, svg_path):
            return result
        svg_text = svg_path.read_text()
    viewbox = _pdf_ink_viewbox(pdf_w, pdf_h, _v6_blocks(rm_file))
    svg = _rewrite_svg_root(svg_text, viewbox=viewbox, width=out_w, height=out_h)
    ink_png = _svg_string_to_png(svg, out_w, out_h, None)
    if ink_png is None:
        return result

    ink = Image.open(io.BytesIO(ink_png)).convert("RGBA").getchannel("A")
    mask = ink.point(lambda v: 255 if v > _INK_ALPHA else 0)
    marked, residual = split_pen_marks(mask, words, scale)
    strokes = _svg_strokes(svg_text, viewbox, pdf_w)
    tables = _table_blocks(words)
    for passage in marked:
        boxes = passage.pop("_boxes", [])
        pen = _pen_for_boxes(strokes, boxes)
        if pen:
            passage["pen"] = pen
        if any(_in_any(box, tables) for box in boxes):
            passage["table"] = True
    result["marked"] = marked
    if residual.histogram()[255] >= _RESIDUAL_MIN_PIXELS * scale:
        buf = io.BytesIO()
        residual.point(lambda v: 0 if v else 255).save(buf, format="PNG")  # black ink on white
        result["residual_png"] = buf.getvalue()
    return result


def _normalize_page_text(text: str) -> str:
    """PDF text as running prose: ligatures undone, line-break hyphens joined."""
    import unicodedata

    text = unicodedata.normalize("NFKC", text).replace("­\n", "").replace("­", "")
    text = re.sub(r"(\w)[-‐]\n(?=[a-z])", r"\1", text)
    return " ".join(text.split())


def pdf_page_annotation_inputs(extracted: Path, page: int, max_side: int = 1568) -> Optional[dict]:
    """The page image with its ink and the printed text blocks, for composing annotations.

    Returns {"png", "blocks"} or None when the page has no PDF underlay.
    """
    import io

    import fitz
    from PIL import Image

    pdf_index = _resolve_pdf_page_index(extracted, page)
    pdf_files = list(extracted.glob("**/*.pdf"))
    if pdf_index is None or not pdf_files:
        return None
    stems = {p.stem for p in extracted.glob("*.content")}
    matching = [p for p in pdf_files if p.stem in stems]
    pdf_path = matching[0] if matching else sorted(pdf_files)[0]
    try:
        with fitz.open(pdf_path) as doc:
            if pdf_index >= len(doc):
                return None
            raw_blocks = doc[pdf_index].get_text("blocks")
    except Exception:
        return None
    blocks = [_normalize_page_text(b[4]) for b in raw_blocks if b[6] == 0 and b[4].strip()]
    png, _ = render_merged_page_from_extracted_document(extracted, page=page)
    if png is None:
        return None
    image = Image.open(io.BytesIO(png))
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        png = buf.getvalue()
    return {"png": png, "blocks": blocks}


def _letters(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKC", text).replace("­", "")
    return re.sub(r"[\W_]+", "", text).lower()


# A reader's description of a drawing ("[curved vertical line]"); not "[?]".
_INK_DESCRIPTION_RE = re.compile(r"\[(?!\?)[^\]]*\]")


def _clean_note(note) -> str:
    """Handwriting only: drawing descriptions out, the writer's line breaks joined."""
    if not isinstance(note, str):
        return ""
    lines = [line.strip() for line in _INK_DESCRIPTION_RE.sub("", note).splitlines()]
    return " ".join(line for line in lines if line)


def _clean_symbols(symbols) -> list:
    """The reader's own marks (?, !, ✓, ...); words such as "circle" are not symbols."""
    out = []
    for sym in symbols or []:
        sym = str(sym).strip()
        if sym and len(sym) <= 3 and not re.search(r"\w", sym) and sym not in out:
            out.append(sym)
    return out


def _row_key(row) -> str:
    return _letters(str(row[0]).split("✎")[0].replace("**", "")) if row else ""


def _merge_tables(tables: list, page: str) -> dict:
    """One table from pieces with the same caption.

    Rows keep the order the composer gave them (a table's text layer is often
    not in row order); pieces are joined with … between them and rows that
    appear in more than one piece are kept once.
    """
    first = dict(tables[0])
    if len(tables) > 1:
        merged, seen = [], set()
        for table in tables:
            rows = list(table.get("rows") or [])
            if merged and merged[-1] != "…" and rows and rows[0] != "…":
                merged.append("…")
            for row in rows:
                if isinstance(row, list):
                    key = _row_key(row)
                    if key in seen:
                        continue
                    seen.add(key)
                elif merged and merged[-1] == "…":
                    continue
                merged.append(row)
        first["rows"] = merged
    notes = [_clean_note(t.get("note")) for t in tables]
    first["note"] = "; ".join(n for n in notes if n)
    first["symbols"] = _clean_symbols(s for t in tables for s in t.get("symbols") or [])
    return first


def _words(text: str) -> set:
    import unicodedata

    return set(re.findall(r"\w\w+", unicodedata.normalize("NFKC", text or "").lower()))


def checked_annotations(
    data: Optional[dict], blocks: list, handwriting: Optional[str] = None
) -> Optional[dict]:
    """Keep only annotation items whose printed text is really on the page.

    A quote or table cell that cannot be found in the page's text (after
    removing the ** marks, spacing and punctuation) is dropped, so the
    composed annotations never put words in the author's mouth. The rest is
    tidied the same way whichever model composed it: items in reading order,
    pieces of one table joined, drawing descriptions left out of notes and
    only real symbols kept. Given the page's ``handwriting`` transcript, a
    note with words that are not in it is dropped: the composer places the
    readers' text, it never reads the handwriting a third time.
    """
    if not isinstance(data, dict):
        return None
    page = _letters(" ".join(blocks))
    written = _words(handwriting) if handwriting is not None else None

    def from_readers(note: str) -> str:
        return note if written is None or _words(note) <= written else ""

    placed = []  # (position on the page, item)
    tables: Dict[str, list] = {}
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        item = dict(item)
        if "symbols" in item:
            item["symbols"] = _clean_symbols(item.get("symbols"))
        if "note" in item:
            item["note"] = from_readers(_clean_note(item.get("note")))
        if item.get("type") == "table":
            cells = [
                c
                for row in item.get("rows") or []
                if isinstance(row, list)
                for c in row
                if isinstance(c, str)
            ]
            words = [
                _letters(c.split("✎")[0].replace("**", ""))
                for c in cells + [item.get("caption", "")]
            ]
            if cells and all(w in page for w in words if w):
                tables.setdefault(_letters(item.get("caption", "")) or str(id(item)), []).append(
                    item
                )
        elif item.get("type") == "passage" and isinstance(item.get("quote"), str):
            quote = _letters(item["quote"].replace("**", "").replace("…", " "))
            if quote and quote in page and "**" in item["quote"]:
                placed.append((page.find(quote), item))
    for pieces in tables.values():
        table = _merge_tables(pieces, page)
        keys = [_row_key(r) for r in table["rows"] if isinstance(r, list)]
        placed.append((min((page.find(k) for k in keys if k), default=10**9), table))
    items = [item for _, item in sorted(placed, key=lambda pair: pair[0])]
    notes = [from_readers(_clean_note(n)) for n in data.get("unplaced_notes") or []]
    notes = [n for n in notes if n]
    if not items and not notes:
        return None
    return {"items": items, "unplaced_notes": notes}
