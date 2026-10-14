"""Background transcription jobs: whole notebooks, one or two LLM readers.

remarkable_transcribe starts a job and returns immediately; the work runs in
the MCP server process (no images pass through the client). Each page is
rendered once, skipped if it has no ink, then read by one or more LLM
backends. In "dual" mode a third pass looks at the page image together with
both readings and writes the final transcription, marking anything still
uncertain with (?). Every step is cached on disk (see llm_ocr), so re-running
a job is cheap and an interrupted job resumes where it stopped.

Results are written as Markdown (+ JSON) under REMARKABLE_TRANSCRIPT_DIR
(default ~/Documents/reMarkable-transcripts). Markdown labels follow
REMARKABLE_TRANSCRIPT_LANG ("en" default, "fi").

A library index (_library.json in the same folder) records, per document id,
the modification time that was last transcribed in full and whether the
document is still on the tablet. remarkable_transcribe_sync uses it to
transcribe only new and changed documents; transcripts of documents deleted
from the tablet stay where they are and are marked "deleted".
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from remarkable_mcp import llm_ocr

logger = logging.getLogger(__name__)

MODES = {
    "dual": ["anthropic", "gemini"],
    "anthropic": ["anthropic"],
    "gemini": ["gemini"],
}

_JOBS: Dict[str, "Job"] = {}


@dataclass
class PageResult:
    page: int
    status: str = (
        "pending"  # pending | blank | done | failed (pending also = awaiting adjudication)
    )
    readings: Dict[str, Optional[str]] = field(default_factory=dict)
    final: Optional[str] = None
    agreement: Optional[float] = None
    error: Optional[str] = None
    # Printed words the pen marked on a PDF page (from the text layer, no OCR)
    marked: List[str] = field(default_factory=list)


@dataclass
class Job:
    id: str
    document: str
    path: str
    mode: str
    pages: List[int]
    total_pages: int
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    state: str = "running"  # running | done | failed
    error: Optional[str] = None
    results: Dict[int, PageResult] = field(default_factory=dict)
    output_md: Optional[str] = None
    output_json: Optional[str] = None
    task: Optional[asyncio.Task] = None
    # backend -> model override; "judge" -> adjudication model
    models: Dict[str, str] = field(default_factory=dict)
    document_id: Optional[str] = None
    modified: Optional[str] = None  # document modification time (ISO 8601)
    page_ids: List[Optional[str]] = field(default_factory=list)  # stable ids, page order
    usage: Dict[str, dict] = field(default_factory=dict)  # model -> calls/tokens
    # PDF pages: callable(source, page) -> {"marked", "residual_png"} or None
    annotate: Optional[Callable] = None
    extracted: Optional[Path] = None  # the document unpacked once, for annotate

    def page_id(self, page: int) -> Optional[str]:
        return self.page_ids[page - 1] if 0 < page <= len(self.page_ids) else None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def parse_pages(spec: Optional[str], total: int) -> List[int]:
    """'all' | '1-20,30,40-' -> sorted unique page numbers within 1..total."""
    if not spec or spec.strip().lower() in ("all", "*"):
        return list(range(1, total + 1))
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d*)\s*-\s*(\d*)", part)
        if m:
            lo = int(m.group(1)) if m.group(1) else 1
            hi = int(m.group(2)) if m.group(2) else total
        elif part.isdigit():
            lo = hi = int(part)
        else:
            raise ValueError(f"Invalid page spec: {part!r}")
        pages.update(range(max(1, lo), min(total, hi) + 1))
    return sorted(pages)


def has_ink(png: bytes, threshold: int = 40) -> bool:
    """True if the rendered page contains at least ``threshold`` dark pixels."""
    import pymupdf

    pix = pymupdf.Pixmap(png)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    if pix.n >= 3 and not pix.colorspace.name.lower().startswith("devicegray"):
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
    samples = pix.samples
    dark = 0
    step = pix.n
    for i in range(0, len(samples), step):
        if samples[i] < 160:
            dark += 1
            if dark >= threshold:
                return True
    return False


def agreement(a: Optional[str], b: Optional[str]) -> Optional[float]:
    if a is None or b is None:
        return None
    wa, wb = a.split(), b.split()
    if not wa and not wb:
        return 1.0
    return difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()


_ALT_RE = re.compile(r"\{[^{}\n|]+(?:\|[^{}\n|]+)+\}")
_SUGGEST_RE = re.compile(r"([^\s→{}|\[\]]+)→([^\s→{}|\[\]]+) \(\?\)")
_DOUBT_RE = re.compile(r"\(\?\)|\[\?: [^\]\n]+\]")
_ILLEGIBLE_RE = re.compile(r"\[\?\]")

# Shared notation, also used by speech transcripts. "basis" says where a marker
# comes from, as in the speech pipeline's word-level JSON.
NOTATION = {
    "{a|b}": "two plausible readings, likelier first (basis: reader_disagreement)",
    "word (?)": "doubtful reading (basis: self_report)",
    "[?: w1 w2]": "run of doubtful words, best guesses kept (basis: self_report)",
    "word→suggestion (?)": "image shows word; context or vocabulary suggests another "
    "(basis: self_report)",
    "[?]": "illegible",
}


def uncertainty(text: Optional[str]) -> Dict[str, int]:
    """Count markers: {a|b}, doubts ((?) and [?: …] runs), → suggestions, [?]."""
    if not text:
        return {"alternatives": 0, "doubtful": 0, "suggestions": 0, "illegible": 0}
    suggestions = len(_SUGGEST_RE.findall(text))
    return {
        "alternatives": len(_ALT_RE.findall(text)),
        "doubtful": len(_DOUBT_RE.findall(text)) - suggestions,
        "suggestions": suggestions,
        "illegible": len(_ILLEGIBLE_RE.findall(text)),
    }


def suggestions(text: Optional[str]) -> List[List[str]]:
    """The word→suggestion (?) pairs, each as [as written, suggested]."""
    if not text:
        return []
    return [[m.group(1), m.group(2)] for m in _SUGGEST_RE.finditer(text)]


def alternatives(text: Optional[str]) -> List[List[str]]:
    """The {a|b} groups in a transcription, each as a list of options."""
    if not text:
        return []
    return [m.group(0)[1:-1].split("|") for m in _ALT_RE.finditer(text)]


# Labels for the Markdown transcript. REMARKABLE_TRANSCRIPT_LANG selects the
# language ("en" default, "fi" available); add a dict to support another one.
TRANSCRIPT_LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "source": "Source",
        "transcribed": "Transcribed",
        "readers": "Readers",
        "adjudicated": " (differences resolved against the image)",
        "legend": "Markers",
        "legend_text": (
            "{a|b} = two plausible readings, likelier first; "
            "word (?) = doubtful reading; [?: …] = doubtful run; "
            "word→suggestion (?) = image shows word, context suggests another; "
            "[?] = illegible"
        ),
        "page": "Page",
        "failed": "Transcription failed",
        "all_failed": "all readers failed (see log)",
        "no_answer": "(no answer)",
        "agreement": "reader agreement",
        "readers_summary": "Readers (agreement {pct})",
        "certainty": "certainty",
        "high": "high",
        "moderate": "moderate",
        "low": "low",
        "alternatives": "{n} alternative readings",
        "doubtful": "{n} doubtful",
        "suggestions": "{n} suggestions",
        "marked": "Marked in the text",
        "illegible": "{n} illegible",
    },
    "fi": {
        "source": "Lähde",
        "transcribed": "Litteroitu",
        "readers": "Lukijat",
        "adjudicated": " (ero ratkaistu kuvaa vasten)",
        "legend": "Merkinnät",
        "legend_text": (
            "{a|b} = kaksi uskottavaa lukua, todennäköisempi ensin; "
            "sana (?) = epävarma luku; [?: …] = epävarma jakso; "
            "sana→ehdotus (?) = kuvassa sana, konteksti ehdottaa toista; "
            "[?] = lukukelvoton"
        ),
        "page": "Sivu",
        "failed": "Litterointi epäonnistui",
        "all_failed": "kaikki lukijat epäonnistuivat (ks. loki)",
        "no_answer": "(ei vastausta)",
        "agreement": "lukijoiden yhtäpitävyys",
        "readers_summary": "Lukijat (yhtäpitävyys {pct})",
        "certainty": "varmuus",
        "high": "korkea",
        "moderate": "kohtalainen",
        "low": "matala",
        "alternatives": "{n} vaihtoehtolukua",
        "doubtful": "{n} epävarmaa",
        "suggestions": "{n} ehdotusta",
        "marked": "Merkityt kohdat",
        "illegible": "{n} lukukelvotonta",
    },
}


def labels() -> Dict[str, str]:
    lang = os.environ.get("REMARKABLE_TRANSCRIPT_LANG", "en").strip().lower()
    return TRANSCRIPT_LABELS.get(lang, TRANSCRIPT_LABELS["en"])


def certainty_level(agreement_value: Optional[float], counts: Dict[str, int]) -> str:
    """Coarse page-level certainty: "high", "moderate", "low" or "unknown".

    Combines word-level agreement between the two readers with the number of
    uncertainty markers the adjudicator left in the final text.
    """
    flagged = (
        counts["alternatives"]
        + counts["doubtful"]
        + counts.get("suggestions", 0)
        + 2 * counts["illegible"]
    )
    if agreement_value is None:
        return "low" if flagged >= 4 else "moderate" if flagged else "unknown"
    if agreement_value >= 0.9 and flagged <= 1:
        return "high"
    if agreement_value >= 0.7 and flagged <= 4:
        return "moderate"
    return "low"


def _pct(value: Optional[float]) -> str:
    return f"{round(value * 100)} %" if value is not None else "?"


def _uncertainty_line(r: "PageResult") -> str:
    lab = labels()
    c = uncertainty(r.final)
    level = certainty_level(r.agreement, c)
    parts = [
        f"{lab['certainty']} {lab.get(level, '?')}",
        f"{lab['agreement']} {_pct(r.agreement)}",
    ]
    for key in ("alternatives", "doubtful", "suggestions", "illegible"):
        if c[key]:
            parts.append(lab[key].format(n=c[key]))
    return "_" + ", ".join(parts) + "_"


def transcript_dir() -> Path:
    raw = os.environ.get("REMARKABLE_TRANSCRIPT_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "Documents" / "reMarkable-transcripts"


def _safe_name(path: str, suffix: str = "") -> str:
    name = path.strip("/").replace("/", " - ") or "document"
    if suffix:
        name += f" [{suffix}]"
    return re.sub(r'[\\:*?"<>|]', "_", name)


_LABELS = {"anthropic": "Claude", "gemini": "Gemini"}


def write_outputs(job: Job) -> None:
    out = transcript_dir()
    out.mkdir(parents=True, exist_ok=True)
    tags = [f"judge {job.models[b]}" if b == "judge" else job.models[b] for b in sorted(job.models)]
    base = out / _safe_name(job.path, " + ".join(tags))
    readers = " + ".join(
        _LABELS.get(b, b) + (f" ({job.models[b]})" if b in job.models else "")
        for b in MODES[job.mode]
    )
    lab = labels()
    lines = [
        f"# {job.document}",
        "",
        f"{lab['source']}: `{job.path}`  ",
        f"{lab['transcribed']}: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"{lab['readers']}: {readers}" + (lab["adjudicated"] if job.mode == "dual" else ""),
    ]
    if job.mode == "dual":
        lines[-1] += "  "
        lines.append(f"{lab['legend']}: {lab['legend_text']}")
    lines.append("")
    for p in sorted(job.results):
        r = job.results[p]
        if r.status == "blank":
            continue
        lines += [f"## {lab['page']} {p}", ""]
        if r.status == "failed":
            lines += [f"_{lab['failed']}: {r.error}_", ""]
            continue
        if r.marked:
            lines += [f"_{lab['marked']}:_", ""] + [f"- {m}" for m in r.marked] + [""]
        if not r.final:
            continue
        if job.mode == "dual":
            lines += [_uncertainty_line(r), ""]
        lines += [r.final, ""]
        if job.mode == "dual" and r.readings:
            summary = lab["readers_summary"].format(pct=_pct(r.agreement))
            lines += [f"<details><summary>{summary}</summary>", ""]
            for b in MODES["dual"]:
                lines += [
                    f"**{_LABELS[b]}**",
                    "",
                    "```",
                    r.readings.get(b) or lab["no_answer"],
                    "```",
                    "",
                ]
            lines += ["</details>", ""]
    md = base.with_name(base.name + ".md")
    md.write_text("\n".join(lines), encoding="utf-8")
    js = base.with_name(base.name + ".json")
    js.write_text(
        json.dumps(
            {
                "document": job.document,
                "path": job.path,
                "document_id": job.document_id,
                "modified": job.modified,
                "mode": job.mode,
                "models": job.models,
                "usage": job.usage,
                "estimated_cost_usd": llm_ocr.estimate_cost(job.usage),
                "notation": NOTATION,
                "pages": {
                    str(p): {
                        "page_id": job.page_id(p),
                        "status": r.status,
                        "marked": r.marked,
                        "final": r.final,
                        "readings": r.readings,
                        "agreement": r.agreement,
                        "uncertainty": uncertainty(r.final),
                        "certainty": certainty_level(r.agreement, uncertainty(r.final))
                        if job.mode == "dual" and r.status == "done"
                        else None,
                        "alternatives": alternatives(r.final),
                        "suggestions": suggestions(r.final),
                        "error": r.error,
                    }
                    for p, r in sorted(job.results.items())
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    job.output_md, job.output_json = str(md), str(js)


# ---------------------------------------------------------------------------
# job runner
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"\w")
# A reader's description of a drawing, e.g. "[two horizontal lines; oval]".
# Uncertainty markers ("[?]", "[?: w1 w2]") are not descriptions.
_DESCRIPTION_RE = re.compile(r"\[(?!\?)[^\]]*\]")


def _has_handwriting(text: str) -> bool:
    """True if a reading holds written words, not only symbols or descriptions."""
    return bool(_WORD_RE.search(_DESCRIPTION_RE.sub("", text)))


async def _page_image(job: Job, zip_path: Path, page: int, render):
    """The image to read for a page, and the printed words its pen marks cover.

    On a PDF page the marks are taken from the text layer and only the rest of
    the ink (margin notes, other handwriting) is returned as an image. Notebook
    pages return the whole ink layer and ``None`` for the marks.
    """
    from remarkable_mcp.concurrency import run_blocking

    if job.annotate is not None:
        analysis = await run_blocking(job.annotate, job.extracted or zip_path, page)
        if analysis is not None:
            return analysis.get("residual_png"), list(analysis.get("marked") or [])
    return await run_blocking(render, zip_path, page), None


async def _read_page(job: Job, zip_path: Path, page: int, render, sem) -> None:
    """Phase 1: render, skip blank pages, collect every reader's text."""
    from remarkable_mcp.concurrency import run_blocking

    r = job.results[page]
    llm_ocr.USAGE_SINK.set(job.usage)  # this task's context; copied into worker threads
    async with sem:
        try:
            png, marked = await _page_image(job, zip_path, page, render)
            r.marked = marked or []
            if png is None or not await run_blocking(has_ink, png):
                r.status = "done" if r.marked else "blank"
                r.final = "" if r.marked else None
                return
            backends = MODES[job.mode]
            texts = await asyncio.gather(
                *(run_blocking(llm_ocr.ocr_png, png, b, 120.0, job.models.get(b)) for b in backends)
            )
            r.readings = dict(zip(backends, texts))
            answered = {b: t for b, t in r.readings.items() if t is not None}
            if not answered:
                r.status = "failed"
                r.error = labels()["all_failed"]
                return
            if all(t == "" for t in answered.values()) or (
                marked is not None and not any(_has_handwriting(t) for t in answered.values())
            ):
                # nothing written, or (on a PDF page) only lines, symbols and
                # descriptions of drawings
                r.status = "done" if r.marked else "blank"
                r.final = "" if r.marked else None
                return
            if job.mode == "dual":
                a, g = r.readings.get("anthropic"), r.readings.get("gemini")
                r.agreement = agreement(a, g)
                r.final = a if a is not None else g
                if a is not None and g is not None and a != g:
                    r.status = "pending"  # adjudicated in phase 2
                    return
            else:
                r.final = next(iter(answered.values()))
            r.status = "done"
        except Exception as e:  # never kill the whole job for one page
            logger.warning("Transcription of page %s failed: %s", page, e)
            r.status = "failed"
            r.error = str(e)


def _context_text(r: Optional["PageResult"]) -> Optional[str]:
    if r is None or r.status not in ("done", "pending"):
        return None
    # Prefer the Gemini reading: it is usually the closer read of names and words
    return r.readings.get("gemini") or r.readings.get("anthropic") or r.final


def page_context(job: Job, page: int) -> Dict[str, Optional[str]]:
    """Text of the nearest non-blank page before and after ``page``."""
    order = sorted(job.results)
    i = order.index(page)
    prev = next((job.results[p] for p in reversed(order[:i]) if job.results[p].final), None)
    nxt = next((job.results[p] for p in order[i + 1 :] if job.results[p].final), None)
    return {"previous": _context_text(prev), "next": _context_text(nxt)}


async def _adjudicate_page(job: Job, zip_path: Path, page: int, render, sem) -> None:
    """Phase 2: resolve differing readings against the image and nearby pages."""
    from remarkable_mcp.concurrency import run_blocking

    r = job.results[page]
    llm_ocr.USAGE_SINK.set(job.usage)
    async with sem:
        try:
            png, _ = await _page_image(job, zip_path, page, render)
            final = await run_blocking(
                llm_ocr.adjudicate_png,
                png,
                r.readings,
                180.0,
                page_context(job, page),
                job.models.get("judge"),
            )
            if final:
                r.final = final
            r.status = "done"
        except Exception as e:
            logger.warning("Adjudication of page %s failed: %s", page, e)
            r.status = "done"  # keep the first reader's text rather than lose the page


async def run_job(job: Job, raw_doc: bytes, render, workers: int) -> None:
    tmp = None
    unpacked = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(raw_doc)
            tmp = Path(f.name)
        if job.annotate is not None:
            import zipfile

            unpacked = tempfile.TemporaryDirectory(prefix="remarkable-doc-")
            with zipfile.ZipFile(tmp) as zf:
                zf.extractall(unpacked.name)
            job.extracted = Path(unpacked.name)
        sem = asyncio.Semaphore(max(1, workers))
        await asyncio.gather(*(_read_page(job, tmp, p, render, sem) for p in job.pages))
        pending = [p for p in job.pages if job.results[p].status == "pending"]
        await asyncio.gather(*(_adjudicate_page(job, tmp, p, render, sem) for p in pending))
        if any(r.status != "blank" for r in job.results.values()):
            write_outputs(job)  # a document without handwriting leaves no files behind
        job.state = "done"
        try:
            record_in_index(job)
        except Exception as e:  # the transcript is written; the index can catch up later
            logger.warning("Could not update the library index: %s", e)
    except asyncio.CancelledError:
        job.state = "cancelled"
        if any(r.status == "done" for r in job.results.values()):
            try:
                write_outputs(job)  # keep what was finished
            except Exception:
                pass
    except Exception as e:
        logger.warning("Transcription job %s failed: %s", job.id, e)
        job.state = "failed"
        job.error = str(e)
        try:
            write_outputs(job)
        except Exception:
            pass
    finally:
        job.finished = time.time()
        job.extracted = None
        if unpacked is not None:
            unpacked.cleanup()
        if tmp:
            tmp.unlink(missing_ok=True)


def start_job(
    *,
    document: str,
    path: str,
    mode: str,
    pages: List[int],
    total_pages: int,
    raw_doc: bytes,
    render,
    workers: int,
    models: Optional[Dict[str, str]] = None,
    document_id: Optional[str] = None,
    modified: Optional[str] = None,
    page_ids: Optional[List[Optional[str]]] = None,
    annotate: Optional[Callable] = None,
) -> Job:
    job = Job(
        id=uuid.uuid4().hex[:8],
        document=document,
        path=path,
        mode=mode,
        pages=pages,
        total_pages=total_pages,
        models=dict(models or {}),
        document_id=document_id,
        modified=modified,
        page_ids=list(page_ids or []),
        annotate=annotate,
    )
    job.results = {p: PageResult(page=p) for p in pages}
    _JOBS[job.id] = job
    _prune_jobs()
    job.task = asyncio.get_running_loop().create_task(run_job(job, raw_doc, render, workers))
    return job


MAX_FINISHED_JOBS = 200


def _prune_jobs() -> None:
    """Forget the oldest finished jobs so a long-running server stays small."""
    finished = [j for j in _JOBS.values() if j.finished is not None]
    for job in sorted(finished, key=lambda j: j.finished)[
        : max(0, len(finished) - MAX_FINISHED_JOBS)
    ]:
        _JOBS.pop(job.id, None)


def cancel_job(job_id: str) -> Optional[Job]:
    """Stop a running job; finished pages are written, the rest stays pending."""
    job = _JOBS.get(job_id)
    if job and job.task and not job.task.done():
        job.task.cancel()
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _JOBS.get(job_id)


def list_jobs() -> List[Job]:
    return list(_JOBS.values())


def summarize(job: Job) -> dict:
    counts = {"pending": 0, "blank": 0, "done": 0, "failed": 0}
    for r in job.results.values():
        counts[r.status] = counts.get(r.status, 0) + 1
    agreements = [r.agreement for r in job.results.values() if r.agreement is not None]
    elapsed = (job.finished or time.time()) - job.started
    out = {
        "job_id": job.id,
        "document": job.document,
        "path": job.path,
        "mode": job.mode,
        **({"models": job.models} if job.models else {}),
        "state": job.state,
        "pages_requested": len(job.pages),
        "pages_total_in_document": job.total_pages,
        **counts,
        "elapsed_seconds": round(elapsed),
    }
    if agreements:
        out["mean_agreement"] = round(sum(agreements) / len(agreements), 3)
        out["pages_below_80pct_agreement"] = sorted(
            r.page for r in job.results.values() if r.agreement is not None and r.agreement < 0.8
        )
    if job.mode == "dual":
        flagged = sorted(
            r.page
            for r in job.results.values()
            if r.status == "done" and any(uncertainty(r.final).values())
        )
        if flagged:
            out["pages_with_uncertainty_markers"] = flagged
    if (
        job.state == "done"
        and job.results
        and all(r.status == "blank" for r in job.results.values())
    ):
        out["no_handwriting"] = True
    marked = sorted(r.page for r in job.results.values() if r.marked)
    if marked:
        out["pages_with_marks"] = marked
    if job.usage:
        out["usage"] = job.usage
        cost = llm_ocr.estimate_cost(job.usage)
        if cost is not None:
            out["estimated_cost_usd"] = cost
    if job.output_md:
        out["output_markdown"] = job.output_md
        out["output_json"] = job.output_json
    if job.error:
        out["error"] = job.error
    failed = [r.page for r in job.results.values() if r.status == "failed"]
    if failed:
        out["failed_pages"] = failed
    return out


def brief(job: Job) -> dict:
    """One line per job for listings."""
    counts = {"done": 0, "blank": 0, "failed": 0, "pending": 0}
    for r in job.results.values():
        counts[r.status] = counts.get(r.status, 0) + 1
    out = {"job_id": job.id, "path": job.path, "state": job.state, **counts}
    cost = llm_ocr.estimate_cost(job.usage) if job.usage else None
    if cost is not None:
        out["estimated_cost_usd"] = cost
    return out


# ---------------------------------------------------------------------------
# library index and sync runs
# ---------------------------------------------------------------------------

INDEX_NAME = "_library.json"


def index_path() -> Path:
    return transcript_dir() / INDEX_NAME


def load_index() -> dict:
    path = index_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("documents"), dict):
            return data
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        logger.warning("Library index unreadable, starting a new one: %s", e)
    return {"documents": {}}


def save_index(index: dict) -> None:
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    index["updated"] = datetime.now().isoformat(timespec="seconds")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".library-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _is_main_run(job: Job) -> bool:
    """Full document with the configured models: the transcript the index tracks."""
    if "judge" in job.models or len(job.pages) != job.total_pages:
        return False
    return all(job.models.get(b) in (None, llm_ocr._model_for(b)) for b in MODES[job.mode])


def record_in_index(job: Job) -> None:
    if not job.document_id or job.state != "done" or not _is_main_run(job):
        return
    if any(r.status == "failed" for r in job.results.values()):
        return  # leave it marked as changed so the next sync retries
    index = load_index()
    index["documents"][job.document_id] = {
        "document": job.document,
        "path": job.path,
        "modified": job.modified,
        "status": "on_device",
        "transcribed": datetime.now().isoformat(timespec="seconds"),
        "mode": job.mode,
        "pages_with_text": sum(1 for r in job.results.values() if r.status == "done"),
        "no_handwriting": all(r.status == "blank" for r in job.results.values()),
        "output_markdown": job.output_md,
        "output_json": job.output_json,
    }
    save_index(index)


def plan_sync(documents: List[dict], index: dict) -> dict:
    """Compare the tablet's documents with the index.

    ``documents`` holds {"id", "path", "modified"} for every document that may
    be transcribed. Returns the new and changed ones, how many are unchanged,
    and the indexed documents that are no longer on the tablet.
    """
    known = index.get("documents", {})
    new, changed, unchanged = [], [], 0
    for doc in documents:
        entry = known.get(doc["id"])
        if entry is None:
            new.append(doc)
        elif entry.get("modified") != doc["modified"] or entry.get("status") != "on_device":
            changed.append(doc)
        else:
            unchanged += 1
    present = {doc["id"] for doc in documents}
    deleted = [
        {"id": doc_id, "path": entry.get("path")}
        for doc_id, entry in known.items()
        if doc_id not in present and entry.get("status") == "on_device"
    ]
    return {"new": new, "changed": changed, "unchanged": unchanged, "deleted": deleted}


def mark_deleted(doc_ids: List[str]) -> None:
    if not doc_ids:
        return
    index = load_index()
    stamp = datetime.now().isoformat(timespec="seconds")
    for doc_id in doc_ids:
        entry = index["documents"].get(doc_id)
        if entry is not None:
            entry["status"] = "deleted"
            entry["deleted"] = stamp
    save_index(index)


@dataclass
class SyncRun:
    id: str
    planned: List[dict]
    deleted: List[dict]
    unchanged: int
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    state: str = "running"
    job_ids: List[str] = field(default_factory=list)
    errors: List[dict] = field(default_factory=list)
    task: Optional[asyncio.Task] = None


_SYNCS: Dict[str, SyncRun] = {}


def start_sync(planned: List[dict], deleted: List[dict], unchanged: int, launch, parallel: int):
    """Transcribe ``planned`` documents in the background, ``parallel`` at a time.

    ``launch(doc)`` is an async callable that starts a job for one document and
    returns it (or raises).
    """
    run = SyncRun(
        id="sync-" + uuid.uuid4().hex[:6], planned=planned, deleted=deleted, unchanged=unchanged
    )
    _SYNCS[run.id] = run
    mark_deleted([d["id"] for d in deleted])

    async def one(doc, sem):
        async with sem:
            try:
                job = await launch(doc)
                run.job_ids.append(job.id)
                await job.task
            except Exception as e:
                logger.warning("Sync of %s failed: %s", doc.get("path"), e)
                run.errors.append({"path": doc.get("path"), "error": str(e)})

    async def go():
        sem = asyncio.Semaphore(max(1, parallel))
        try:
            await asyncio.gather(*(one(doc, sem) for doc in planned))
            run.state = "done"
        except asyncio.CancelledError:
            run.state = "cancelled"
        finally:
            run.finished = time.time()

    run.task = asyncio.get_running_loop().create_task(go())
    return run


def cancel_sync(sync_id: str) -> Optional[SyncRun]:
    """Stop a sync: no new documents start and running ones are cancelled."""
    run = _SYNCS.get(sync_id)
    if run is None:
        return None
    if run.task and not run.task.done():
        run.task.cancel()
    for job_id in run.job_ids:
        cancel_job(job_id)
    return run


def get_sync(sync_id: str) -> Optional[SyncRun]:
    return _SYNCS.get(sync_id)


def list_syncs() -> List[SyncRun]:
    return list(_SYNCS.values())


def summarize_sync(run: SyncRun) -> dict:
    jobs = [j for j in (get_job(i) for i in run.job_ids) if j is not None]
    usage: Dict[str, dict] = {}
    for job in jobs:
        for model, entry in job.usage.items():
            total = usage.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            for key in total:
                total[key] += entry.get(key, 0)
    out = {
        "sync_id": run.id,
        "state": run.state,
        "documents_planned": len(run.planned),
        "documents_finished": sum(1 for j in jobs if j.finished is not None),
        "unchanged": run.unchanged,
        "deleted_from_tablet": [d["path"] for d in run.deleted],
        "pages_transcribed": sum(1 for j in jobs for r in j.results.values() if r.status == "done"),
        "elapsed_seconds": round((run.finished or time.time()) - run.started),
        "jobs": [brief(j) for j in jobs],
    }
    if usage:
        out["usage"] = usage
        cost = llm_ocr.estimate_cost(usage)
        if cost is not None:
            out["estimated_cost_usd"] = cost
    if run.errors:
        out["errors"] = run.errors
    return out
