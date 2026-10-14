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
from typing import Dict, List, Optional

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
        if job.mode == "dual":
            lines += [_uncertainty_line(r), ""]
        lines += [r.final or "", ""]
        if job.mode == "dual":
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
                "mode": job.mode,
                "models": job.models,
                "notation": NOTATION,
                "pages": {
                    str(p): {
                        "status": r.status,
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


async def _read_page(job: Job, zip_path: Path, page: int, render, sem) -> None:
    """Phase 1: render, skip blank pages, collect every reader's text."""
    from remarkable_mcp.concurrency import run_blocking

    r = job.results[page]
    async with sem:
        try:
            png = await run_blocking(render, zip_path, page)
            if png is None or not await run_blocking(has_ink, png):
                r.status = "blank"
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
            if all(t == "" for t in answered.values()):
                r.status = "blank"
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
    async with sem:
        try:
            png = await run_blocking(render, zip_path, page)
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
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(raw_doc)
            tmp = Path(f.name)
        sem = asyncio.Semaphore(max(1, workers))
        await asyncio.gather(*(_read_page(job, tmp, p, render, sem) for p in job.pages))
        pending = [p for p in job.pages if job.results[p].status == "pending"]
        await asyncio.gather(*(_adjudicate_page(job, tmp, p, render, sem) for p in pending))
        write_outputs(job)
        job.state = "done"
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
) -> Job:
    job = Job(
        id=uuid.uuid4().hex[:8],
        document=document,
        path=path,
        mode=mode,
        pages=pages,
        total_pages=total_pages,
        models=dict(models or {}),
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
    if job.output_md:
        out["output_markdown"] = job.output_md
        out["output_json"] = job.output_json
    if job.error:
        out["error"] = job.error
    failed = [r.page for r in job.results.values() if r.status == "failed"]
    if failed:
        out["failed_pages"] = failed
    return out
