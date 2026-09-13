"""
Phase 2a: Markdown -> retrieval chunks.

Rules
-----
1. A table is never split across a chunk boundary. If a table is genuinely too
   large for one chunk it is split by *rows*, and the header row is repeated on
   every part so each part still makes sense standing alone.
2. Every chunk carries its full heading breadcrumb, both as metadata and
   prefixed into the embedded text. This is what preserves the relationship
   between a section header and the prose/tables beneath it -- an isolated
   "$40 copay" row is useless without "Outpatient Services > Specialist Visits".
3. Page provenance is carried through from the <!-- page: N --> anchors so
   answers can cite a page in the original COC.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

#: Bump when extraction or chunking logic changes in a way that invalidates
#: previously stored chunks. Lets you find and re-process stale rows.
PIPELINE_VERSION = "1.1.0"

PAGE_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
#: Emitted only for compilations where each bound document restarts its own
#: page numbering. Absent for documents that number straight through.
FORM_RE = re.compile(r"<!--\s*form:\s*(.+?)\s+printed:\s*(\d+)\s*-->")
TABLE_OPEN_RE = re.compile(r"<!--\s*table\s+(.*?)\s*-->")
TABLE_CLOSE = "<!-- /table -->"
ROWS_OPEN = "<!-- table-rows -->"
ROWS_CLOSE = "<!-- /table-rows -->"
DOC_RE = re.compile(r"<!--\s*document:\s*(.*?)\s*-->")
SKIPPED_PAGE_RE = re.compile(r"<!--\s*skipped-page:\s*(\d+)\s*(.*?)-->")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
ATTR_RE = re.compile(r"(\w+)=([^\s]+)")


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #
_ENCODER = None


def _get_encoder():
    """Use tiktoken only if it is installed *and* its BPE files are cached
    locally (TIKTOKEN_CACHE_DIR), so nothing is downloaded at runtime."""
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER or None
    if not os.environ.get("TIKTOKEN_CACHE_DIR"):
        _ENCODER = False
        return None
    try:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = False
    return _ENCODER or None


def estimate_tokens(text: str) -> int:
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # ~4 chars/token is a good approximation for English policy prose.
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# Options / model
# --------------------------------------------------------------------------- #
@dataclass
class ChunkOptions:
    target_tokens: int = 650
    max_tokens: int = 900
    overlap_tokens: int = 90
    max_table_tokens: int = 1100
    min_chunk_tokens: int = 25
    include_row_sentences: bool = True
    prefix_breadcrumb: bool = True
    #: Empty-cell ratio above which a table chunk is flagged for human review.
    table_review_empty_ratio: float = 0.35

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class DocumentContext:
    """Identity and provenance for one COC, stamped onto every chunk it produces.

    Plan identity is the field group most worth getting right before you load
    production data. Without `plan_id` / `plan_year` you cannot answer "what was
    the copay under the 2025 plan" and you cannot safely hold two carriers in one
    collection -- and a wrong-plan answer is the highest-consequence failure this
    system has. These are declared as filter fields on the vector index so they
    are enforced at query time, not just stored.
    """

    doc_id: str

    # --- plan identity ---
    plan_id: Optional[str] = None
    group_number: Optional[str] = None
    plan_year: Optional[int] = None
    effective_date: Optional[str] = None        # ISO-8601 date string
    termination_date: Optional[str] = None      # ISO-8601 date string, or None
    carrier: Optional[str] = None
    market_segment: Optional[str] = None        # e.g. large_group, small_group, individual
    states: List[str] = field(default_factory=list)

    # --- provenance ---
    source_filename: Optional[str] = None
    source_sha256: Optional[str] = None
    pipeline_version: str = PIPELINE_VERSION

    def stamp(self) -> Dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "group_number": self.group_number,
            "plan_year": self.plan_year,
            "effective_date": self.effective_date,
            "termination_date": self.termination_date,
            "carrier": self.carrier,
            "market_segment": self.market_segment,
            "states": list(self.states),
            "source_filename": self.source_filename,
            "source_sha256": self.source_sha256,
            "pipeline_version": self.pipeline_version,
        }


def sha256_file(path: str, buf_size: int = 1 << 20) -> str:
    """Hash the source PDF so a re-upload can be recognised as the same document."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(buf_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    seq: int
    type: str                     # "text" | "table"
    heading: str
    section_path: List[str]
    breadcrumb: str
    text: str                     # raw content, no breadcrumb
    embed_text: str               # what actually gets embedded
    page_start: int
    page_end: int
    n_tokens: int

    # --- table detail ---
    table_id: Optional[str] = None
    part_index: int = 1
    part_total: int = 1
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    column_labels: List[str] = field(default_factory=list)

    # --- bound-document identity (compilations only) ---
    #: Which form-coded document inside the PDF this chunk came from, and the
    #: page number printed on it. A compilation restarts page numbering per
    #: document, so a printed page number alone cannot identify a provision.
    form_code: Optional[str] = None
    printed_page_start: Optional[int] = None
    printed_page_end: Optional[int] = None
    has_spans: bool = False

    # --- plan identity (stamped from DocumentContext) ---
    plan_id: Optional[str] = None
    group_number: Optional[str] = None
    plan_year: Optional[int] = None
    effective_date: Optional[str] = None
    termination_date: Optional[str] = None
    carrier: Optional[str] = None
    market_segment: Optional[str] = None
    states: List[str] = field(default_factory=list)

    # --- provenance ---
    source_filename: Optional[str] = None
    source_sha256: Optional[str] = None
    pipeline_version: str = PIPELINE_VERSION
    chunk_config_hash: str = ""
    created_at: str = ""
    updated_at: str = ""

    # --- embedding lifecycle (populated at embed time) ---
    embed_model: Optional[str] = None
    embed_dim: Optional[int] = None
    embedded_at: Optional[str] = None

    # --- review workflow ---
    needs_review: bool = False
    extraction_notes: List[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        """Human-readable source reference, unambiguous within a compilation."""
        if self.printed_page_start:
            pages = (f"p. {self.printed_page_start}"
                     if self.printed_page_start == self.printed_page_end
                     else f"pp. {self.printed_page_start}-{self.printed_page_end}")
        else:
            pages = (f"PDF p. {self.page_start}" if self.page_start == self.page_end
                     else f"PDF pp. {self.page_start}-{self.page_end}")
        prefix = f"{self.form_code}, " if self.form_code else ""
        return f"{prefix}{pages}"

    def to_doc(self) -> dict:
        """MongoDB document. `chunk_id` becomes `_id` -- the IDs are already
        deterministic, so this gives free idempotent upserts and drops both a
        duplicated string and a second unique index."""
        d = asdict(self)
        d["_id"] = d.pop("chunk_id")
        return d


@dataclass
class _Block:
    kind: str                     # "para" | "table"
    text: str
    page_start: int
    page_end: int
    section_path: List[str]
    meta: Dict[str, str] = field(default_factory=dict)
    row_sentences: str = ""
    form_code: Optional[str] = None
    printed_start: Optional[int] = None
    printed_end: Optional[int] = None


# --------------------------------------------------------------------------- #
# Markdown parsing
# --------------------------------------------------------------------------- #
def parse_markdown(markdown: str) -> Tuple[List[_Block], Optional[str]]:
    lines = markdown.splitlines()
    blocks: List[_Block] = []
    stack: List[Tuple[int, str]] = []
    page = 0
    form_code: Optional[str] = None
    printed: Optional[int] = None
    doc_title = None
    para: List[str] = []
    para_page = 0
    para_printed: Optional[int] = None

    def flush_para():
        nonlocal para, para_page, para_printed
        # rstrip only: leading whitespace encodes list nesting, and a nested
        # exclusion needs its parent item to read as a rule rather than a bare
        # list of nouns.
        text = "\n".join(para).rstrip()
        if text:
            blocks.append(
                _Block("para", text, para_page or page, page, [t for _, t in stack],
                       form_code=form_code, printed_start=para_printed or printed,
                       printed_end=printed)
            )
        para = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        m = DOC_RE.match(line.strip())
        if m:
            doc_title = m.group(1)
            i += 1
            continue

        if SKIPPED_PAGE_RE.match(line.strip()):
            i += 1
            continue

        m = PAGE_RE.match(line.strip())
        if m:
            flush_para()
            page = int(m.group(1))
            i += 1
            continue

        m = FORM_RE.match(line.strip())
        if m:
            form_code = m.group(1).strip()
            printed = int(m.group(2))
            i += 1
            continue

        m = TABLE_OPEN_RE.match(line.strip())
        if m:
            flush_para()
            attrs = dict(ATTR_RE.findall(m.group(1)))
            body: List[str] = []
            rows: List[str] = []
            in_rows = False
            i += 1
            while i < len(lines) and lines[i].strip() != TABLE_CLOSE:
                s = lines[i].strip()
                if s == ROWS_OPEN:
                    in_rows = True
                elif s == ROWS_CLOSE:
                    in_rows = False
                elif in_rows:
                    rows.append(lines[i])
                else:
                    body.append(lines[i])
                i += 1
            i += 1  # consume the closing comment
            pages = attrs.get("pages", str(page))
            first, _, last = pages.partition("-")
            try:
                p0 = int(first)
                p1 = int(last) if last else p0
            except ValueError:
                p0 = p1 = page
            blocks.append(
                _Block(
                    "table",
                    "\n".join(body).strip(),
                    p0,
                    p1,
                    [t for _, t in stack],
                    meta=attrs,
                    row_sentences="\n".join(rows).strip(),
                    form_code=form_code,
                    printed_start=printed,
                    printed_end=printed,
                )
            )
            page = max(page, p1)
            continue

        m = HEADING_RE.match(line)
        if m:
            flush_para()
            level, text = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            i += 1
            continue

        if not line.strip():
            flush_para()
            i += 1
            continue

        if not para:
            para_page = page
            para_printed = printed
        para.append(line)
        i += 1

    flush_para()
    return blocks, doc_title


# --------------------------------------------------------------------------- #
# Table helpers
# --------------------------------------------------------------------------- #
def _split_pipe_table(text: str) -> Tuple[List[str], List[str]]:
    """Return (header_lines, data_lines) for a GFM pipe table."""
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(rows) >= 2 and set(rows[1].replace("|", "").strip()) <= set("- :"):
        return rows[:2], rows[2:]
    return rows[:1], rows[1:]


def _pipe_row_sentences(text: str) -> str:
    """Row-per-line expansion, carrying the running banner label.

    A benefit table groups rows under a full-width banner naming the benefit
    ("Acupuncture"), while the rows beneath only ask "What Is the Coinsurance
    You Pay?". Carrying the banner forward is what lets a query about
    acupuncture reach the row that holds its coinsurance."""
    header, data = _split_pipe_table(text)
    if not header:
        return ""
    clean = lambda v: " ".join(v.replace("<br>", " ").split())
    labels = [clean(c) for c in header[0].strip().strip("|").split("|")]
    out: List[str] = []
    group = ""
    for row in data:
        vals = [clean(c) for c in row.strip().strip("|").split("|")]
        if vals and vals[0] and not any(v for v in vals[1:]):
            group = vals[0]
            continue
        parts = [f"{l}: {v}" for l, v in zip(labels, vals) if v]
        if parts:
            out.append((f"{group} - " if group else "") + " | ".join(parts))
    return "\n".join(out)


def _split_html_table(text: str) -> Tuple[str, List[str], str]:
    rows = re.findall(r"<tr>.*?</tr>", text, flags=re.S)
    header = [r for r in rows if "<th" in r]
    data = [r for r in rows if "<th" not in r]
    return "\n".join(header), data, ""


def _pipe_labels(text: str) -> List[str]:
    header, _ = _split_pipe_table(text)
    if not header:
        return []
    return [c.strip() for c in header[0].strip().strip("|").split("|")]


def _html_labels(text: str) -> List[str]:
    cells = re.findall(r"<th[^>]*>(.*?)</th>", text, flags=re.S)
    return [" ".join(re.sub(r"<[^>]+>", " ", c).split()) for c in cells]


def _table_quality(text: str, is_html: bool, threshold: float) -> Tuple[bool, List[str]]:
    """Cheap extraction-confidence check for a table chunk.

    A high proportion of empty cells usually means the ruling-line grid picked up
    more boundaries than the table really has -- the classic symptom of a
    mis-detected table. Flagging it is the hook for the human-review step rather
    than something to silently 'fix'.
    """
    notes: List[str] = []
    if is_html:
        return False, notes
    _, data = _split_pipe_table(text)
    if not data:
        return False, notes
    total = empty = 0
    for row in data:
        for cell in row.strip().strip("|").split("|"):
            total += 1
            if not cell.strip():
                empty += 1
    if total and empty / total > threshold:
        notes.append(f"{empty}/{total} table cells empty ({empty / total:.0%}) - verify grid detection")
        return True, notes
    return False, notes


# --------------------------------------------------------------------------- #
# Chunk assembly
# --------------------------------------------------------------------------- #
def _breadcrumb(path: Sequence[str]) -> str:
    return " > ".join(p for p in path if p)


def _wrap(breadcrumb: str, body: str, note: str = "", enabled: bool = True) -> str:
    if not enabled:
        return body
    head = []
    if breadcrumb:
        head.append(f"Section: {breadcrumb}")
    if note:
        head.append(note)
    return ("\n".join(head) + "\n\n" + body).strip() if head else body


def _mk_id(doc_id: str, seq: int) -> str:
    digest = hashlib.md5(f"{doc_id}#{seq}".encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}#{seq:05d}-{digest}"


def chunk_markdown(
    markdown: str,
    doc: Union[str, DocumentContext],
    options: Optional[ChunkOptions] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> List[Chunk]:
    """Chunk a converted COC.

    `doc` accepts either a plain doc_id (back-compatible) or a DocumentContext
    carrying plan identity and provenance, which is what you want in production.
    """
    ctx = DocumentContext(doc_id=doc) if isinstance(doc, str) else doc
    doc_id = ctx.doc_id
    opts = options or ChunkOptions()
    blocks, _ = parse_markdown(markdown)
    chunks: List[Chunk] = []
    seq = 0

    stamp = ctx.stamp()
    config_hash = opts.config_hash()
    now = _now()

    def emit(**kw) -> None:
        nonlocal seq
        embed_text = kw.pop("embed_text")
        chunks.append(
            Chunk(
                chunk_id=_mk_id(doc_id, seq),
                doc_id=doc_id,
                seq=seq,
                embed_text=embed_text,
                n_tokens=estimate_tokens(embed_text),
                chunk_config_hash=config_hash,
                created_at=now,
                updated_at=now,
                **stamp,
                **kw,
            )
        )
        seq += 1

    # group consecutive paragraph blocks that share a section path
    i = 0
    n = len(blocks)
    while i < n:
        blk = blocks[i]
        if progress and n:
            progress(i / n, f"Chunking block {i + 1}/{n}")

        if blk.kind == "table":
            _emit_table(blk, opts, emit)
            i += 1
            continue

        group = [blk]
        j = i + 1
        while j < n and blocks[j].kind == "para" and blocks[j].section_path == blk.section_path:
            group.append(blocks[j])
            j += 1
        _emit_text_group(group, opts, emit)
        i = j

    return chunks


def _emit_text_group(group: List[_Block], opts: ChunkOptions, emit) -> None:
    path = group[0].section_path
    form_code = group[0].form_code
    breadcrumb = _breadcrumb(path)
    heading = path[-1] if path else ""
    prefix_cost = estimate_tokens(f"Section: {breadcrumb}\n\n") if opts.prefix_breadcrumb else 0

    buf: List[str] = []
    buf_tokens = 0
    page_start = group[0].page_start
    page_end = group[0].page_end
    printed_lo = group[0].printed_start
    printed_hi = group[0].printed_end

    def flush(carry: bool = True) -> None:
        nonlocal buf, buf_tokens, page_start, page_end, printed_lo, printed_hi
        body = "\n\n".join(buf).strip()
        if not body:
            buf, buf_tokens = [], 0
            return
        if estimate_tokens(body) >= opts.min_chunk_tokens or len(buf) > 1:
            emit(
                type="text",
                heading=heading,
                form_code=form_code,
                printed_page_start=printed_lo,
                printed_page_end=printed_hi,
                section_path=list(path),
                breadcrumb=breadcrumb,
                text=body,
                embed_text=_wrap(breadcrumb, body, enabled=opts.prefix_breadcrumb),
                page_start=page_start,
                page_end=page_end,
            )
        # overlap: carry the trailing paragraph into the next chunk
        tail: List[str] = []
        tail_tokens = 0
        if carry and opts.overlap_tokens > 0:
            for para in reversed(buf):
                t = estimate_tokens(para)
                if tail_tokens + t > opts.overlap_tokens:
                    break
                tail.insert(0, para)
                tail_tokens += t
        buf, buf_tokens = tail, tail_tokens

    for b in group:
        for para in [p for p in b.text.split("\n\n") if p.strip()]:
            t = estimate_tokens(para)
            if buf and buf_tokens + t + prefix_cost > opts.max_tokens:
                flush()
                page_start = b.page_start
            if not buf:
                page_start = b.page_start
            buf.append(para)
            buf_tokens += t
            page_end = b.page_end
            if printed_lo is None:
                printed_lo = b.printed_start
            if b.printed_end is not None:
                printed_hi = b.printed_end
            if buf_tokens + prefix_cost >= opts.target_tokens:
                flush()
                page_start = b.page_end
    flush(carry=False)


def _emit_table(blk: _Block, opts: ChunkOptions, emit) -> None:
    breadcrumb = _breadcrumb(blk.section_path)
    heading = blk.section_path[-1] if blk.section_path else ""
    table_id = blk.meta.get("id")
    n_rows = int(blk.meta.get("rows", 0) or 0)
    n_cols = int(blk.meta.get("cols", 0) or 0)
    has_spans = str(blk.meta.get("spans", "")).lower() == "yes"
    is_html = blk.text.lstrip().startswith("<table")

    labels = _html_labels(blk.text) if is_html else _pipe_labels(blk.text)
    needs_review, notes = _table_quality(blk.text, is_html, opts.table_review_empty_ratio)

    row_sentences = blk.row_sentences
    if opts.include_row_sentences and not row_sentences and not is_html:
        row_sentences = _pipe_row_sentences(blk.text)

    def build(body: str, rows_text: str, part: int, total: int) -> str:
        note = f"Table {table_id or ''} (pages {blk.page_start}-{blk.page_end})".strip()
        if total > 1:
            note += f", part {part} of {total}"
        payload = body
        if rows_text:
            payload += "\n\nRow details:\n" + rows_text
        return _wrap(breadcrumb, payload, note=note, enabled=True)

    whole = build(blk.text, row_sentences, 1, 1)
    if estimate_tokens(whole) <= opts.max_table_tokens:
        emit(
            type="table",
            heading=heading,
            section_path=list(blk.section_path),
            breadcrumb=breadcrumb,
            text=blk.text,
            embed_text=whole,
            page_start=blk.page_start,
            page_end=blk.page_end,
            table_id=table_id,
            form_code=blk.form_code,
            printed_page_start=blk.printed_start,
            printed_page_end=blk.printed_end,
            part_index=1,
            part_total=1,
            n_rows=n_rows or None,
            n_cols=n_cols or None,
            column_labels=labels,
            has_spans=has_spans,
            needs_review=needs_review,
            extraction_notes=list(notes),
        )
        return

    # Too large: split by rows, repeating the header on every part.
    if is_html:
        header, data, _ = _split_html_table(blk.text)
        make = lambda rows: "<table>\n" + header + "\n" + "\n".join(rows) + "\n</table>"
        sentences = [""] * len(data)
    else:
        header, data = _split_pipe_table(blk.text)
        make = lambda rows: "\n".join(header + rows)
        sentences = row_sentences.splitlines() if row_sentences else [""] * len(data)
        if len(sentences) != len(data):
            sentences = [""] * len(data)

    header_cost = estimate_tokens(make([])) + estimate_tokens(breadcrumb) + 40
    groups: List[Tuple[List[str], List[str]]] = []
    cur_rows: List[str] = []
    cur_sents: List[str] = []
    cur_tokens = header_cost
    for row, sent in zip(data, sentences):
        t = estimate_tokens(row) + estimate_tokens(sent)
        if cur_rows and cur_tokens + t > opts.max_table_tokens:
            groups.append((cur_rows, cur_sents))
            cur_rows, cur_sents, cur_tokens = [], [], header_cost
        cur_rows.append(row)
        cur_sents.append(sent)
        cur_tokens += t
    if cur_rows:
        groups.append((cur_rows, cur_sents))

    total = len(groups)
    for idx, (rows, sents) in enumerate(groups, start=1):
        body = make(rows)
        rows_text = "\n".join(s for s in sents if s)
        emit(
            type="table",
            heading=heading,
            section_path=list(blk.section_path),
            breadcrumb=breadcrumb,
            text=body,
            embed_text=build(body, rows_text, idx, total),
            page_start=blk.page_start,
            page_end=blk.page_end,
            table_id=table_id,
            form_code=blk.form_code,
            printed_page_start=blk.printed_start,
            printed_page_end=blk.printed_end,
            part_index=idx,
            part_total=total,
            n_rows=len(rows),
            n_cols=n_cols or None,
            column_labels=labels,
            has_spans=has_spans,
            needs_review=needs_review,
            extraction_notes=list(notes),
        )
