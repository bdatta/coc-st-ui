"""
Phase 1: COC PDF -> high-fidelity Markdown.

Design notes
------------
* Text and tables are extracted from the same page pass and re-interleaved by
  vertical position, so a table stays where it belongs relative to its heading
  and surrounding prose.
* Words falling inside a detected table's bbox are removed from the prose flow,
  otherwise every table would also appear as garbled paragraphs.
* Tables that continue across a page break are stitched back together before
  rendering, so downstream chunking never has to deal with half a table.
* Machine-readable anchors are emitted as HTML comments:
      <!-- page: 12 -->
      <!-- table id=tbl-0007 pages=12-13 rows=24 cols=4 spans=yes -->
      <!-- /table -->
  The chunker uses these for page metadata and to keep tables atomic. They are
  invisible when the Markdown is rendered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections import Counter
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pdfplumber

from . import tables as T
from .layout import (
    BULLET_RE,
    INLINE_BULLET_RE,
    Line,
    detect_foreign_pages,
    detect_form_sections,
    find_running_lines,
    build_size_levels,
    heading_level,
    is_page_number,
    modal_body_size,
    normalize,
    words_to_lines,
)

WORD_EXTRAS = ["size", "fontname"]

TABLE_SETTINGS: Dict[str, dict] = {
    "lines": {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 3,
    },
    "lines_strict": {
        "vertical_strategy": "lines_strict",
        "horizontal_strategy": "lines_strict",
        "snap_tolerance": 2,
        "join_tolerance": 2,
        "intersection_tolerance": 2,
    },
    "text": {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "text_x_tolerance": 2,
        "text_y_tolerance": 2,
    },
    "hybrid": {
        "vertical_strategy": "lines",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
    },
}


#: "Section 5: Eligibility .................................. 4"
TOC_LINE_RE = re.compile(r"^\s*\S.*?[\.\u2026]{4,}\s*\d{1,4}\s*$")


def _is_toc_page(lines: Sequence[Line], min_ratio: float = 0.4) -> bool:
    """Does this page consist mainly of dot-leader contents entries?

    A bound policy carries a contents page for each instrument it contains, not
    just one at the front, so a page range cannot skip them. Left in, each entry
    becomes a chunk that weakly matches almost any query -- pure retrieval noise
    that also inflates the heading count."""
    body = [ln for ln in lines if len(ln.text.strip()) > 3]
    if len(body) < 5:
        return False
    leaders = sum(1 for ln in body if TOC_LINE_RE.match(ln.text))
    return leaders / len(body) >= min_ratio


@dataclass
class ConversionOptions:
    table_strategy: str = "auto"          # auto | lines | lines_strict | text | hybrid
    table_format: str = "pipe"            # pipe | html
    heading_size_ratio: float = 1.08
    treat_allcaps_as_heading: bool = True
    treat_bold_as_heading: bool = True
    drop_running_lines: bool = True
    header_band: float = 0.10
    stitch_tables: bool = True
    detect_form_codes: bool = True
    #: Skip language-assistance pages. Translated notices carry no plan content
    #: but produce chunks that match weakly against almost any query.
    skip_foreign_pages: bool = True
    min_table_fill: float = 0.15
    emit_row_sentences: bool = False
    skip_toc_pages: bool = True
    force_heading_patterns: List[str] = field(default_factory=list)
    page_range: Optional[Tuple[int, int]] = None   # 1-based inclusive


@dataclass
class ConversionResult:
    markdown: str
    tables_json: List[dict]
    stats: Dict[str, object]
    warnings: List[str]


@dataclass
class _Block:
    kind: str          # "text" | "table"
    top: float
    page: int
    lines: List[Line] = field(default_factory=list)
    grid: Optional[T.Grid] = None
    #: True when every line sits in the top/bottom band -- page furniture that
    #: escaped running-line detection. Such blocks must not be allowed to break
    #: table stitching across a page boundary.
    in_band: bool = False


# --------------------------------------------------------------------------- #
# Page-level extraction
# --------------------------------------------------------------------------- #
def _find_grids(page, words, strategy: str, min_fill: float) -> List[T.Grid]:
    order = [strategy]
    if strategy == "auto":
        order = ["lines", "lines_strict"]

    for name in order:
        settings = TABLE_SETTINGS.get(name, TABLE_SETTINGS["lines"])
        try:
            found = page.find_tables(table_settings=settings)
        except Exception:
            found = []
        grids: List[T.Grid] = []
        for tbl in found:
            grid = T.grid_from_table(tbl, words, page.page_number)
            if grid and not T.looks_degenerate(grid, min_fill):
                grids.append(grid)
        grids = _drop_nested(grids)
        if grids:
            return grids
    return []


def _drop_nested(grids: List[T.Grid], pad: float = 2.0) -> List[T.Grid]:
    """Discard grids wholly inside another grid.

    Borders drawn as filled rectangles make pdfplumber report the inner boxes of
    individual cells as tables in their own right. Left in, they surface as
    nonsense two-column fragments and -- worse -- they sit between a table and
    its continuation, breaking cross-page stitching.
    """
    if len(grids) < 2:
        return grids
    boxes = [(_grid_bbox(g), g) for g in grids]
    boxes.sort(key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]), reverse=True)
    kept: List[T.Grid] = []
    kept_boxes: List[Tuple[float, float, float, float]] = []
    for (x0, t, x1, b), grid in boxes:
        inside = any(
            X0 - pad <= x0 and t >= T0 - pad and x1 <= X1 + pad and b <= B + pad
            for (X0, T0, X1, B) in kept_boxes
        )
        if not inside:
            kept.append(grid)
            kept_boxes.append((x0, t, x1, b))
    kept.sort(key=lambda g: (g.row_edges[0] if g.row_edges else 0.0))
    return kept


def _words_outside(words: Sequence[dict], boxes: Sequence[Tuple[float, float, float, float]],
                   pad: float = 1.0) -> List[dict]:
    if not boxes:
        return list(words)
    out = []
    for w in words:
        cx = (w["x0"] + w["x1"]) / 2.0
        cy = (w["top"] + w["bottom"]) / 2.0
        inside = any(
            (x0 - pad) <= cx <= (x1 + pad) and (t - pad) <= cy <= (b + pad)
            for (x0, t, x1, b) in boxes
        )
        if not inside:
            out.append(w)
    return out


def _grid_bbox(grid: T.Grid) -> Tuple[float, float, float, float]:
    xs, ys = grid.col_edges, grid.row_edges
    return (xs[0], ys[0], xs[-1], ys[-1])


def _grid_to_dict(grid: T.Grid) -> dict:
    return {
        "table_id": grid.table_id,
        "pages": grid.pages,
        "n_rows": grid.n_rows,
        "n_cols": grid.n_cols,
        "header_rows": grid.header_rows,
        "has_spans": grid.has_spans,
        "caption": grid.caption,
        "column_labels": T.header_labels(grid),
        "cells": [
            {"row": c.r0, "col": c.c0, "rowspan": c.rowspan,
             "colspan": c.colspan, "text": c.text}
            for c in grid.cells
        ],
    }


# --------------------------------------------------------------------------- #
# Prose assembly
# --------------------------------------------------------------------------- #
def _dehyphenate(prev: str, nxt: str) -> Optional[str]:
    if prev.endswith("-") and len(prev) > 2 and nxt[:1].islower():
        return prev[:-1] + nxt
    return None


def _indent_for(para: Sequence[Line], base_x0: Optional[float],
                indent_step: float) -> str:
    """Markdown indent for a nested list item.

    Sub-bullets sit further right on the page: in a real SPD the parent bullets
    start at x=72 and their examples at x=93.6. Keeping that nesting matters for
    citation, because a leaf like "Chairs, bath chairs, feeding chairs" is a list
    of nouns with no operative verb -- what makes it a denial basis is the parent
    ("Supplies, equipment ... for personal comfort. Examples include:") and the
    lead-in above it ("The following are not Covered Health Care Services:").
    """
    if base_x0 is None or not para:
        return ""
    depth = int(max(0.0, (para[0].x0 - base_x0)) // indent_step)
    return "  " * min(depth, 3)


def _lines_to_paragraphs(lines: Sequence[Line]) -> List[List[Line]]:
    """Split a run of body lines into paragraphs on vertical gaps / bullets."""
    if not lines:
        return []
    heights = [ln.height for ln in lines if ln.height > 0] or [10.0]
    typical = median(heights)
    paras: List[List[Line]] = [[lines[0]]]
    for prev, cur in zip(lines, lines[1:]):
        gap = cur.top - prev.bottom
        new_para = (
            gap > typical * 0.9
            or bool(BULLET_RE.match(cur.text))
            or (cur.x0 - prev.x0 > 18 and gap > typical * 0.35)
        )
        if new_para:
            paras.append([cur])
        else:
            paras[-1].append(cur)
    return paras


def _render_paragraph(para: Sequence[Line], base_x0: Optional[float] = None,
                      indent_step: float = 12.0) -> str:
    parts: List[str] = []
    for ln in para:
        t = " ".join(ln.text.split())
        if not t:
            continue
        if parts:
            joined = _dehyphenate(parts[-1], t)
            if joined is not None:
                parts[-1] = joined
                continue
        parts.append(t)
    text = " ".join(parts).strip()
    if not text:
        return ""

    # A paragraph may carry several bullet markers inline when the PDF wrapped
    # the items onto shared lines. Split so each rule becomes its own list item
    # and can be cited on its own.
    pieces = [x.strip() for x in INLINE_BULLET_RE.split(text) if x.strip()]
    if len(pieces) > 1:
        pad = _indent_for(para, base_x0, indent_step)
        lead_in = "" if INLINE_BULLET_RE.match(text) else pieces.pop(0)
        items = "\n".join(f"{pad}- {x}" for x in pieces)
        return f"{lead_in}\n{items}".strip() if lead_in else items

    if BULLET_RE.match(text):
        text = BULLET_RE.sub("", text, count=1).strip()
        return f"{_indent_for(para, base_x0, indent_step)}- {text}"
    return text


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def convert_pdf(
    pdf_path: str,
    options: Optional[ConversionOptions] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    doc_title: Optional[str] = None,
) -> ConversionResult:
    opts = options or ConversionOptions()
    warnings: List[str] = []
    force_patterns = []
    for pat in opts.force_heading_patterns:
        try:
            force_patterns.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            warnings.append(f"Ignored invalid heading regex: {pat!r}")

    pages_lines: List[List[Line]] = []
    pages_grids: List[List[T.Grid]] = []
    page_heights: List[float] = []
    page_numbers: List[int] = []

    form_layout = None
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        lo, hi = opts.page_range or (1, total)
        # Either end may be omitted by a caller; fill it in rather than fail.
        lo = max(1, lo or 1)
        hi = min(total, hi or total)
        if lo > hi:
            lo, hi = hi, lo

        if opts.detect_form_codes:
            found = detect_form_sections(pdf, page_range=(lo, hi))
            form_layout = found if found.found else None

        foreign_pages = (detect_foreign_pages(pdf, page_range=(lo, hi))
                         if opts.skip_foreign_pages else [])
        skip = {f.page for f in foreign_pages}
        for f in foreign_pages:
            warnings.append(f"Skipped page {f.page} as a language-assistance "
                            f"page ({f.reason}).")

        for idx in range(lo - 1, hi):
            if (idx + 1) in skip:
                continue
            page = pdf.pages[idx]
            if progress:
                progress((idx - lo + 1) / max(hi - lo + 1, 1), f"Reading page {idx + 1}/{hi}")

            words = page.extract_words(
                extra_attrs=WORD_EXTRAS, keep_blank_chars=False, use_text_flow=False
            )
            grids = _find_grids(page, words, opts.table_strategy, opts.min_table_fill)
            boxes = [_grid_bbox(g) for g in grids]
            prose_words = _words_outside(words, boxes)

            pages_lines.append(words_to_lines(prose_words, page.page_number))
            pages_grids.append(grids)
            page_heights.append(float(page.height))
            page_numbers.append(page.page_number)
            page.flush_cache()

    if not pages_lines:
        return ConversionResult("", [], {"pages": 0}, ["No pages processed."])

    # ---- document-level statistics -------------------------------------- #
    margin_counts: Counter = Counter()
    for lines in pages_lines:
        for ln in lines:
            margin_counts[round(ln.x0)] += len(ln.text)
    doc_margin_x0 = (float(min(x for x, c in margin_counts.items()
                               if c >= max(margin_counts.values()) * 0.05))
                     if margin_counts else None)

    body_size = modal_body_size(pages_lines)
    size_levels = build_size_levels(pages_lines, body_size, opts.heading_size_ratio)
    running = (
        find_running_lines(pages_lines, page_heights, band=opts.header_band)
        if opts.drop_running_lines
        else set()
    )

    # ---- build ordered blocks ------------------------------------------- #
    doc_blocks: List[_Block] = []
    toc_pages_skipped: List[int] = []
    for lines, grids, height, pno in zip(pages_lines, pages_grids, page_heights, page_numbers):
        kept: List[Line] = []
        for ln in lines:
            in_band = ln.top <= height * opts.header_band or ln.bottom >= height * (1 - opts.header_band)
            if in_band and (normalize(ln.text) in running or is_page_number(ln.text)):
                continue
            kept.append(ln)

        band_top = height * opts.header_band
        band_bottom = height * (1 - opts.header_band)

        def _mk_text_block(lines: List[Line]) -> _Block:
            in_band = all(ln.top <= band_top or ln.bottom >= band_bottom for ln in lines)
            short = sum(len(ln.text) for ln in lines) <= 120
            return _Block("text", lines[0].top, pno, lines=list(lines),
                          in_band=in_band and short)

        if opts.skip_toc_pages and _is_toc_page(kept):
            toc_pages_skipped.append(pno)
            continue

        blocks: List[_Block] = []
        run: List[Line] = []
        grid_tops = sorted(((g.row_edges[0] if g.row_edges else 0.0), g) for g in grids)
        gi = 0
        for ln in kept:
            while gi < len(grid_tops) and grid_tops[gi][0] <= ln.top:
                if run:
                    blocks.append(_mk_text_block(run))
                    run = []
                top, grid = grid_tops[gi]
                blocks.append(_Block("table", top, pno, grid=grid))
                gi += 1
            run.append(ln)
        if run:
            blocks.append(_mk_text_block(run))
        while gi < len(grid_tops):
            top, grid = grid_tops[gi]
            blocks.append(_Block("table", top, pno, grid=grid))
            gi += 1

        doc_blocks.extend(blocks)

    # ---- stitch tables across page breaks -------------------------------- #
    if opts.stitch_tables:
        stitched: List[_Block] = []
        for blk in doc_blocks:
            if blk.kind == "table":
                # Look back past page furniture. A footer that escaped detection
                # sits between a table and its continuation; requiring strict
                # adjacency would let one stray line silently split the table.
                j = len(stitched) - 1
                while j >= 0 and stitched[j].kind == "text" and stitched[j].in_band:
                    j -= 1
                if (
                    j >= 0
                    and stitched[j].kind == "table"
                    and stitched[j].page != blk.page
                    and T.columns_align(stitched[j].grid, blk.grid)
                ):
                    stitched[j].grid = T.merge_grids(stitched[j].grid, blk.grid)
                    continue
            stitched.append(blk)
        doc_blocks = stitched

    # ---- render ---------------------------------------------------------- #
    md: List[str] = []
    tables_json: List[dict] = []
    heading_count = 0
    current_page: Optional[int] = None
    table_seq = 0

    if doc_title:
        md.append(f"<!-- document: {doc_title} -->")
    for f in foreign_pages:
        # Recorded so validation can tell a deliberate omission from content
        # that went missing: a page with text and no output is otherwise the
        # signature of a dropped region.
        md.append(f'<!-- skipped-page: {f.page} reason="language assistance: '
                  f'{f.reason}" -->')
    # Record deliberate omissions. Without this the validator sees a page that
    # carries text but produced no content and correctly calls it missing --
    # a false alarm that would quarantine an otherwise clean conversion.
    for pno in toc_pages_skipped:
        md.append(f"<!-- skipped page={pno} reason=table-of-contents -->")

    for blk in doc_blocks:
        if blk.page != current_page:
            current_page = blk.page
            md.append(f"\n<!-- page: {current_page} -->")
            if form_layout is not None:
                section = form_layout.for_page(current_page)
                if section is not None:
                    printed = section.printed_for(current_page)
                    md.append(
                        f"<!-- form: {section.form_code} printed: {printed} -->"
                    )

        if blk.kind == "table":
            table_seq += 1
            grid = blk.grid
            grid.table_id = f"tbl-{table_seq:04d}"
            pages = f"{grid.page_start}-{grid.page_end}" if grid.page_start != grid.page_end else str(grid.page_start)
            caption = grid.caption
            md.append(
                f"\n<!-- table id={grid.table_id} pages={pages} rows={grid.n_rows} "
                f"cols={grid.n_cols} spans={'yes' if grid.has_spans else 'no'} -->"
            )
            if caption:
                # Kept above the grid rather than discarded: the lead-in states
                # what the whole table is subject to, and a benefit row read
                # without it can be materially misleading.
                md.append(f"*{caption}*\n")
            if opts.table_format == "html":
                md.append(T.to_html(grid))
            else:
                md.append(T.to_markdown_pipe(grid))
            if opts.emit_row_sentences:
                md.append("\n<!-- table-rows -->")
                md.append(T.to_row_sentences(grid))
                md.append("<!-- /table-rows -->")
            md.append("<!-- /table -->")
            tables_json.append(_grid_to_dict(grid))
            continue

        # text block: split into headings and paragraphs
        buffer: List[Line] = []
        pending: Optional[Tuple[int, List[Line]]] = None  # (level, lines)

        def flush_pending() -> None:
            nonlocal pending, heading_count
            if pending is None:
                return
            level, hlines = pending
            text = " ".join(" ".join(ln.text.split()) for ln in hlines)
            heading_count += 1
            md.append("\n" + "#" * level + " " + text)
            pending = None

        def flush_buffer() -> None:
            nonlocal buffer
            if not buffer:
                return
            # Measure indentation against the document's left margin, not the
            # block's own leftmost line. A list that continues across a page
            # break leaves the continuation page holding only sub-bullets, and
            # taking the block minimum would read those as top level -- losing
            # the parent that makes them a rule rather than a list of nouns.
            base_x0 = doc_margin_x0 if doc_margin_x0 is not None else min(
                ln.x0 for ln in buffer)
            for para in _lines_to_paragraphs(buffer):
                rendered = _render_paragraph(para, base_x0)
                if rendered:
                    md.append("\n" + rendered)
            buffer = []

        for ln in blk.lines:
            level = heading_level(
                ln,
                body_size,
                size_levels,
                treat_allcaps_as_heading=opts.treat_allcaps_as_heading,
                treat_bold_as_heading=opts.treat_bold_as_heading,
                force_patterns=force_patterns,
            )
            if level is None:
                flush_pending()
                buffer.append(ln)
                continue

            # A heading that wraps onto a second line is still one heading:
            # same level, same size, immediately below, no blank line between.
            if pending is not None:
                plevel, plines = pending
                prev = plines[-1]
                contiguous = (
                    plevel == level
                    and abs(prev.size - ln.size) < 0.6
                    and 0 <= (ln.top - prev.bottom) <= max(prev.height, 1.0) * 0.8
                )
                if contiguous:
                    plines.append(ln)
                    continue
                flush_pending()

            flush_buffer()
            pending = (level, [ln])

        flush_pending()
        flush_buffer()

    markdown = "\n".join(md).strip() + "\n"
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)

    if heading_count == 0:
        warnings.append(
            "No headings detected. Lower 'heading size ratio' or add a forced "
            "heading pattern (e.g. ^SECTION\\s+\\d+)."
        )
    if table_seq == 0:
        warnings.append(
            "No tables detected. If the COC uses whitespace-aligned tables with "
            "no ruling lines, try the 'text' table strategy."
        )

    stats = {
        "pages": len(pages_lines),
        "headings": heading_count,
        "tables": table_seq,
        "tables_with_spans": sum(1 for t in tables_json if t["has_spans"]),
        "tables_spanning_pages": sum(1 for t in tables_json if len(t["pages"]) > 1),
        "body_font_size": body_size,
        "heading_font_sizes": sorted(size_levels, reverse=True),
        "running_lines_removed": len(running),
        "foreign_pages_skipped": sorted(skip),
        "form_sections": (
            [{"form_code": x.form_code, "pdf_start": x.pdf_start, "pdf_end": x.pdf_end,
              "printed_start": x.printed_start, "printed_end": x.printed_end}
             for x in form_layout.sections] if form_layout else []
        ),
        "toc_pages_skipped": toc_pages_skipped,
        "characters": len(markdown),
    }
    return ConversionResult(markdown, tables_json, stats, warnings)
