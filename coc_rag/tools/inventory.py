"""
Verification: did every section of the COC actually survive conversion?

Two questions this answers, neither of which you should take on faith:

  1. What sections did the parser find, and what did each one produce?
     `section_inventory()` walks the generated Markdown and reports, per heading,
     how many text blocks and tables landed underneath it and across which pages.
     A section from the table of contents that shows zero blocks was missed.

  2. Does anything collide?
     `duplicate_breadcrumbs()` finds sections whose full heading path is
     identical. A COC that carries both a medical "Schedule of Benefits" and a
     prescription-drug "Schedule of Benefits" will produce two sets of chunks
     with the same breadcrumb -- and a retrieval hit on one is indistinguishable
     from the other. This is a correctness bug, not a cosmetic one.

Usage:
    python inventory.py coc.md
    python inventory.py coc.md --toc toc.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
import pathlib
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from coc.chunker import parse_markdown  # noqa: E402

#: "About this Booklet ............ 1"  ->  ("About this Booklet", 1)
TOC_LINE_RE = re.compile(r"^\s*(.+?)\s*[\.\u2026]{2,}\s*(\d{1,4})\s*$")


# --------------------------------------------------------------------------- #
# Section inventory
# --------------------------------------------------------------------------- #
@dataclass
class SectionRow:
    breadcrumb: str
    heading: str
    depth: int
    page_start: int = 0
    page_end: int = 0
    text_blocks: int = 0
    tables: int = 0
    chars: int = 0

    @property
    def blocks(self) -> int:
        return self.text_blocks + self.tables

    @property
    def profile(self) -> str:
        if self.tables and not self.text_blocks:
            return "table-only"
        if self.text_blocks and not self.tables:
            return "text-only"
        if not self.blocks:
            return "EMPTY"
        return "mixed"


def section_inventory(markdown: str) -> List[SectionRow]:
    """One row per *occurrence* of a heading path, in document order.

    Occurrence, not distinct path, on purpose. A COC that carries a medical
    "Schedule of Benefits" on page 8 and a prescription-drug one on page 163
    produces the same breadcrumb twice. Keying by path would merge them into a
    single row spanning pages 8-163 and hide the collision -- which is the exact
    bug this report exists to surface.
    """
    blocks, _ = parse_markdown(markdown)
    rows: List[SectionRow] = []

    for blk in blocks:
        key = " > ".join(blk.section_path) or "(no heading)"
        if not rows or rows[-1].breadcrumb != key:
            rows.append(SectionRow(
                breadcrumb=key,
                heading=blk.section_path[-1] if blk.section_path else "(no heading)",
                depth=len(blk.section_path),
                page_start=blk.page_start,
                page_end=blk.page_end,
            ))
        row = rows[-1]
        row.page_start = min(row.page_start or blk.page_start, blk.page_start)
        row.page_end = max(row.page_end, blk.page_end)
        row.chars += len(blk.text)
        if blk.kind == "table":
            row.tables += 1
        else:
            row.text_blocks += 1

    return rows


def top_level_sections(rows: Sequence[SectionRow]) -> List[SectionRow]:
    """Roll child sections up into their level-1 parent, which is the grain a
    table of contents is written at. Consecutive runs only, so a repeated
    top-level name still shows as two entries."""
    merged: List[SectionRow] = []
    for row in rows:
        top = row.breadcrumb.split(" > ")[0]
        if not merged or merged[-1].heading != top:
            merged.append(SectionRow(breadcrumb=top, heading=top, depth=1,
                                     page_start=row.page_start, page_end=row.page_end))
        m = merged[-1]
        m.page_start = min(m.page_start or row.page_start, row.page_start)
        m.page_end = max(m.page_end, row.page_end)
        m.text_blocks += row.text_blocks
        m.tables += row.tables
        m.chars += row.chars
    return merged


# --------------------------------------------------------------------------- #
# Collision detection
# --------------------------------------------------------------------------- #
def duplicate_breadcrumbs(rows: Sequence[SectionRow]) -> List[Tuple[str, List[SectionRow]]]:
    """Headings whose full path repeats. Ambiguous at retrieval time."""
    groups: Dict[str, List[SectionRow]] = defaultdict(list)
    for row in rows:
        groups[row.breadcrumb].append(row)
    return [(k, v) for k, v in groups.items() if len(v) > 1]


def repeated_headings(rows: Sequence[SectionRow]) -> List[Tuple[str, List[SectionRow]]]:
    """Same leaf heading text under different parents -- e.g. a medical and a
    prescription-drug 'Schedule of Benefits'. Safe only if the parent path
    actually distinguishes them."""
    groups: Dict[str, List[SectionRow]] = defaultdict(list)
    for row in rows:
        groups[row.heading.strip().lower()].append(row)
    return [(k, v) for k, v in groups.items() if len(v) > 1]


def suspected_toc_sections(rows: Sequence[SectionRow], markdown: str) -> List[SectionRow]:
    """Sections whose body looks like a table of contents rather than content."""
    blocks, _ = parse_markdown(markdown)
    hits: Dict[str, int] = defaultdict(int)
    for blk in blocks:
        if blk.kind != "para":
            continue
        lines = [ln for ln in blk.text.splitlines() if ln.strip()]
        if not lines:
            continue
        leader = sum(1 for ln in lines if TOC_LINE_RE.match(ln))
        if leader / len(lines) > 0.5:
            hits[" > ".join(blk.section_path) or "(no heading)"] += leader
    return [r for r in rows if hits.get(r.breadcrumb)]


# --------------------------------------------------------------------------- #
# Table-of-contents coverage
# --------------------------------------------------------------------------- #
def parse_toc(text: str) -> List[Tuple[str, Optional[int]]]:
    """Accepts pasted ToC text, with or without dot leaders and page numbers."""
    out: List[Tuple[str, Optional[int]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = TOC_LINE_RE.match(line)
        if m:
            out.append((m.group(1).strip(), int(m.group(2))))
        else:
            line = re.sub(r"\s+\d{1,4}$", "", line).strip()
            if line:
                out.append((line, None))
    return out


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def toc_coverage(
    markdown: str,
    toc: Sequence[Tuple[str, Optional[int]]],
    threshold: float = 0.82,
) -> List[dict]:
    """Match each ToC entry to a detected heading at *any* depth.

    Matching only leaf headings gives false misses: a section whose first child
    is a subsection never appears as a leaf, so "Section 1: Covered Health Care
    Services" looks absent even though it was detected correctly. Every
    component of every heading path is a candidate here, with pages and block
    counts rolled up from all its descendants.
    """
    rows = section_inventory(markdown)

    # component name -> aggregated coverage across every path it appears in
    agg: Dict[str, dict] = {}
    for row in rows:
        parts = row.breadcrumb.split(" > ")
        for depth, name in enumerate(parts, start=1):
            entry = agg.setdefault(name, {
                "name": name, "depth": depth, "page_start": row.page_start,
                "page_end": row.page_end, "text_blocks": 0, "tables": 0,
                "breadcrumb": " > ".join(parts[:depth]),
            })
            entry["page_start"] = min(entry["page_start"] or row.page_start, row.page_start)
            entry["page_end"] = max(entry["page_end"], row.page_end)
            entry["text_blocks"] += row.text_blocks
            entry["tables"] += row.tables

    candidates = [(v, _norm(k)) for k, v in agg.items()]

    report: List[dict] = []
    for title, page in toc:
        target = _norm(title)
        best: Optional[dict] = None
        best_score = 0.0
        for entry, norm_name in candidates:
            if not norm_name:
                continue
            score = SequenceMatcher(None, target, norm_name).ratio()
            if target and (target in norm_name or norm_name in target):
                score = max(score, 0.9)
            if score > best_score:
                best, best_score = entry, score
        matched = best is not None and best_score >= threshold
        report.append({
            "toc_title": title,
            "toc_page": page,
            "matched": matched,
            "score": round(best_score, 2),
            "detected_heading": best["name"] if matched and best else "",
            "breadcrumb": best["breadcrumb"] if matched and best else "",
            "pdf_pages": f"{best['page_start']}-{best['page_end']}" if matched and best else "",
            "text_blocks": best["text_blocks"] if matched and best else 0,
            "tables": best["tables"] if matched and best else 0,
            "profile": ("mixed" if matched and best and best["tables"] and best["text_blocks"]
                        else "table-only" if matched and best and best["tables"]
                        else "text-only" if matched and best else "NOT FOUND"),
        })
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _fmt(rows: Sequence[SectionRow], limit: Optional[int] = 60) -> str:
    """`limit=None` prints every row: --all-levels is asked for precisely when
    the caller wants the full list, and truncating it there looks exactly like
    missing headings."""
    out = [f"{'PAGES':>11}  {'TEXT':>5} {'TBL':>4}  {'PROFILE':<10} SECTION"]
    for r in (rows if limit is None else rows[:limit]):
        pages = f"{r.page_start}-{r.page_end}"
        indent = "  " * (r.depth - 1) if r.depth > 0 else ""
        out.append(f"{pages:>11}  {r.text_blocks:>5} {r.tables:>4}  "
                   f"{r.profile:<10} {indent}{r.heading[:70]}")
    if limit is not None and len(rows) > limit:
        out.append(f"  ... {len(rows) - limit} more "
                   f"(pass --all-levels to print every heading)")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify COC section coverage.")
    ap.add_argument("markdown", help="converted .md file")
    ap.add_argument("--toc", help="text file of table-of-contents lines")
    ap.add_argument("--all-levels", action="store_true",
                    help="show every heading, not just top-level sections")
    ap.add_argument("--divisions", action="store_true",
                    help="list only named divisions (Section N, Appendix X) with "
                         "their heading depth — the quickest check that no "
                         "section was missed or mis-nested")
    args = ap.parse_args()

    md = open(args.markdown, encoding="utf-8").read()
    rows = section_inventory(md)
    shown = rows if args.all_levels else top_level_sections(rows)

    if args.divisions:
        import re as _re

        div = _re.compile(r"^(section|appendix|schedule|part|article|exhibit)\b", _re.I)
        print(f"\n=== NAMED DIVISIONS ===\n")
        seen = set()
        for row in rows:
            for depth, name in enumerate(row.breadcrumb.split(" > "), start=1):
                if div.match(name.strip()) and name not in seen:
                    seen.add(name)
                    flag = "" if depth == 1 else f"   << nested at depth {depth}"
                    print(f"   H{depth}  p{row.page_start:<4} {name[:66]}{flag}")
        nested = [n for n in seen]
        print(f"\n   {len(seen)} division(s) found.")
        return

    print(f"\n=== SECTION INVENTORY ({len(rows)} headings, "
          f"{len(top_level_sections(rows))} top-level) ===\n")
    print(_fmt(shown, limit=None if args.all_levels else 60))

    empty = [r for r in shown if r.blocks == 0]
    if empty:
        print(f"\n!! {len(empty)} section(s) with no content:")
        for r in empty:
            print(f"   - {r.breadcrumb}")

    dupes = duplicate_breadcrumbs(rows)
    if dupes:
        print(f"\n!! {len(dupes)} DUPLICATE breadcrumb(s) - ambiguous at retrieval time:")
        for key, group in dupes:
            pages = ", ".join(f"p{g.page_start}-{g.page_end}" for g in group)
            print(f"   - {key}  ({pages})")

    repeats = [(k, v) for k, v in repeated_headings(rows)
               if len({r.breadcrumb for r in v}) > 1]
    if repeats:
        print(f"\n?  {len(repeats)} heading(s) reused under different parents "
              f"(fine only if the parent path disambiguates):")
        for key, group in repeats:
            for g in group:
                print(f"   - {g.breadcrumb}  (p{g.page_start}-{g.page_end})")

    toc_like = suspected_toc_sections(rows, md)
    if toc_like:
        print(f"\n?  {len(toc_like)} section(s) look like a table of contents "
              f"(dot leaders) - consider excluding from the index:")
        for r in toc_like:
            print(f"   - {r.breadcrumb}  (p{r.page_start}-{r.page_end})")

    if args.toc:
        toc = parse_toc(open(args.toc, encoding="utf-8").read())
        report = toc_coverage(md, toc)
        missing = [r for r in report if not r["matched"]]
        print(f"\n=== ToC COVERAGE: {len(report) - len(missing)}/{len(report)} matched ===\n")
        print(f"{'MATCH':<6} {'ToC PG':>6} {'PDF PAGES':>11} {'TEXT':>5} {'TBL':>4}  TITLE")
        for r in report:
            mark = "ok" if r["matched"] else "MISS"
            print(f"{mark:<6} {str(r['toc_page'] or ''):>6} {r['pdf_pages']:>11} "
                  f"{r['text_blocks']:>5} {r['tables']:>4}  {r['toc_title'][:56]}")
        if missing:
            print(f"\n!! {len(missing)} ToC entries not detected as headings. Their text is "
                  f"\n   probably still present but absorbed into the preceding section, "
                  f"\n   which means those chunks carry the wrong breadcrumb. Lower the "
                  f"\n   heading size ratio or add a forced pattern for each.")


if __name__ == "__main__":
    main()
