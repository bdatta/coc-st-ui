"""
Validation gate for converted COC Markdown.

Answers one question: is this Markdown good enough to embed?

Two families of check, and the distinction matters more than the individual
tests. Structural checks read the Markdown and confirm it is internally
well-formed -- cheap, but they share assumptions with the converter that
produced it, so they cannot catch a systematically wrong interpretation.
Reconciliation against the source PDF is independent: it compares the output to
something the converter did not author, which is the only way to detect content
that never made it out of the PDF at all.

Only the second family can fail in a way the converter did not anticipate, so
recall carries the largest share of the score.

Usage:
    python tools/validate.py coc.md --pdf coc.pdf
    python tools/validate.py coc.md --pdf coc.pdf --toc toc.txt --json report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from inventory import (  # noqa: E402
    duplicate_breadcrumbs,
    parse_toc,
    section_inventory,
    suspected_toc_sections,
    toc_coverage,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9$%]+")
PAGE_ANCHOR_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
FORM_ANCHOR_RE = re.compile(r"<!--\s*form:\s*(.+?)\s+printed:\s*(\d+)\s*-->")
SKIPPED_PAGE_RE = re.compile(r"<!--\s*skipped-page:\s*(\d+)")
#: Contents entries that name a division of the document. A miss here matters;
#: a missed sub-entry usually does not.
DIVISION_ENTRY_RE = re.compile(
    r"^\s*(section|appendix|schedule|part|article|exhibit|chapter)\b", re.I)
TABLE_OPEN_RE = re.compile(r"<!--\s*table\s+id=(\S+)")
TABLE_CLOSE = "<!-- /table -->"
HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*)$", re.M)

BLOCKER, WARNING, INFO = "blocker", "warning", "info"


# --------------------------------------------------------------------------- #
# Report model
# --------------------------------------------------------------------------- #
@dataclass
class Finding:
    check: str
    severity: str
    message: str
    detail: List[str] = field(default_factory=list)


@dataclass
class Report:
    score: float
    verdict: str                       # pass | review | quarantine
    metrics: Dict[str, object] = field(default_factory=dict)
    components: Dict[str, float] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)

    def add(self, check: str, severity: str, message: str,
            detail: Sequence[str] = ()) -> None:
        self.findings.append(Finding(check, severity, message, list(detail)[:20]))

    @property
    def blockers(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == BLOCKER]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["blocker_count"] = len(self.blockers)
        return d


# --------------------------------------------------------------------------- #
# 1. PDF text recall  (the independent check)
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def _split_shortfall(token: str, shortfall: int, band_count: int,
                     md_vocab: Sequence[str]) -> Dict[str, int]:
    """Apportion a token's shortfall across benign causes and real loss.

    Counts are split rather than the token being classified whole. An earlier
    version excused a token entirely whenever its shortfall fit within its
    header/footer occurrences -- which meant deleting an entire section could
    hide behind that section's running footer, and a 35,000-character deletion
    still scored 98/100. Apportioning gives furniture credit only for the
    occurrences the bands can actually account for; the remainder stays
    unexplained.

    Benign causes:
      * page furniture -- running headers/footers, removed on purpose. Detected
        geometrically from the top/bottom tenth of each page rather than guessed,
        because the furniture is not just page numbers: a footer reading
        "UnitedHealth Group HSA Medical Plans Booklet" contributes ordinary
        English words on 156 pages.
      * rejoined fragments -- the PDF layer emits 'mid' and 'year' where the
        Markdown correctly holds 'midyear', so the fragment is evidence the
        de-hyphenation worked.
    """
    out = {"page_furniture": 0, "rejoined": 0, "unexplained": 0}
    left = shortfall

    take = min(left, band_count)
    out["page_furniture"] += take
    left -= take
    if left <= 0:
        return out

    if token.isdigit():
        out["page_furniture"] += left
        return out

    if len(token) >= 2:
        for w in md_vocab:
            if len(w) > len(token) and (w.startswith(token) or w.endswith(token)):
                out["rejoined"] += left
                return out

    out["unexplained"] += left
    return out


def check_recall(report: Report, markdown: str, pdf_path: str) -> Optional[float]:
    try:
        import pdfplumber
    except ImportError:
        report.add("recall", WARNING, "pdfplumber not installed; recall not checked.")
        return None

    pages = sorted({int(p) for p in PAGE_ANCHOR_RE.findall(markdown)})
    if not pages:
        report.add("recall", BLOCKER, "No page anchors in the Markdown; cannot reconcile.")
        return 0.0

    md_counts = Counter(tokenize(markdown))
    md_vocab = list(md_counts)
    remaining = Counter(md_counts)

    total = found = 0
    page_totals: Dict[int, int] = {}
    page_missing: Dict[int, Counter] = {}
    unaccounted: Counter = Counter()
    band_only: Counter = Counter()

    # A table stitched across a page break is emitted once, under the anchor of
    # the page it began on, so the pages it continues onto legitimately produce
    # no anchor of their own. Credit those from the table's declared range.
    covered = set(pages)
    for span in re.findall(r"<!--\s*table\s+id=\S+\s+pages=(\d+)(?:-(\d+))?", markdown):
        start = int(span[0])
        end = int(span[1]) if span[1] else start
        covered.update(range(start, end + 1))

    # Pages the converter deliberately omitted, and said so.
    for pno in re.findall(r"<!--\s*skipped\s+page=(\d+)", markdown):
        covered.add(int(pno))

    # Pages the converter deliberately omitted -- language-assistance notices --
    # are recorded in the Markdown and are not content loss.
    skipped = {int(p) for p in SKIPPED_PAGE_RE.findall(markdown)}
    covered.update(skipped)
    missing_pages: List[str] = []
    anchors_with_text = set(pages)

    with pdfplumber.open(pdf_path) as pdf:
        n = len(pdf.pages)
        # Walk the whole converted range, not just the pages that produced an
        # anchor. A page carrying text but no anchor means its content never
        # reached the Markdown -- the fingerprint of a deleted or dropped
        # region, which token recall alone cannot see because a deleted
        # section's vocabulary usually also occurs elsewhere in the document.
        for pno in range(min(pages), max(pages) + 1):
            if pno < 1 or pno > n:
                continue
            page = pdf.pages[pno - 1]
            if pno not in covered:
                if len(tokenize(page.extract_text() or "")) > 25:
                    missing_pages.append(str(pno))
                page.flush_cache()
                continue
            if pno not in anchors_with_text:
                page.flush_cache()
                continue
            page_counts = Counter(tokenize(page.extract_text() or ""))
            if not page_counts:
                continue

            # Split the page's vocabulary by position so header/footer words can
            # be told apart from body words that merely look similar.
            height = float(page.height)
            for w in page.extract_words():
                tok = w["text"].lower()
                if w["top"] <= height * 0.10 or w["bottom"] >= height * 0.90:
                    for t in tokenize(tok):
                        band_only[t] += 1


            miss: Counter = Counter()
            hit = 0
            for tok, count in page_counts.items():
                # Consume from the remaining pool so a token is not credited twice.
                take = min(count, remaining.get(tok, 0))
                remaining[tok] -= take
                hit += take
                if count > take:
                    miss[tok] += count - take
                    unaccounted[tok] += count - take
            total += sum(page_counts.values())
            found += hit
            page_totals[pno] = sum(page_counts.values())
            if miss:
                page_missing[pno] = miss
            page.flush_cache()

    band_count = band_only  # occurrences inside header/footer bands, per token

    if not total:
        report.add("recall", BLOCKER, "PDF yielded no text; is it a scanned document?")
        return 0.0

    buckets: Counter = Counter()
    unexplained_detail: List[str] = []
    unexplained = 0
    unexplained_share: Dict[str, float] = {}
    for tok, count in unaccounted.items():
        split = _split_shortfall(tok, count, band_count.get(tok, 0), md_vocab)
        for kind, n in split.items():
            buckets[kind] += n
        if split["unexplained"]:
            unexplained += split["unexplained"]
            unexplained_detail.append(f"{tok} x{split['unexplained']}")
            unexplained_share[tok] = split["unexplained"] / count
    unexplained_detail.sort(key=lambda s: -int(s.rsplit("x", 1)[1]))

    per_page: Dict[int, Tuple[int, int]] = {}
    for pno, miss in page_missing.items():
        bad = round(sum(c * unexplained_share.get(t, 0.0) for t, c in miss.items()))
        if bad:
            per_page[pno] = (bad, page_totals[pno])

    recall = found / total
    unexplained_rate = unexplained / total

    report.metrics.update({
        "pdf_tokens": total,
        "tokens_found": found,
        "raw_recall": round(recall, 5),
        "band_vocabulary": len(band_count),
        "unaccounted_page_furniture": buckets.get("page_furniture", 0),
        "unaccounted_rejoined": buckets.get("rejoined", 0),
        "unaccounted_unexplained": unexplained,
        "unexplained_rate": round(unexplained_rate, 5),
        "pages_checked": len(pages),
        "pages_without_anchor": len(missing_pages),
        "pages_skipped_deliberately": len(re.findall(r"<!--\s*skipped\s+page=", markdown)),
    })

    if missing_pages:
        report.add("page_coverage", BLOCKER,
                   f"{len(missing_pages)} page(s) in the converted range carry text "
                   "but produced no content in the Markdown. Content is missing.",
                   missing_pages)

    if unexplained_rate > 0.02:
        report.add("recall", BLOCKER,
                   f"{unexplained_rate:.2%} of PDF text is unaccounted for "
                   f"({unexplained:,} tokens). Content is being dropped.",
                   unexplained_detail)
    elif unexplained_rate > 0.005:
        report.add("recall", WARNING,
                   f"{unexplained_rate:.2%} of PDF text unaccounted for.",
                   unexplained_detail)
    else:
        report.add("recall", INFO,
                   f"Recall {recall:.3%}; unexplained loss {unexplained_rate:.3%}.",
                   unexplained_detail)

    worst = sorted(per_page.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1))[:10]
    thin = [f"p{p}: {m}/{t} missing ({m/t:.0%})" for p, (m, t) in worst if m / t > 0.05]
    if thin:
        report.add("recall_pages", WARNING,
                   f"{len(thin)} page(s) lost more than 5% of their text.", thin)

    # Full marks below 0.1% unexplained, zero at 2%.
    return max(0.0, min(1.0, 1.0 - (unexplained_rate - 0.001) / 0.019))


# --------------------------------------------------------------------------- #
# 2. Structural invariants
# --------------------------------------------------------------------------- #
def check_table_markers(report: Report, markdown: str) -> float:
    opens = TABLE_OPEN_RE.findall(markdown)
    n_open = len(opens)
    n_close = markdown.count(TABLE_CLOSE)
    report.metrics["table_blocks"] = n_open

    if n_open != n_close:
        report.add("table_markers", BLOCKER,
                   f"Unbalanced table markers: {n_open} opened, {n_close} closed. "
                   "The chunker will mis-scope table boundaries.")
        return 0.0

    dupes = [t for t, c in Counter(opens).items() if c > 1]
    if dupes:
        report.add("table_markers", BLOCKER,
                   f"{len(dupes)} duplicate table id(s); the audit join to "
                   "coc_documents.tables would be ambiguous.", dupes)
        return 0.0

    # Nesting: no open before the previous close.
    depth = 0
    for m in re.finditer(r"<!--\s*table\s+id=|<!-- /table -->", markdown):
        depth += 1 if m.group(0).startswith("<!-- table") else -1
        if depth not in (0, 1):
            report.add("table_markers", BLOCKER, "Nested or crossed table markers.")
            return 0.0

    report.add("table_markers", INFO, f"{n_open} table blocks, all balanced.")
    return 1.0


def check_table_columns(report: Report, markdown: str) -> float:
    """Every row of a pipe table must have the same cell count as its header.

    A ragged table means the grid was mis-detected; downstream, values land
    under the wrong column labels, which is how a copay gets attributed to the
    wrong network tier.
    """
    blocks = re.findall(r"<!--\s*table\s+id=(\S+).*?-->\n(.*?)\n<!-- /table -->",
                        markdown, re.S)
    ragged: List[str] = []
    checked = 0
    for table_id, body in blocks:
        rows = [ln for ln in body.splitlines() if ln.strip().startswith("|")]
        if len(rows) < 2:
            continue
        checked += 1
        widths = [len(r.strip().strip("|").split("|")) for r in rows]
        expected = widths[0]
        bad = sum(1 for w in widths if w != expected)
        if bad:
            ragged.append(f"{table_id}: {bad}/{len(rows)} rows differ from header "
                          f"width {expected}")
    report.metrics["tables_checked"] = checked
    report.metrics["tables_ragged"] = len(ragged)

    if not checked:
        return 1.0
    rate = len(ragged) / checked
    if rate > 0.2:
        report.add("table_columns", BLOCKER,
                   f"{len(ragged)}/{checked} tables have inconsistent column "
                   "counts. Values are landing under the wrong headers.", ragged)
    elif ragged:
        report.add("table_columns", WARNING,
                   f"{len(ragged)}/{checked} tables have inconsistent column counts.",
                   ragged)
    else:
        report.add("table_columns", INFO, f"All {checked} tables are rectangular.")
    return max(0.0, 1.0 - rate * 2.5)


def check_heading_levels(report: Report, markdown: str) -> float:
    """Heading levels should not jump by more than one.

    An H1 followed directly by an H3 means a level was never detected, so every
    chunk beneath it carries a breadcrumb missing a rung.
    """
    levels = [(len(m.group(1)), m.group(2).strip()) for m in HEADING_RE.finditer(markdown)]
    report.metrics["headings"] = len(levels)
    if not levels:
        report.add("heading_levels", BLOCKER,
                   "No headings detected. Section context cannot be preserved; "
                   "lower heading_size_ratio or add a forced pattern.")
        return 0.0

    skips: List[str] = []
    prev = levels[0][0]
    for lvl, text in levels[1:]:
        if lvl > prev + 1:
            skips.append(f"H{prev} -> H{lvl}: {text[:60]}")
        prev = lvl
    report.metrics["heading_level_skips"] = len(skips)

    rate = len(skips) / len(levels)
    if rate > 0.15:
        report.add("heading_levels", WARNING,
                   f"{len(skips)} heading-level skips ({rate:.0%}). A level is "
                   "likely going undetected.", skips)
    elif skips:
        report.add("heading_levels", INFO, f"{len(skips)} heading-level skips.", skips)
    else:
        report.add("heading_levels", INFO, f"{len(levels)} headings, no level skips.")
    return max(0.0, 1.0 - rate * 3)


def check_page_anchors(report: Report, markdown: str) -> float:
    """Page anchors must not go backwards. An inversion means blocks were
    emitted out of reading order, so cited page numbers would be wrong."""
    pages = [int(p) for p in PAGE_ANCHOR_RE.findall(markdown)]
    report.metrics["page_anchors"] = len(pages)
    if not pages:
        report.add("page_anchors", BLOCKER,
                   "No page anchors; answers cannot cite a page.")
        return 0.0

    inversions = [f"...{pages[i-1]} -> {pages[i]}" for i in range(1, len(pages))
                  if pages[i] < pages[i - 1]]
    dupes = [p for p, c in Counter(pages).items() if c > 1]
    report.metrics["page_anchor_inversions"] = len(inversions)
    report.metrics["page_range"] = [min(pages), max(pages)]

    if inversions:
        report.add("page_anchors", BLOCKER,
                   f"{len(inversions)} page anchor inversion(s); blocks are out "
                   "of reading order and page citations would be wrong.", inversions)
        return 0.0
    if dupes:
        report.add("page_anchors", WARNING,
                   f"{len(dupes)} page number(s) emitted more than once.",
                   [str(d) for d in dupes[:10]])
        return 0.7
    report.add("page_anchors", INFO,
               f"{len(pages)} anchors, pages {min(pages)}-{max(pages)}, monotonic.")
    return 1.0


# --------------------------------------------------------------------------- #
# 3. Section-level checks (reused from inventory.py)
# --------------------------------------------------------------------------- #
def check_form_sections(report: Report, markdown: str) -> None:
    """Report the bound documents found, and whether citations are unambiguous.

    Informational rather than scored: a document that numbers straight through
    has no form codes and that is entirely correct. What matters is the
    combination -- restarting page numbers with no form code means a citation
    like "Schedule of Benefits, p. 12" cannot identify which bound document it
    refers to.
    """
    anchors = FORM_ANCHOR_RE.findall(markdown)
    if not anchors:
        report.metrics["form_sections"] = 0
        report.add("form_sections", INFO,
                   "No form codes found — treated as a single document numbered "
                   "straight through. If page numbers restart mid-document, "
                   "citations will not be unambiguous.")
        return

    spans: Dict[str, List[int]] = {}
    for code, printed in anchors:
        spans.setdefault(code, []).append(int(printed))

    report.metrics["form_sections"] = len(spans)
    report.metrics["form_codes"] = sorted(spans)
    detail = [f"{code}: printed {min(v)}-{max(v)}" for code, v in spans.items()]

    restarting = sum(1 for v in spans.values() if min(v) <= 2)
    report.add("form_sections", INFO,
               f"{len(spans)} bound document(s) detected; {restarting} restart "
               "page numbering. Citations carry the form code, so they stay "
               "unambiguous.", detail)

    pages = [int(p) for p in PAGE_ANCHOR_RE.findall(markdown)]
    covered = len(anchors)
    if pages and covered < len(pages) * 0.8:
        report.add("form_sections", WARNING,
                   f"Only {covered} of {len(pages)} pages carry a form code. "
                   "Chunks on the remaining pages cite by PDF page instead of "
                   "printed page.")


def check_sections(report: Report, markdown: str) -> float:
    rows = section_inventory(markdown)
    empty = [r for r in rows if r.blocks == 0]
    dupes = duplicate_breadcrumbs(rows)
    toc_like = suspected_toc_sections(rows, markdown)

    report.metrics.update({
        "sections": len(rows),
        "sections_empty": len(empty),
        "duplicate_breadcrumbs": len(dupes),
        "toc_like_sections": len(toc_like),
    })

    score = 1.0
    if dupes:
        score -= min(0.5, 0.15 * len(dupes))
        report.add("duplicate_breadcrumbs", WARNING,
                   f"{len(dupes)} section path(s) appear more than once. Retrieval "
                   "cannot distinguish them; disambiguate before embedding.",
                   [f"{k} ({', '.join(f'p{g.page_start}-{g.page_end}' for g in v)})"
                    for k, v in dupes])
    if toc_like:
        score -= min(0.3, 0.1 * len(toc_like))
        report.add("toc_like_sections", WARNING,
                   f"{len(toc_like)} section(s) look like a table of contents "
                   "(dot leaders). Index noise; consider excluding those pages.",
                   [r.breadcrumb for r in toc_like])
    if empty:
        score -= min(0.2, 0.02 * len(empty))
        report.add("empty_sections", INFO,
                   f"{len(empty)} heading(s) with no content beneath them.",
                   [r.breadcrumb for r in empty])
    return max(0.0, score)


def check_toc(report: Report, markdown: str, toc_path: str) -> Optional[float]:
    toc = parse_toc(pathlib.Path(toc_path).read_text(encoding="utf-8"))
    if not toc:
        report.add("toc_coverage", WARNING, "ToC file parsed to zero entries.")
        return None
    rows = toc_coverage(markdown, toc)
    missing = [r["toc_title"] for r in rows if not r["matched"]]
    report.metrics["toc_entries"] = len(rows)
    report.metrics["toc_matched"] = len(rows) - len(missing)

    # A contents list may run to hundreds of entries including sub-headings.
    # Missing "Section 2: Exclusions and Limitations" is serious; missing
    # "Selecting a Network Primary Care Physician" is not, and treating both as
    # blockers quarantines documents that are fine. Weight by whether the entry
    # names a division of the document.
    division_missing = [m for m in missing if DIVISION_ENTRY_RE.match(m)]
    rate = (len(rows) - len(missing)) / len(rows)
    report.metrics["toc_divisions_missing"] = len(division_missing)

    if division_missing:
        report.add("toc_coverage", BLOCKER,
                   f"{len(division_missing)} named division(s) in the contents were "
                   "not detected as headings. Their text is present but absorbed "
                   "into the preceding section, so those chunks carry the wrong "
                   "breadcrumb.", division_missing)
    elif rate < 0.90:
        # Not a blocker: contents lists vary enormously in depth. One document's
        # runs to 203 entries including benefit-table row labels ("Ambulance
        # Services"), which are rows rather than headings and are correctly
        # absent. Only a missing *division* means a section went astray.
        report.add("toc_coverage", WARNING,
                   f"{rate:.0%} of {len(rows)} contents entries matched a heading. "
                   "A detailed contents list includes sub-entries and table row "
                   "labels that are not headings; all named divisions were found.",
                   missing)
    elif missing:
        report.add("toc_coverage", WARNING,
                   f"{len(missing)} of {len(rows)} contents entries were not "
                   "detected as headings ({rate:.0%} matched). Sub-entries in a "
                   "contents list are often not headings in the body.".format(),
                   missing)
    else:
        report.add("toc_coverage", INFO, f"All {len(rows)} ToC entries matched.")
    return rate


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
#: Recall dominates because it is the only check the converter cannot fake:
#: everything else reads the Markdown and validates it against the converter's
#: own assumptions.
WEIGHTS = {
    "recall": 0.35,
    "table_markers": 0.10,
    "table_columns": 0.15,
    "heading_levels": 0.13,
    "page_anchors": 0.07,
    "sections": 0.10,
    "toc_coverage": 0.10,
}

PASS_SCORE = 90.0
REVIEW_SCORE = 70.0


def validate(markdown: str, pdf_path: Optional[str] = None,
             toc_path: Optional[str] = None) -> Report:
    report = Report(score=0.0, verdict="quarantine")
    parts: Dict[str, float] = {}

    if pdf_path:
        r = check_recall(report, markdown, pdf_path)
        if r is not None:
            parts["recall"] = r
    else:
        report.add("recall", WARNING,
                   "No PDF supplied. Structural checks alone cannot detect content "
                   "that never left the PDF -- they share assumptions with the "
                   "converter that produced this file.")

    parts["table_markers"] = check_table_markers(report, markdown)
    parts["table_columns"] = check_table_columns(report, markdown)
    parts["heading_levels"] = check_heading_levels(report, markdown)
    parts["page_anchors"] = check_page_anchors(report, markdown)
    parts["sections"] = check_sections(report, markdown)
    check_form_sections(report, markdown)

    if toc_path:
        t = check_toc(report, markdown, toc_path)
        if t is not None:
            parts["toc_coverage"] = t

    # Renormalise over the checks that actually ran, so a missing PDF or ToC
    # lowers confidence without silently inflating the score.
    total_weight = sum(WEIGHTS[k] for k in parts)
    report.score = round(
        100.0 * sum(parts[k] * WEIGHTS[k] for k in parts) / total_weight, 1
    ) if total_weight else 0.0
    report.components = {k: round(v, 3) for k, v in parts.items()}
    report.metrics["checks_run"] = sorted(parts)
    report.metrics["coverage_of_weights"] = round(total_weight, 2)

    if report.blockers:
        report.verdict = "quarantine"
    elif report.score >= PASS_SCORE and "recall" in parts:
        report.verdict = "pass"
    elif report.score >= REVIEW_SCORE:
        report.verdict = "review"
    else:
        report.verdict = "quarantine"

    # A structurally clean file that was never reconciled against its source is
    # not a pass -- it is unverified.
    if report.verdict == "pass" and "recall" not in parts:
        report.verdict = "review"

    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
SEV_MARK = {BLOCKER: "BLOCK", WARNING: "WARN ", INFO: "ok   "}
VERDICT_EXIT = {"pass": 0, "review": 1, "quarantine": 2}


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate converted COC Markdown.")
    ap.add_argument("markdown")
    ap.add_argument("--pdf", help="source PDF; enables recall reconciliation")
    ap.add_argument("--toc", help="table-of-contents text file")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--quiet", action="store_true", help="verdict line only")
    args = ap.parse_args()

    md = pathlib.Path(args.markdown).read_text(encoding="utf-8")
    report = validate(md, args.pdf, args.toc)

    if not args.quiet:
        print(f"\n=== VALIDATION: {pathlib.Path(args.markdown).name} ===\n")
        for f in report.findings:
            print(f"{SEV_MARK[f.severity]} [{f.check}] {f.message}")
            for d in f.detail[:6]:
                print(f"        - {d}")
            if len(f.detail) > 6:
                print(f"        ... {len(f.detail) - 6} more")
        print("\nComponents:")
        for k, v in sorted(report.components.items()):
            print(f"  {k:<16} {v:5.3f}  x weight {WEIGHTS[k]:.2f}")

    print(f"\nSCORE {report.score:.1f}/100   VERDICT: {report.verdict.upper()}"
          f"   blockers: {len(report.blockers)}")
    if report.verdict == "quarantine":
        print("Do not embed. Fix the conversion (parameters or code) and re-run;\n"
              "do not hand-edit the Markdown -- it is a derived artifact and the\n"
              "edit will be lost on the next conversion.")

    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"Report written to {args.json}")

    sys.exit(VERDICT_EXIT[report.verdict])


if __name__ == "__main__":
    main()
