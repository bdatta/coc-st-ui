"""
Convert a COC PDF to Markdown from the command line.

Exists because Streamlit caches imported modules: editing coc/*.py has no effect
until the process is fully stopped, so it is easy to convert with current code
loaded from a stale session and get Markdown that predates a fix. Running here
guarantees the code on disk is the code that runs.

Defaults match the settings verified against two real UnitedHealth documents:
bold and ALL-CAPS heading detection off, cover and contents skipped.

    python tools/convert.py coc.pdf                       # auto-detect start page
    python tools/convert.py coc.pdf -o out.md --first 3 --last 185
    python tools/convert.py coc.pdf --bold-headings       # opt back in
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from coc.pdf_to_markdown import ConversionOptions, convert_pdf  # noqa: E402


def write_toc(pdf, out_path: str) -> int:
    """Extract the document's first table of contents to a text file.

    validate.py reconciles that list against the headings the converter found,
    which is how a section absorbed into its predecessor gets noticed. Typing
    the list by hand is the reason that check often goes unused.
    """
    from coc.layout import detect_toc_pages

    pages = detect_toc_pages(pdf)
    if not pages:
        print("No table of contents detected; nothing written.")
        print("   Some documents have none, and some place it after the front "
              "matter\n   than this search covers. You can write toc.txt by hand: "
              "one entry\n   per line, page numbers and dot leaders optional.")
        return 0

    # Structural front matter is listed in the contents but is not plan content,
    # and is normally skipped by --first. Left in, each becomes a phantom
    # "missing section" when the conversion is validated.
    front_matter = re.compile(
        r"(title page|cover page|table of contents|contents|contact us|"
        r"how to (use|read) this|member service|nondiscrimination|"
        r"language assistance|notice of privacy)", re.IGNORECASE)

    lines, dropped = [], []
    for page in pages:
        for title, number in page.entries:
            if front_matter.search(title):
                dropped.append(title)
            else:
                lines.append(f"{title}...{number}")

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    span = ", ".join(str(p.page) for p in pages)
    print(f"\nTable of contents: {len(lines)} entries from PDF page(s) {span}")
    if dropped:
        print(f"   ({len(dropped)} front-matter entries omitted: "
              f"{', '.join(d[:28] for d in dropped[:3])}"
              f"{'...' if len(dropped) > 3 else ''})")
    for line in lines[:5]:
        print(f"   {line[:70]}")
    if len(lines) > 5:
        print(f"   ... {len(lines) - 5} more")
    print(f"Wrote {out}")
    print("   Review it — a contents list may include sub-entries that are not "
          "headings.\n   Then: python tools/validate.py <markdown> --pdf <pdf> "
          f"--toc {out}")
    return len(lines)


def page_inventory(pdf, limit: int = 16) -> None:
    """Show enough of the front of the document to choose a start page.

    Printed number, first line and word count per page. Front matter -- member
    forms, pharmacy inserts, notices, contents -- is usually obvious once the
    printed numbering is visible beside the content.
    """
    from coc.layout import _form_code_of, profile_page_language

    print(f"\n{'PDF':>4}  {'PRINTED':<9}{'WORDS':>6}  FIRST LINE")
    print("-" * 78)
    for idx in range(min(limit, len(pdf.pages))):
        page = pdf.pages[idx]
        text = page.extract_text() or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        height = float(page.height)
        footer = " ".join(w["text"] for w in page.extract_words()
                          if w["bottom"] >= height * 0.88)
        parsed = _form_code_of(footer)
        printed = ""
        if parsed:
            printed = str(parsed[1])
        else:
            tail = footer.split()[-1] if footer.split() else ""
            if tail.isdigit():
                printed = tail
        lang = profile_page_language(text, idx + 1)
        mark = "  [foreign]" if lang.foreign else ""
        words = len(text.split())
        print(f"{idx + 1:>4}  {printed:<9}{words:>6}  "
              f"{(lines[0][:52] if lines else '(no text)')}{mark}")
        try:
            page.flush_cache()
        except Exception:
            pass
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a COC PDF to Markdown.")
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", help="output .md (default: alongside the PDF)")
    ap.add_argument("--first", type=int, help="first PDF page (default: auto-detect)")
    ap.add_argument("--last", type=int, help="last PDF page (default: end)")
    ap.add_argument("--bold-headings", action="store_true",
                    help="treat bold short lines as headings (off by default: it "
                         "promotes bold cross-references such as 'Section 1: "
                         "Covered Health Care Services for details.')")
    ap.add_argument("--allcaps-headings", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="print a page inventory and exit without converting")
    ap.add_argument("--toc-out", nargs="?", const="toc.txt", default=None,
                    metavar="PATH",
                    help="with --inspect, write the detected table of contents "
                         "(default toc.txt). Feed it to validate.py --toc to "
                         "check that every listed section survived conversion")
    ap.add_argument("--min-confidence", type=float, default=0.90,
                    help="stop for inspection when page-number detection is less "
                         "confident than this (default 0.90). Low confidence "
                         "usually means numbering restarts, i.e. a compilation")
    ap.add_argument("--accept-detected", action="store_true",
                    help="use the detected start page even at low confidence")
    ap.add_argument("--keep-foreign-pages", action="store_true",
                    help="keep language-assistance pages (skipped by default: "
                         "translated notices carry no plan content but match "
                         "weakly against almost any query)")
    ap.add_argument("--tables-json", help="also write the cell-level table export")
    args = ap.parse_args()

    pdf_path = pathlib.Path(args.pdf)
    if not pdf_path.exists():
        print(f"No such file: {pdf_path}")
        sys.exit(2)

    import pdfplumber

    from coc.layout import detect_page_numbering

    first, last = args.first, args.last

    # Always resolve both ends. An earlier version only filled in `last` on the
    # auto-detect path, so passing --first alone produced a (first, None) range
    # and failed inside the converter.
    with pdfplumber.open(str(pdf_path)) as probe:
        total = len(probe.pages)
        found = detect_page_numbering(probe) if first is None else None

        if args.inspect:
            print(f"{pdf_path.name}: {total} pages")
            if found is not None and found.found:
                print(f"Printed page 1 looks like PDF page "
                      f"{found.first_numbered_page} "
                      f"(offset +{found.offset}, {found.confidence:.0%} confidence)")
            page_inventory(probe)
            if args.toc_out:
                write_toc(probe, args.toc_out)
            print("\nRe-run with --first N once you have chosen a start page.")
            sys.exit(0)

        # Stop rather than guess. On a compilation the detector locks onto the
        # largest consistently numbered block, which is often deep inside the
        # document -- accepting it silently skips everything before it.
        low = (found is not None and found.found
               and found.confidence < args.min_confidence)
        none_found = found is not None and not found.found
        if (low or none_found) and not args.accept_detected:
            print(f"\n{pdf_path.name}: {total} pages")
            if low:
                print(f"Page-number detection is only {found.confidence:.0%} "
                      f"confident (threshold {args.min_confidence:.0%}).")
                print(f"It suggests PDF page {found.first_numbered_page}, but low "
                      "confidence usually means the numbering\nrestarts — a "
                      "compilation of bound documents — and the detector locks "
                      "onto the\nlargest block rather than the start of content.")
            else:
                print("No consistent page numbering found.")
            page_inventory(probe)
            print("Choose a start page and re-run, for example:")
            print(f"   python tools/convert.py \"{pdf_path}\" -o out.md --first N")
            print("\nOr accept the detected value with --accept-detected, "
                  "or inspect further with --inspect.")
            sys.exit(3)

    last = last or total
    if first is not None:
        print(f"Converting PDF pages {first}-{last} of {total}")
    else:
        if found is not None and found.found:
            first = found.first_numbered_page
            first = found.first_numbered_page
            print(f"Detected printed page 1 at PDF page {first} "
                  f"(offset +{found.offset}, {found.confidence:.0%} confidence)")
            if found.confidence < 0.90:
                print("   !! Confidence is low, which usually means page numbering "
                      "restarts\n      (a compilation of bound documents). Check the "
                      "start page and\n      pass --first explicitly if this is wrong.")
        else:
            first = 1
            print("No consistent page numbering found; starting at page 1.")

    options = ConversionOptions(
        page_range=(first or 1, last or total),
        treat_bold_as_heading=args.bold_headings,
        treat_allcaps_as_heading=args.allcaps_headings,
        skip_foreign_pages=not args.keep_foreign_pages,
    )

    started = time.time()
    result = convert_pdf(str(pdf_path), options,
                         progress=lambda f, m: print(f"\r{m:<60}", end="", flush=True),
                         doc_title=pdf_path.name)
    print(f"\rConverted in {time.time() - started:.1f}s{' ' * 40}")

    out = pathlib.Path(args.out) if args.out else pdf_path.with_suffix(".md")
    # Create the output directory rather than failing on a path the caller
    # clearly intended: "-o build/coc.md" means "put it in build".
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.markdown, encoding="utf-8")

    for key in ("pages", "headings", "tables", "tables_spanning_pages",
                "form_sections", "running_lines_removed"):
        value = result.stats.get(key)
        if isinstance(value, list):
            value = len(value)
        print(f"   {key:<24} {value}")
    for warning in result.warnings:
        print(f"   !! {warning}")

    if args.tables_json:
        pathlib.Path(args.tables_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.tables_json).write_text(
            json.dumps(result.tables_json, indent=2), encoding="utf-8")
        print(f"   table structures -> {args.tables_json}")

    print(f"\nWrote {out}")
    print(f"Next:  python tools/validate.py \"{out}\" --pdf \"{pdf_path}\"")


if __name__ == "__main__":
    main()
