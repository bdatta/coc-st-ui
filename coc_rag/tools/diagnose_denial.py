"""
Diagnose a denial-support lookup outside Streamlit.

Reports which code fixes are loaded, what the Markdown looks like, and the full
ranked result -- so a wrong answer can be traced to stale code, stale Markdown,
or genuine scoring in one run.

    python tools/diagnose_denial.py coc.md E0240 "bath chair" "bath chair,bath seat"
    python tools/diagnose_denial.py coc.md S4025 "Donor services for IVF" \\
        --diagnosis Z31.9 --diagnosis-desc "Encounter for procreative management"

Batch mode runs a whole case file and, where an expected outcome is given,
reports pass/fail. That file is also your evaluation set: it is the only way to
tell whether a change to expansions, reranking or thresholds actually helped
rather than just felt better.

    python tools/diagnose_denial.py coc.md --batch cases.csv
    python tools/diagnose_denial.py coc.md --batch cases.csv --verbose

Against Atlas, or both paths side by side:

    python tools/diagnose_denial.py --batch cases.csv --atlas --doc-id acme_2026
    python tools/diagnose_denial.py coc.md --batch cases.csv --compare --doc-id acme_2026

`--compare` runs each case through the Markdown and the Atlas path and flags
disagreements. The two paths have diverged before -- scoring on different scales
gave `supported` on one and `not_found` on the other for the same code -- and a
reviewer should never get a different verdict depending on which source was
selected.

CSV columns (only `code` and `descriptor` are required):

    code,descriptor,diagnosis_code,diagnosis_descriptor,synonyms,expected

`synonyms` is pipe- or semicolon-separated. `expected` is supported, weak or
not_found.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def run_case(index, md: str, code: str, descriptor: str,
             diagnosis_code: str = "", diagnosis_desc: str = "",
             synonyms: str = "") -> dict:
    """Score one case against an already-built index."""
    from coc.denial_support import assess, build_core_terms, build_query_terms

    descriptors = [d for d in (descriptor, diagnosis_desc) if d and d.strip()]
    syn = [s.strip() for s in re.split(r"[|;,\n]", synonyms or "") if s.strip()]
    terms = build_query_terms(descriptors, syn)
    core = build_core_terms(descriptors, syn)
    provisions = index_search(index, terms, descriptors + syn, core)
    codes = [c for c in (code, diagnosis_code) if c]
    result = assess(codes, descriptors, terms, provisions, idf=index.idf)
    return {
        "code": code,
        "diagnosis": diagnosis_code,
        "outcome": result.outcome,
        "score": provisions[0].score if provisions else 0.0,
        "provision": provisions[0].text if provisions else "",
        "citation": provisions[0].citation if provisions else "",
        "provisions": provisions,
        "result": result,
    }


def index_search(index, terms, descriptors, core):
    return index.search(terms, limit=6, descriptors=descriptors, core_terms=core)


class AtlasSearcher:
    """Runs the Atlas path with the same signature as the Markdown index."""

    def __init__(self, doc_id: str, scopes=None):
        from coc.config import load_settings
        from coc.embeddings import get_embedder
        from coc.vectorstore import get_collection

        settings = load_settings()
        missing = [n for n, v in (("MONGODB_URI", settings.mongodb_uri),
                                  ("embedding API key",
                                   settings.voyage_api_key or settings.openai_api_key))
                   if not v]
        if missing:
            raise RuntimeError(f"missing from the environment: {', '.join(missing)}")

        key = (settings.voyage_api_key if settings.embed_provider == "voyage"
               else settings.openai_api_key)
        self.embedder = get_embedder(settings.embed_provider, settings.embed_model,
                                     int(settings.embed_dim), api_key=key)
        self.coll = get_collection(settings.mongodb_uri, settings.mongodb_db,
                                   settings.mongodb_collection)
        self.doc_id = doc_id
        self.scopes = scopes
        self.idf: dict = {}
        self.settings = settings
        self.preflight()

    def preflight(self) -> None:
        """Explain an empty result set before running any case.

        A misconfigured Atlas query returns nothing rather than failing, so
        every case scores 0.00 with no provision and the report looks like a
        relevance problem. These are the four things that actually cause it.
        """
        from coc.denial_support import DEFAULT_SCOPES, scope_of
        from coc.vectorstore import list_documents, list_search_indexes

        problems = []

        docs = list_documents(self.coll)
        ids = [d["doc_id"] for d in docs]
        if not docs:
            problems.append("The collection is empty — nothing has been embedded.")
        elif self.doc_id and self.doc_id not in ids:
            problems.append(
                f"No document with doc_id '{self.doc_id}'. Indexed: "
                f"{', '.join(ids[:6])}"
                + (" ..." if len(ids) > 6 else ""))

        indexes = list_search_indexes(self.coll)
        vector = next((i for i in indexes
                       if i.get("name") == self.settings.vector_index), None)
        if vector is None:
            problems.append(
                f"No vector index named '{self.settings.vector_index}'. "
                f"Found: {[i.get('name') for i in indexes] or 'none'}")
        else:
            if not (vector.get("queryable") or vector.get("status") == "READY"):
                problems.append(
                    f"Index '{self.settings.vector_index}' is not queryable yet "
                    f"(status {vector.get('status')}); queries return nothing "
                    "until it is READY.")
            definition = vector.get("latestDefinition") or vector.get("definition") or {}
            dims = next((f.get("numDimensions") for f in definition.get("fields", [])
                         if f.get("type") == "vector"), None)
            if dims and int(dims) != int(self.settings.embed_dim):
                problems.append(
                    f"Index was built for {dims} dimensions, configuration uses "
                    f"{self.settings.embed_dim}. A mismatched index returns "
                    "nothing rather than erroring — rebuild with "
                    "tools/embed.py --recreate-index.")

        # Are any chunks in a searchable section at all?
        query = {"doc_id": self.doc_id} if self.doc_id else {}
        scopes = self.scopes or DEFAULT_SCOPES
        in_scope = 0
        sample = []
        try:
            for row in self.coll.find(query, {"_id": 0, "breadcrumb": 1}).limit(4000):
                bc = row.get("breadcrumb", "")
                if scope_of(bc) in scopes:
                    in_scope += 1
                elif len(sample) < 3 and bc:
                    sample.append(bc)
        except Exception:
            pass
        if docs and in_scope == 0:
            problems.append(
                "No chunks fall in an exclusion / limitation / definition "
                "section. The breadcrumbs may predate the current conversion. "
                + (f"Examples: {sample[0][:70]}" if sample else ""))

        if problems:
            print("\n!! Atlas preflight found problems:\n")
            for item in problems:
                print(f"   - {item}")
            print("\n   Every case will report not_found until these are "
                  "resolved.\n   Check: python tools/index_status.py\n")
        else:
            print(f"   preflight ok: {in_scope} chunk(s) in a searchable section")

    def search(self, terms, limit=6, descriptors=(), core_terms=()):
        from coc.denial_support import DEFAULT_SCOPES, search_provisions_vector

        self.idf = {}
        return search_provisions_vector(
            self.coll, self.embedder, " ".join(descriptors), terms,
            doc_id=self.doc_id or None, scopes=self.scopes or DEFAULT_SCOPES,
            limit=limit, descriptors=descriptors, core_terms=core_terms,
            idf_out=self.idf)


def _load_cases(csv_path: str) -> list:
    import csv as _csv

    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in _csv.DictReader(fh):
            if (row.get("code") or "").strip():
                rows.append(row)
    return rows


def run_compare(md: str, csv_path: str, doc_id: str, verbose: bool) -> int:
    """Run every case through both paths and report disagreements."""
    from coc.denial_support import MarkdownProvisionIndex

    rows = _load_cases(csv_path)
    if not rows:
        print("No cases with a 'code' column.")
        return 2

    lexical = MarkdownProvisionIndex(md)
    try:
        atlas = AtlasSearcher(doc_id)
    except Exception as exc:
        print(f"Atlas unavailable: {exc}")
        return 2

    header = (f"{'':<3}{'CODE':<9}{'MARKDOWN':<12}{'ATLAS':<12}"
              f"{'MD SCORE':>9}{'ATLAS':>8}  TOP PROVISION (markdown)")
    print(f"\n{len(rows)} case(s), both paths\n")
    print(header)
    print("-" * len(header))

    disagreements = []
    for row in rows:
        args = ((row.get("code") or "").strip(),
                (row.get("descriptor") or "").strip(),
                (row.get("diagnosis_code") or "").strip(),
                (row.get("diagnosis_descriptor") or "").strip(),
                (row.get("synonyms") or "").strip())
        a = run_case(lexical, md, *args)
        b = run_case(atlas, md, *args)
        same = a["outcome"] == b["outcome"]
        mark = "  " if same else "!!"
        if not same:
            disagreements.append((a["code"], a["outcome"], b["outcome"]))
        print(f"{mark:<3}{a['code']:<9}{a['outcome']:<12}{b['outcome']:<12}"
              f"{a['score']:>9.2f}{b['score']:>8.2f}  {a['provision'][:40]}")
        if verbose and not same:
            print(f"      markdown: {a['provision'][:78]}")
            print(f"      atlas   : {b['provision'][:78]}")

    if disagreements:
        print(f"\n!! {len(disagreements)} disagreement(s) between paths:")
        for code, md_out, at_out in disagreements:
            print(f"   {code}: markdown={md_out}, atlas={at_out}")
        print("\n   A reviewer should not get a different verdict depending on "
              "which\n   source they picked. Common causes: Atlas chunks predate "
              "the current\n   conversion, or the two paths score differently.")
        return 1
    print(f"\nBoth paths agree on all {len(rows)} case(s).")
    return 0


def parse_scopes(raw: str):
    from coc.denial_support import DEFAULT_SCOPES

    if not raw:
        return DEFAULT_SCOPES
    chosen = tuple(x.strip().lower() for x in raw.split(",") if x.strip())
    unknown = [x for x in chosen if x not in DEFAULT_SCOPES]
    if unknown:
        print(f"Unknown scope(s): {unknown}. Valid: {', '.join(DEFAULT_SCOPES)}")
        sys.exit(2)
    return chosen


def run_batch(index, md: str, csv_path: str, verbose: bool, label: str = "") -> int:
    rows = _load_cases(csv_path)
    if not rows:
        print("No cases with a 'code' column.")
        return 2

    scope = (f"{len(index.units)} citable sentences" if hasattr(index, "units")
             else f"Atlas document '{getattr(index, 'doc_id', '')}'")
    print(f"\n{len(rows)} case(s) against {scope}{label}\n")
    header = f"{'':<3}{'CODE':<9}{'DX':<9}{'OUTCOME':<11}{'SCORE':>7}  PROVISION"
    print(header)
    print("-" * len(header))

    checked = passed = 0
    failures = []
    for row in rows:
        case = run_case(index, md,
                        (row.get("code") or "").strip(),
                        (row.get("descriptor") or "").strip(),
                        (row.get("diagnosis_code") or "").strip(),
                        (row.get("diagnosis_descriptor") or "").strip(),
                        (row.get("synonyms") or "").strip())
        expected = (row.get("expected") or "").strip().lower()
        mark = "  "
        if expected:
            checked += 1
            if expected == case["outcome"]:
                passed += 1
                mark = "ok"
            else:
                mark = "!!"
                failures.append((case["code"], expected, case["outcome"]))
        print(f"{mark:<3}{case['code']:<9}{case['diagnosis']:<9}"
              f"{case['outcome']:<11}{case['score']:>7.2f}  {case['provision'][:52]}")
        if verbose and case["provisions"]:
            print(f"      {case['citation'][:96]}")
            for note in case["result"].notes[:2]:
                print(f"      note: {note[:92]}")

    if checked:
        print(f"\n{passed}/{checked} matched the expected outcome.")
        for code, want, got in failures:
            print(f"   {code}: expected {want}, got {got}")
    else:
        print("\nNo 'expected' column — add one to turn this into a regression check.")
    return 0 if not failures else 1


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("markdown", nargs="?")
    ap.add_argument("code", nargs="?")
    ap.add_argument("descriptor", nargs="?")
    ap.add_argument("synonyms", nargs="?", default="")
    ap.add_argument("--diagnosis", default="")
    ap.add_argument("--diagnosis-desc", default="")
    ap.add_argument("--batch")
    ap.add_argument("--atlas", action="store_true",
                    help="run against the Atlas index instead of a Markdown file")
    ap.add_argument("--compare", action="store_true",
                    help="run both paths and flag disagreements")
    ap.add_argument("--doc-id", default="",
                    help="scope Atlas searches to one indexed document")
    ap.add_argument("--scopes", default="",
                    help="comma-separated subset of exclusion,limitation,definition "
                         "(default: all three). Definitions are searched because "
                         "an exclusion often turns on a defined term, but they are "
                         "damped and can never be reported as 'supported' alone.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    args = ap.parse_args()

    if args.help or (not args.batch and not (args.code and args.descriptor)):
        print(__doc__)
        sys.exit(0 if args.help else 2)

    if args.batch:
        if args.compare:
            if not args.markdown:
                print("--compare needs the Markdown file as well as --doc-id.")
                sys.exit(2)
            md = pathlib.Path(args.markdown).read_text(encoding="utf-8")
            sys.exit(run_compare(md, args.batch, args.doc_id, args.verbose))

        if args.atlas:
            try:
                index = AtlasSearcher(args.doc_id, parse_scopes(args.scopes))
            except Exception as exc:
                print(f"Atlas unavailable: {exc}")
                sys.exit(2)
            label = f" (doc_id={args.doc_id})" if args.doc_id else " (all documents)"
            sys.exit(run_batch(index, "", args.batch, args.verbose, label))

        if not args.markdown:
            print("Give a Markdown file, or use --atlas.")
            sys.exit(2)
        md = pathlib.Path(args.markdown).read_text(encoding="utf-8")
        from coc.denial_support import MarkdownProvisionIndex

        sys.exit(run_batch(MarkdownProvisionIndex(md, scopes=parse_scopes(args.scopes)),
                           md, args.batch, args.verbose))

    md_path, code, descriptor = args.markdown, args.code, args.descriptor
    synonyms = args.synonyms
    diagnosis_code, diagnosis_desc = args.diagnosis, args.diagnosis_desc

    print("=" * 78)
    print("1. CODE BUILD MARKERS  (each fix in the order it shipped)")
    print("=" * 78)
    checks = []
    try:
        from coc import layout

        checks.append(("bullet fix (private-use bullets)",
                       bool(layout.BULLET_RE.match("\uf0a7 Car seats."))))
        checks.append(("division fix (Section N forced to H1)",
                       "structurally top level" in (layout.heading_level.__doc__ or "")
                       or "named division" in (layout.__doc__ or "")
                       or "KEYWORD_RE.match(text) and len(text) < 120" in
                       pathlib.Path(layout.__file__).read_text()))
        checks.append(("form-code detection", hasattr(layout, "detect_form_sections")))
        checks.append(("page-numbering detection", hasattr(layout, "detect_page_numbering")))
    except Exception as exc:
        print("  !! could not import coc.layout:", exc)
    try:
        from coc import denial_support as ds

        checks.append(("definition damping", hasattr(ds, "DEFINITION_WEIGHT")))
        checks.append(("core-term gating", hasattr(ds, "build_core_terms")))
        checks.append(("limit-language gate", hasattr(ds, "is_citable_provision")))
        checks.append(("lead-in context chain", "lead_in" in ds.Provision.__annotations__))
    except Exception as exc:
        print("  !! could not import coc.denial_support:", exc)
    for name, ok in checks:
        print(f"   {'OK  ' if ok else 'MISS'}  {name}")
    if not all(ok for _, ok in checks):
        print("\n   -> Replace the whole coc/ folder from the zip and rerun.")

    print()
    print("=" * 78)
    print("2. MARKDOWN MARKERS")
    print("=" * 78)
    if args.atlas:
        try:
            index = AtlasSearcher(args.doc_id, parse_scopes(args.scopes))
        except Exception as exc:
            print(f"Atlas unavailable: {exc}")
            sys.exit(2)
        case = run_case(index, "", code, descriptor, diagnosis_code,
                        diagnosis_desc, synonyms)
        print(f"\nOUTCOME: {case['outcome'].upper()}   score {case['score']:.2f}")
        for i, prov in enumerate(case["provisions"][:5], start=1):
            print(f"\n   {i}. [{prov.score:7.2f}] {prov.source_type} "
                  f"matched={prov.matched_terms}")
            print(f"      {prov.citation[:88]}")
            print(f"      {prov.text[:96]}")
        for note in case["result"].notes:
            print(f"\n   note: {note[:110]}")
        sys.exit(0)

    md = pathlib.Path(md_path).read_text(encoding="utf-8")
    pua = sum(1 for c in md if 0xE000 <= ord(c) <= 0xF8FF)
    nested = len(re.findall(r"^  - ", md, re.M))
    sections_h1 = re.findall(r"^# (Section \d+[^\n]*)$", md, re.M)
    sections_deep = re.findall(r"^#{2,6} (Section \d+[^\n]*)$", md, re.M)
    print(f"   private-use bullet chars : {pua}   (want 0 -- bullet fix applied)")
    print(f"   nested list items '  - ' : {nested}   (want > 0 -- nesting preserved)")
    print(f"   'Section N' at H1        : {len(sections_h1)}")
    print(f"   'Section N' deeper than H1: {len(sections_deep)}   (want 0 -- division fix)")
    if sections_deep:
        print("   -> Markdown predates the division fix. Re-run Phase 1.")
        for s in sections_deep[:4]:
            print(f"      {s[:64]}")

    print()
    print("=" * 78)
    print("3. SEARCH RESULT")
    print("=" * 78)
    from coc.denial_support import (
        MarkdownProvisionIndex,
        assess,
        build_core_terms,
        build_query_terms,
        format_citation_block,
    )

    desc = [d for d in (descriptor, diagnosis_desc) if d and d.strip()]
    syn = [s.strip() for s in re.split(r"[|;,\n]", synonyms) if s.strip()]
    terms = build_query_terms(desc, syn)
    core = build_core_terms(desc, syn)
    index = MarkdownProvisionIndex(md, scopes=parse_scopes(args.scopes))
    provisions = index.search(terms, limit=6, descriptors=desc + syn, core_terms=core)
    result = assess([c for c in (code, diagnosis_code) if c], desc, terms,
                    provisions, idf=index.idf)

    print(f"   citable units : {len(index.units)}")
    print(f"   query terms   : {terms}")
    print(f"   core terms    : {core}")
    print(f"   OUTCOME       : {result.outcome.upper()}\n")
    for i, p in enumerate(provisions, start=1):
        print(f"   {i}. [{p.score:7.2f}] {p.source_type:<11} matched={p.matched_terms}")
        print(f"      {p.breadcrumb[:88]}")
        print(f"      {p.text[:96]}")
    print()
    print(format_citation_block(result))


if __name__ == "__main__":
    main()
