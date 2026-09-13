"""
Chunk a converted Markdown file from the command line.

Completes the CLI path: convert -> validate -> chunk -> audit, with no Streamlit
in the loop. The JSONL it writes is the same shape Phase 2 stores in
`coc_chunks`, so a table can be audited and chunk sizing reviewed before any
embedding is paid for.

    python tools/chunk.py coc.md -o coc_chunks.jsonl
    python tools/chunk.py coc.md -o out.jsonl --doc-id hsa_spd_2025 \\
        --plan-id UHG-HSA --plan-year 2025 --carrier "UnitedHealth Group"
    python tools/chunk.py coc.md --stats-only
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from coc.chunker import (  # noqa: E402
    ChunkOptions,
    DocumentContext,
    chunk_markdown,
    sha256_file,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk converted COC Markdown.")
    ap.add_argument("markdown")
    ap.add_argument("-o", "--out", help="output .jsonl (default: alongside the .md)")
    ap.add_argument("--doc-id", help="default: the Markdown file stem")
    ap.add_argument("--pdf", help="source PDF, to record source_sha256")

    ap.add_argument("--plan-id")
    ap.add_argument("--group-number")
    ap.add_argument("--plan-year", type=int)
    ap.add_argument("--carrier")
    ap.add_argument("--effective-date", help="ISO date, e.g. 2026-01-01")
    ap.add_argument("--market-segment")
    ap.add_argument("--states", help="comma-separated, e.g. IL,IN")

    ap.add_argument("--target-tokens", type=int, default=650)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--overlap-tokens", type=int, default=90)
    ap.add_argument("--max-table-tokens", type=int, default=1100)
    ap.add_argument("--no-row-sentences", action="store_true")
    ap.add_argument("--stats-only", action="store_true",
                    help="report without writing the JSONL")
    args = ap.parse_args()

    md_path = pathlib.Path(args.markdown)
    if not md_path.exists():
        print(f"No such file: {md_path}")
        sys.exit(2)

    doc_id = args.doc_id or md_path.stem.replace(" ", "_").lower()
    context = DocumentContext(
        doc_id=doc_id,
        plan_id=args.plan_id,
        group_number=args.group_number,
        plan_year=args.plan_year,
        effective_date=args.effective_date,
        carrier=args.carrier,
        market_segment=args.market_segment,
        states=[s.strip() for s in (args.states or "").split(",") if s.strip()],
        source_filename=pathlib.Path(args.pdf).name if args.pdf else md_path.name,
        source_sha256=sha256_file(args.pdf) if args.pdf else None,
    )

    options = ChunkOptions(
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        max_table_tokens=args.max_table_tokens,
        include_row_sentences=not args.no_row_sentences,
    )

    chunks = chunk_markdown(md_path.read_text(encoding="utf-8"), context, options)
    if not chunks:
        print("No chunks produced — is this a converted COC Markdown file?")
        sys.exit(1)

    tokens = sorted(c.n_tokens for c in chunks)
    tables = [c for c in chunks if c.type == "table"]
    by_table = collections.Counter(c.table_id for c in tables if c.table_id)
    split = {t: n for t, n in by_table.items() if n > 1}

    print(f"\ndoc_id            {doc_id}")
    print(f"chunks            {len(chunks)}  "
          f"({len(chunks) - len(tables)} text, {len(tables)} table)")
    print(f"tokens            median {tokens[len(tokens) // 2]}, "
          f"p90 {tokens[int(len(tokens) * 0.9)]}, max {tokens[-1]}")
    print(f"total tokens      {sum(tokens):,}  (one-time embedding cost)")
    print(f"distinct tables   {len(by_table)}  ({len(split)} split across parts)")
    print(f"flagged review    {sum(1 for c in chunks if c.needs_review)}")
    print(f"sections          {len({c.breadcrumb for c in chunks})}")
    forms = {c.form_code for c in chunks if c.form_code}
    if forms:
        print(f"form codes        {len(forms)}  ({', '.join(sorted(forms)[:3])}...)")
    est_mb = len(chunks) * 8 / 1024
    print(f"atlas storage     ~{est_mb:.1f} MB at 1024 dims")

    if split:
        print("\nsplit tables (audit these):")
        for table_id, parts in sorted(split.items()):
            print(f"   {table_id}  {parts} parts")

    if args.stats_only:
        return

    out = pathlib.Path(args.out) if args.out else md_path.with_suffix(".chunks.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_doc(), ensure_ascii=False, default=str) + "\n")
    print(f"\nWrote {out}")
    if split:
        first = sorted(split)[0]
        print(f"Next:  python tools/table_audit.py --from-jsonl \"{out}\" "
              f"--table-id {first}")


if __name__ == "__main__":
    main()
