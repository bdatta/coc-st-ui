"""
Audit a table's chunks: do the parts together represent the whole table?

A large table is split by rows across several chunks, with the header repeated on
each part. That is the point at which rows can silently go missing -- a dropped
part, an off-by-one in the row grouping, or a header that drifts between parts
would all still look plausible chunk by chunk. This reassembles the parts and
checks them against the table's recorded structure.

Sources:
  --doc-id / --table-id      read chunks from MongoDB Atlas
  --from-jsonl chunks.jsonl  read the Phase 2 preview export instead, so a table
                             can be checked before anything is embedded

    python tools/table_audit.py --doc-id hsa_spd
    python tools/table_audit.py --doc-id hsa_spd --table-id tbl-0002
    python tools/table_audit.py --doc-id hsa_spd --table-id tbl-0002 --show-text
    python tools/table_audit.py --from-jsonl hsa_chunks.jsonl --table-id tbl-0002
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SEPARATOR_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


# --------------------------------------------------------------------------- #
# Reassembly (pure, so it can be checked without a database)
# --------------------------------------------------------------------------- #
@dataclass
class TableAudit:
    table_id: str
    doc_id: str
    parts_found: int
    parts_expected: int
    header: str = ""
    #: Every data row across all parts, in order. Repeats are kept: a benefit
    #: grid legitimately repeats rows such as "Does the Annual Deductible
    #: Apply? | Yes | Yes" under each benefit, and discarding them would
    #: understate the table and hide genuinely missing rows.
    data_rows: List[str] = field(default_factory=list)
    unique_rows: int = 0
    repeated_rows: int = 0
    pages: str = ""
    breadcrumbs: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def split_pipe_block(text: str) -> tuple[List[str], List[str]]:
    """Return (header_lines, data_lines) from a chunk's pipe table."""
    rows = [ln.rstrip() for ln in text.splitlines() if PIPE_ROW_RE.match(ln)]
    if len(rows) >= 2 and SEPARATOR_RE.match(rows[1]):
        return rows[:2], rows[2:]
    return (rows[:1], rows[1:]) if rows else ([], [])


def _norm_row(row: str) -> str:
    return "|".join(c.strip() for c in row.strip().strip("|").split("|"))


def audit_table(chunks: Sequence[dict]) -> TableAudit:
    """Reassemble a table from its chunks and check completeness."""
    ordered = sorted(chunks, key=lambda c: int(c.get("part_index") or 1))
    first = ordered[0]
    expected = int(first.get("part_total") or 1)

    audit = TableAudit(
        table_id=str(first.get("table_id") or ""),
        doc_id=str(first.get("doc_id") or ""),
        parts_found=len(ordered),
        parts_expected=expected,
        breadcrumbs=sorted({c.get("breadcrumb", "") for c in ordered}),
    )

    indices = [int(c.get("part_index") or 1) for c in ordered]
    if sorted(indices) != list(range(1, expected + 1)):
        missing = sorted(set(range(1, expected + 1)) - set(indices))
        dupes = sorted({i for i in indices if indices.count(i) > 1})
        if missing:
            audit.problems.append(
                f"missing part(s) {missing} of {expected} — those rows are absent "
                "from the index entirely")
        if dupes:
            audit.problems.append(f"duplicate part(s) {dupes}")

    pages = [int(c["page_start"]) for c in ordered if c.get("page_start")] + \
            [int(c["page_end"]) for c in ordered if c.get("page_end")]
    if pages:
        audit.pages = f"{min(pages)}-{max(pages)}" if min(pages) != max(pages) else str(min(pages))

    headers: List[str] = []
    seen: Dict[str, int] = {}
    for chunk in ordered:
        header, data = split_pipe_block(chunk.get("text") or "")
        if header:
            headers.append(_norm_row(header[0]))
            if not audit.header:
                audit.header = header[0].strip()
        for row in data:
            key = _norm_row(row)
            if not key.replace("|", "").strip():
                continue
            audit.data_rows.append(row.strip())
            seen[key] = seen.get(key, 0) + 1
    audit.unique_rows = len(seen)
    audit.repeated_rows = sum(c - 1 for c in seen.values() if c > 1)

    if headers and len(set(headers)) > 1:
        audit.problems.append(
            f"header differs between parts ({len(set(headers))} variants) — rows in "
            "some parts would be read under the wrong column labels")
    elif headers and len(headers) < len(ordered):
        audit.problems.append(
            f"only {len(headers)} of {len(ordered)} parts carry a header row")

    if audit.repeated_rows:
        audit.notes.append(
            f"{audit.repeated_rows} repeated row(s) ({audit.unique_rows} distinct "
            "of {} total) — benefit grids repeat rows like \"Does the Annual "
            "Deductible Apply?\" under each benefit".format(len(audit.data_rows)))
    if len(audit.breadcrumbs) > 1:
        audit.notes.append(
            f"parts carry {len(audit.breadcrumbs)} different breadcrumbs")
    return audit


def compare_to_structure(audit: TableAudit, record: Optional[dict]) -> None:
    """Check the reassembled rows against coc_documents.tables[]."""
    if not record:
        audit.notes.append(
            "No table structure on file — completeness checked only across parts, "
            "not against the source grid. Store the tables JSON in coc_documents "
            "to enable this.")
        return

    n_rows = int(record.get("n_rows") or 0)
    header_rows = int(record.get("header_rows") or 1)
    caption_rows = int(record.get("caption_rows") or 0)
    expected_data = max(n_rows - header_rows - caption_rows, 0)
    found = len(audit.data_rows)

    if expected_data and found != expected_data:
        delta = found - expected_data
        audit.problems.append(
            f"row count mismatch: chunks hold {found} data rows, the extracted "
            f"grid has {expected_data} ({delta:+d})")
    elif expected_data:
        audit.notes.append(f"row count matches the grid ({found} data rows)")

    labels = record.get("column_labels") or []
    if labels and audit.header:
        header_cells = [c.strip() for c in audit.header.strip().strip("|").split("|")]
        if len(header_cells) != len(labels):
            audit.problems.append(
                f"column count mismatch: chunk header has {len(header_cells)}, "
                f"the grid has {len(labels)}")


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def load_from_jsonl(path: str) -> List[dict]:
    out = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_from_atlas(doc_id: str) -> tuple[List[dict], Dict[str, dict]]:
    from coc.config import load_settings
    from coc.vectorstore import DEFAULT_DOC_COLLECTION, get_collection

    settings = load_settings()
    chunks = get_collection(settings.mongodb_uri, settings.mongodb_db,
                            settings.mongodb_collection)
    rows = list(chunks.find(
        {"doc_id": doc_id, "type": "table"},
        {"_id": 0, "embedding": 0}))

    docs = get_collection(settings.mongodb_uri, settings.mongodb_db,
                          DEFAULT_DOC_COLLECTION)
    record = docs.find_one({"_id": doc_id}, {"tables": 1}) or {}
    structures = {t.get("table_id"): t for t in (record.get("tables") or [])}
    return rows, structures


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Audit a table's chunks for completeness.")
    ap.add_argument("--doc-id")
    ap.add_argument("--table-id")
    ap.add_argument("--from-jsonl", help="Phase 2 chunk export, instead of Atlas")
    ap.add_argument("--show-text", action="store_true",
                    help="print each part's embedded text")
    ap.add_argument("--reassembled", action="store_true",
                    help="print the reassembled table")
    args = ap.parse_args()

    structures: Dict[str, dict] = {}
    if args.from_jsonl:
        rows = [c for c in load_from_jsonl(args.from_jsonl) if c.get("type") == "table"]
        if args.doc_id:
            rows = [c for c in rows if c.get("doc_id") == args.doc_id]
    else:
        if not args.doc_id:
            print("--doc-id is required when reading from Atlas.")
            sys.exit(2)
        rows, structures = load_from_atlas(args.doc_id)

    if not rows:
        print("No table chunks found.")
        sys.exit(1)

    by_table: Dict[str, List[dict]] = {}
    for chunk in rows:
        by_table.setdefault(str(chunk.get("table_id") or "(none)"), []).append(chunk)

    if not args.table_id:
        print(f"\n{len(by_table)} table(s) in '{args.doc_id or args.from_jsonl}':\n")
        print(f"{'TABLE':<12}{'PARTS':<8}{'ROWS':<7}{'PAGES':<12}SECTION")
        for tid in sorted(by_table):
            audit = audit_table(by_table[tid])
            compare_to_structure(audit, structures.get(tid))
            mark = "  " if audit.ok else "!!"
            print(f"{mark}{tid:<10}{audit.parts_found}/{audit.parts_expected:<6}"
                  f"{len(audit.data_rows):<7}{audit.pages:<12}"
                  f"{(audit.breadcrumbs[0] if audit.breadcrumbs else '')[:44]}")
        bad = [t for t in by_table if not audit_table(by_table[t]).ok]
        print(f"\n{len(by_table) - len(bad)} of {len(by_table)} tables reassemble cleanly.")
        if bad:
            print(f"Check: {', '.join(bad)}")
        return

    chunks = by_table.get(args.table_id)
    if not chunks:
        print(f"No chunks for table '{args.table_id}'. Available: {', '.join(sorted(by_table))}")
        sys.exit(1)

    audit = audit_table(chunks)
    compare_to_structure(audit, structures.get(args.table_id))

    print(f"\n=== TABLE {audit.table_id}  (doc {audit.doc_id}) ===")
    print(f"   parts        : {audit.parts_found} of {audit.parts_expected}")
    print(f"   data rows    : {len(audit.data_rows)} "
          f"({audit.unique_rows} distinct)")
    print(f"   pages        : {audit.pages}")
    for bc in audit.breadcrumbs:
        print(f"   section      : {bc[:88]}")
    print(f"   header       : {audit.header[:88]}")

    print()
    for chunk in sorted(chunks, key=lambda c: int(c.get("part_index") or 1)):
        _, data = split_pipe_block(chunk.get("text") or "")
        print(f"   part {chunk.get('part_index')}/{chunk.get('part_total')}  "
              f"rows={len(data):<4} tokens={chunk.get('n_tokens')}  "
              f"id={str(chunk.get('chunk_id') or chunk.get('_id') or '')[:38]}")

    if audit.notes:
        print()
        for n in audit.notes:
            print(f"   note: {n}")
    if audit.problems:
        print()
        for p in audit.problems:
            print(f"   !! {p}")

    print(f"\n   VERDICT: {'COMPLETE' if audit.ok else 'REVIEW NEEDED'}")

    if args.reassembled:
        print("\n--- reassembled table ---")
        print(audit.header)
        for row in audit.data_rows:
            print(row)

    if args.show_text:
        for chunk in sorted(chunks, key=lambda c: int(c.get("part_index") or 1)):
            print(f"\n--- part {chunk.get('part_index')} embedded text ---")
            print(chunk.get("embed_text") or chunk.get("text") or "")


if __name__ == "__main__":
    main()
