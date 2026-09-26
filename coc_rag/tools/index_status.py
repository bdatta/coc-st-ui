"""Report Atlas search index readiness and filter coverage.

    python tools/index_status.py
    python tools/index_status.py --json
    python tools/index_status.py --create-text-index --wait

Creating the lexical index needs no reprocessing: Atlas builds it over the
documents already stored, so nothing is re-chunked or re-embedded.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from coc.config import load_settings  # noqa: E402
from coc.vectorstore import (  # noqa: E402
    FILTER_FIELDS,
    get_collection,
    list_search_indexes,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Atlas search index status.")
    ap.add_argument("--json", action="store_true",
                    help="also dump the raw index definitions")
    ap.add_argument("--create-text-index", action="store_true",
                    help="create the lexical index for hybrid search over the "
                         "documents already stored (nothing is re-embedded)")
    ap.add_argument("--wait", action="store_true",
                    help="with --create-text-index, wait until it is queryable")
    args = ap.parse_args()

    s = load_settings()
    coll = get_collection(s.mongodb_uri, s.mongodb_db, s.mongodb_collection)
    print(f"collection : {s.mongodb_db}.{s.mongodb_collection}")
    print(f"documents  : {coll.estimated_document_count():,}\n")

    if args.create_text_index:
        from coc.vectorstore import ensure_text_index, wait_until_queryable

        state = ensure_text_index(coll, s.text_index)
        print(f"lexical index '{s.text_index}': {state}\n")
        if state == "exists":
            print("   Already present. Nothing to do.\n")
        else:
            print("   Building over the documents already stored. Atlas usually")
            print("   needs a minute or two before it answers queries.\n")
            if args.wait:
                ready = wait_until_queryable(coll, s.text_index, timeout=600,
                                             progress=lambda m: print(f"   {m}"))
                print(f"   {'queryable' if ready else 'still building'}\n")

    indexes = list_search_indexes(coll)
    if not indexes:
        print("No search indexes. Run Phase 2, or create one in the Atlas UI.")
        return

    for idx in indexes:
        name = idx.get("name")
        status = idx.get("status", "?")
        queryable = idx.get("queryable", False)
        mark = "READY" if queryable else "NOT READY"
        print(f"[{mark}] {name}  type={idx.get('type')}  status={status}")

        definition = idx.get("latestDefinition") or idx.get("definition") or {}
        fields = definition.get("fields") or []
        vector = [f for f in fields if f.get("type") == "vector"]
        filters = sorted(f.get("path") for f in fields if f.get("type") == "filter")

        for v in vector:
            print(f"    vector path={v.get('path')} dims={v.get('numDimensions')} "
                  f"similarity={v.get('similarity')}")
            if int(v.get("numDimensions") or 0) != int(s.embed_dim):
                print(f"    !! numDimensions {v.get('numDimensions')} != EMBED_DIM "
                      f"{s.embed_dim}. Queries will return nothing.")
        if filters:
            print(f"    filters: {', '.join(filters)}")
            if name == s.vector_index:
                missing = sorted(set(FILTER_FIELDS) - set(filters))
                if missing:
                    # Atlas does not add filter paths retroactively; an index
                    # built before a field existed cannot filter on it.
                    print(f"    !! MISSING filter path(s): {', '.join(missing)}")
                    print("       Add them in the Atlas UI, or drop the index and "
                          "re-run Phase 2.")
                else:
                    print("    all expected filter paths present")
        if not queryable:
            print("    still building — queries return empty until this says READY")
        print()

    if args.json:
        print(json.dumps(indexes, indent=2, default=str))


if __name__ == "__main__":
    main()
