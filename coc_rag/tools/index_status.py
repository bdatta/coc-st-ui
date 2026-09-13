"""Report Atlas search index readiness and filter coverage.

    python tools/index_status.py
"""
from __future__ import annotations

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
    s = load_settings()
    coll = get_collection(s.mongodb_uri, s.mongodb_db, s.mongodb_collection)
    print(f"collection : {s.mongodb_db}.{s.mongodb_collection}")
    print(f"documents  : {coll.estimated_document_count():,}\n")

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

    if "--json" in sys.argv:
        print(json.dumps(indexes, indent=2, default=str))


if __name__ == "__main__":
    main()
