"""Generate sample MongoDB documents for each collection, from real chunker output."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coc.chunker import ChunkOptions, DocumentContext, chunk_markdown, sha256_bytes

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema")
os.makedirs(OUT, exist_ok=True)

MD = """<!-- document: acme_ppo_2026_coc.pdf -->
<!-- page: 11 -->
# SECTION 4 - COVERED SERVICES

This section describes the services covered under the Plan. All services are \
subject to Medical Necessity review and, where indicated, prior authorization. \
Benefits are payable only for services received while coverage is in force.

## Outpatient Services
### Physician Office Visits

You may select any Participating Provider without a referral. Benefits for \
services received from a Non-Participating Provider are payable at the \
Out-of-Network level shown in the Schedule of Benefits.

<!-- table id=tbl-0007 pages=12-13 rows=5 cols=3 spans=yes -->
| Covered Service | Cost Share - In-Network | Cost Share - Out-of-Network |
| --- | --- | --- |
| Office Visits | $25 copay | 40% after deductible |
| Specialist Visits | $40 copay | 50% after deductible |
| Urgent Care | $75 copay | $75 copay |
<!-- /table -->

<!-- page: 14 -->
## Ambiguous Grid

<!-- table id=tbl-0008 pages=14 rows=4 cols=4 spans=no -->
| A | B | C | D |
| --- | --- | --- | --- |
| Chiropractic |  | 20 visits |  |
|  |  | per year |  |
|  |  |  |  |
<!-- /table -->
"""

SOURCE_SHA = sha256_bytes(b"acme ppo 2026 coc pdf bytes")

ctx = DocumentContext(
    doc_id="acme_ppo_2026",
    plan_id="ACME-PPO-2000",
    group_number="0084512",
    plan_year=2026,
    effective_date="2026-01-01",
    termination_date=None,
    carrier="Acme Health",
    market_segment="large_group",
    states=["IL", "IN"],
    source_filename="acme_ppo_2026_coc.pdf",
    source_sha256=SOURCE_SHA,
)
opts = ChunkOptions()
chunks = chunk_markdown(MD, ctx, opts)

# A short stand-in vector. Real vectors are 1024 floats for voyage-3.5.
FAKE_VEC = [0.0231, -0.0417, 0.0688, 0.0129, -0.0553, 0.0342, -0.0106, 0.0475]
STAMPED = "2026-08-15T14:32:11Z"
CREATED = "2026-08-15T14:28:40Z"


def as_date(value):
    """MongoDB Extended JSON v2 requires a full ISO-8601 instant."""
    if not value:
        return None
    text = str(value)
    if len(text) == 10:                      # date-only, e.g. 2026-01-01
        text += "T00:00:00Z"
    text = text.replace("+00:00", "Z")
    if not text.endswith("Z") and "+" not in text[10:]:
        text += "Z"
    return {"$date": text}


def to_extended(chunk, vector):
    d = chunk.to_doc()
    d["embedding"] = list(vector)
    d["embed_model"] = "voyage-3.5"
    d["embed_dim"] = 1024
    d["created_at"] = CREATED
    d["embedded_at"] = STAMPED
    d["updated_at"] = STAMPED
    for f in ("created_at", "updated_at", "embedded_at", "effective_date", "termination_date"):
        d[f] = as_date(d.get(f))
    # keep _id first for readability
    ordered = {"_id": d.pop("_id")}
    ordered.update(d)
    return ordered


chunk_docs = [to_extended(c, FAKE_VEC) for c in chunks]

with open(f"{OUT}/coc_chunks.sample.json", "w") as fh:
    json.dump(chunk_docs, fh, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- documents --
document_doc = {
    "_id": "acme_ppo_2026",
    "title": "Acme Health PPO 2000 - Certificate of Coverage (2026)",

    "plan_id": "ACME-PPO-2000",
    "group_number": "0084512",
    "plan_year": 2026,
    "effective_date": {"$date": "2026-01-01T00:00:00Z"},
    "termination_date": None,
    "carrier": "Acme Health",
    "market_segment": "large_group",
    "states": ["IL", "IN"],

    "source_filename": "acme_ppo_2026_coc.pdf",
    "source_sha256": SOURCE_SHA,
    "source_pages": 203,
    "pipeline_version": "1.1.0",

    "markdown": "<!-- document: acme_ppo_2026_coc.pdf -->\n<!-- page: 1 -->\n\n"
                "# ACME HEALTH PPO 2000\n\n... full converted Markdown, typically "
                "1-3 MB for a 200-page COC ...",

    "tables": [
        {
            "table_id": "tbl-0007",
            "pages": [12, 13],
            "n_rows": 5,
            "n_cols": 3,
            "header_rows": 2,
            "has_spans": True,
            "column_labels": [
                "Covered Service",
                "Cost Share - In-Network",
                "Cost Share - Out-of-Network",
            ],
            "cells": [
                {"row": 0, "col": 0, "rowspan": 2, "colspan": 1, "text": "Covered Service"},
                {"row": 0, "col": 1, "rowspan": 1, "colspan": 2, "text": "Cost Share"},
                {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "text": "In-Network"},
                {"row": 1, "col": 2, "rowspan": 1, "colspan": 1, "text": "Out-of-Network"},
                {"row": 2, "col": 0, "rowspan": 1, "colspan": 1, "text": "Office Visits"},
                {"row": 2, "col": 1, "rowspan": 1, "colspan": 1, "text": "$25 copay"},
                {"row": 2, "col": 2, "rowspan": 1, "colspan": 1, "text": "40% after deductible"}
            ],
        }
    ],

    "conversion_options": {
        "table_strategy": "auto",
        "table_format": "pipe",
        "heading_size_ratio": 1.08,
        "treat_allcaps_as_heading": True,
        "treat_bold_as_heading": True,
        "drop_running_lines": True,
        "header_band": 0.1,
        "stitch_tables": True,
        "min_table_fill": 0.15,
        "emit_row_sentences": False,
        "force_heading_patterns": ["^SECTION\\s+\\d+", "^ARTICLE\\s+[IVX\\d]+"],
    },
    "conversion_stats": {
        "pages": 203,
        "headings": 412,
        "tables": 87,
        "tables_with_spans": 34,
        "tables_spanning_pages": 12,
        "body_font_size": 9.5,
        "heading_font_sizes": [16.0, 13.0, 11.0],
        "running_lines_removed": 4,
        "characters": 1_284_119,
    },

    "chunk_options": {
        "target_tokens": 650,
        "max_tokens": 900,
        "overlap_tokens": 90,
        "max_table_tokens": 1100,
        "min_chunk_tokens": 25,
        "include_row_sentences": True,
        "prefix_breadcrumb": True,
        "table_review_empty_ratio": 0.35,
    },
    "tables_complete": True,
    "chunk_config_hash": opts.config_hash(),
    "n_chunks": 1146,
    "n_chunks_needing_review": 9,

    "embed_model": "voyage-3.5",
    "embed_dim": 1024,

    "status": "indexed",
    "ingested_at": {"$date": "2026-08-15T14:28:02Z"},
    "updated_at": {"$date": STAMPED},
}

with open(f"{OUT}/coc_documents.sample.json", "w") as fh:
    json.dump([document_doc], fh, indent=2, ensure_ascii=False)

# --------------------------------------------------------------- query log --
query_log = [
    {
        "_id": "q_01J8XA2K9F3D",
        "question": "What is the copay for a specialist visit out-of-network?",
        "doc_id": "acme_ppo_2026",
        "plan_id": "ACME-PPO-2000",
        "plan_year": 2026,
        "retrieval": {
            "mode": "vector",
            "k": 8,
            "num_candidates": 160,
            "used_rerank": False,
            "embed_model": "voyage-3.5",
            "filters": {"doc_id": "acme_ppo_2026"},
        },
        "retrieved": [
            {"chunk_id": "acme_ppo_2026#00001-3408a8cb", "score": 0.8412, "rank": 1,
             "type": "table", "breadcrumb": "SECTION 4 - COVERED SERVICES > Outpatient Services > Physician Office Visits"},
            {"chunk_id": "acme_ppo_2026#00000-9c1e77f4", "score": 0.7318, "rank": 2,
             "type": "text", "breadcrumb": "SECTION 4 - COVERED SERVICES"}
        ],
        "answer": "Specialist visits are covered at 50% after the deductible when "
                  "you use an Out-of-Network provider [1].",
        "answer_model": "claude-sonnet-4-6",
        "cited_chunk_ids": ["acme_ppo_2026#00001-3408a8cb"],
        "latency_ms": {"embed": 118, "search": 74, "generate": 2140},
        "feedback": {"rating": "up", "expected_pages": [12, 13], "note": None},
        "asked_at": {"$date": "2026-08-15T15:02:44Z"},
        "user_id": "svc-rep-0142",
    }
]

with open(f"{OUT}/coc_query_log.sample.json", "w") as fh:
    json.dump(query_log, fh, indent=2, ensure_ascii=False)

# ------------------------------------------------------------ index configs --
indexes = {
    "coc_chunks": {
        "atlas_search_indexes": [
            {
                "name": "coc_vector_index",
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {"type": "vector", "path": "embedding",
                         "numDimensions": 1024, "similarity": "cosine"},
                        {"type": "filter", "path": "doc_id"},
                        {"type": "filter", "path": "plan_id"},
                        {"type": "filter", "path": "plan_year"},
                        {"type": "filter", "path": "type"},
                        {"type": "filter", "path": "section_path"},
                        {"type": "filter", "path": "page_start"},
                        {"type": "filter", "path": "needs_review"}
                    ]
                },
            },
            {
                "name": "coc_text_index",
                "type": "search",
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "embed_text": {"type": "string"},
                            "breadcrumb": {"type": "string"},
                            "column_labels": {"type": "string"},
                            "doc_id": {"type": "token"},
                            "plan_id": {"type": "token"}
                        },
                    }
                },
            },
        ],
        "btree_indexes": [
            {"keys": {"doc_id": 1, "seq": 1}},
            {"keys": {"plan_id": 1, "plan_year": 1}},
            {"keys": {"doc_id": 1, "table_id": 1, "part_index": 1}},
            {"keys": {"needs_review": 1}},
            {"keys": {"pipeline_version": 1, "chunk_config_hash": 1}}
        ],
    },
    "coc_documents": {
        "btree_indexes": [
            {"keys": {"plan_id": 1, "plan_year": 1}},
            {"keys": {"source_sha256": 1}}
        ]
    },
    "coc_query_log": {
        "btree_indexes": [
            {"keys": {"asked_at": -1}},
            {"keys": {"doc_id": 1, "asked_at": -1}},
            {"keys": {"feedback.rating": 1}}
        ]
    },
}

with open(f"{OUT}/search_indexes.json", "w") as fh:
    json.dump(indexes, fh, indent=2)

print(f"wrote {len(chunk_docs)} chunk samples to {OUT}")
for f in sorted(os.listdir(OUT)):
    print("  ", f)
