"""
MongoDB Atlas vector store.

Free-tier (M0) realities this module is built around:
  * 512 MB total storage. A 200-page COC at 1024 dims lands around 10-20 MB, so
    you have plenty of room -- but drop to 512 dims if you plan to load many
    documents.
  * 100 operations/second. Writes are batched and optionally throttled.
  * No `allowDiskUse`, and aggregation pipelines are capped at 50 stages. The
    pipelines here are short, so neither bites.
  * Search index count on free clusters is small -- the hybrid path adds a
    second index, so enable it deliberately.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from pymongo import ASCENDING, MongoClient, ReplaceOne
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

DEFAULT_VECTOR_INDEX = "coc_vector_index"
DEFAULT_TEXT_INDEX = "coc_text_index"
DEFAULT_CHUNK_COLLECTION = "coc_chunks"
DEFAULT_DOC_COLLECTION = "coc_documents"
DEFAULT_QUERY_LOG_COLLECTION = "coc_query_log"
EMBEDDING_PATH = "embedding"

#: Declared as `filter` fields on the vector index so they can be applied as a
#: pre-filter inside $vectorSearch rather than after the fact. plan_id and
#: plan_year matter most: answering from the wrong plan or the wrong plan year is
#: the highest-consequence failure this system has.
FILTER_FIELDS = [
    "doc_id",
    "plan_id",
    "plan_year",
    "form_code",
    "type",
    "section_path",
    "page_start",
    "needs_review",
]

#: Fields stored as BSON dates rather than strings, so range queries work.
DATE_FIELDS = ("created_at", "updated_at", "embedded_at",
               "effective_date", "termination_date", "ingested_at")

#: `embedding` is deliberately excluded everywhere -- at 1024 dims it is ~8 KB
#: per document, and pulling it back on every read is pure waste.
PROJECTION = {
    "_id": 0,
    "chunk_id": "$_id",
    "doc_id": 1,
    "seq": 1,
    "type": 1,
    "heading": 1,
    "breadcrumb": 1,
    "section_path": 1,
    "text": 1,
    "embed_text": 1,
    "page_start": 1,
    "page_end": 1,
    "table_id": 1,
    "part_index": 1,
    "part_total": 1,
    "column_labels": 1,
    "has_spans": 1,
    "form_code": 1,
    "printed_page_start": 1,
    "printed_page_end": 1,
    "plan_id": 1,
    "plan_year": 1,
    "carrier": 1,
    "needs_review": 1,
}


def get_collection(uri: str, db_name: str, collection_name: str) -> Collection:
    client = MongoClient(uri, appname="coc-rag", serverSelectionTimeoutMS=20000)
    client.admin.command("ping")
    return client[db_name][collection_name]


def _to_datetime(value):
    """Coerce ISO strings to BSON dates. Chunks keep ISO strings so they stay
    JSON-serializable; conversion happens only at the storage boundary."""
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        for parse in (datetime.fromisoformat,):
            try:
                return parse(text)
            except ValueError:
                continue
    return value


def coerce_dates(doc: dict) -> dict:
    out = dict(doc)
    for field_name in DATE_FIELDS:
        if field_name in out:
            out[field_name] = _to_datetime(out[field_name])
    return out


# --------------------------------------------------------------------------- #
# Index management
# --------------------------------------------------------------------------- #
def list_search_indexes(coll: Collection) -> List[dict]:
    try:
        return list(coll.list_search_indexes())
    except Exception:
        return []


def index_state(coll: Collection, name: str) -> Optional[dict]:
    for idx in list_search_indexes(coll):
        if idx.get("name") == name:
            return idx
    return None


def ensure_vector_index(
    coll: Collection,
    dim: int,
    name: str = DEFAULT_VECTOR_INDEX,
    similarity: str = "cosine",
    filter_fields: Sequence[str] = tuple(FILTER_FIELDS),
) -> str:
    existing = index_state(coll, name)
    if existing:
        return "exists"

    # Atlas refuses to create a search index on a namespace that does not exist
    # yet, and on a first run the index is created before any chunk is written.
    # Creating the (empty) collection first makes the order irrelevant.
    try:
        coll.database.create_collection(coll.name)
    except Exception:
        pass  # already exists, or created concurrently

    definition = {
        "fields": [
            {
                "type": "vector",
                "path": EMBEDDING_PATH,
                "numDimensions": int(dim),
                "similarity": similarity,
            }
        ]
        + [{"type": "filter", "path": f} for f in filter_fields]
    }
    coll.create_search_index(
        SearchIndexModel(definition=definition, name=name, type="vectorSearch")
    )
    return "created"


def ensure_text_index(coll: Collection, name: str = DEFAULT_TEXT_INDEX) -> str:
    """Lexical index for hybrid retrieval. Optional -- costs one search index."""
    if index_state(coll, name):
        return "exists"
    try:
        coll.database.create_collection(coll.name)
    except Exception:
        pass
    definition = {
        "mappings": {
            "dynamic": False,
            "fields": {
                "embed_text": {"type": "string"},
                "breadcrumb": {"type": "string"},
                "doc_id": {"type": "token"},
            },
        }
    }
    coll.create_search_index(SearchIndexModel(definition=definition, name=name, type="search"))
    return "created"


def wait_until_queryable(coll: Collection, name: str, timeout: float = 300.0,
                         progress: Optional[Callable[[str], None]] = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        idx = index_state(coll, name)
        if idx and (idx.get("queryable") or idx.get("status") == "READY"):
            return True
        if progress:
            progress(f"Waiting for index '{name}' ({(idx or {}).get('status', 'PENDING')})...")
        time.sleep(5)
    return False


def drop_search_index(coll: Collection, name: str) -> None:
    try:
        coll.drop_search_index(name)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
def upsert_chunks(
    coll: Collection,
    docs: Sequence[dict],
    batch_size: int = 100,
    throttle: float = 0.4,
    progress: Optional[Callable[[float, str], None]] = None,
) -> int:
    written = 0
    total = max(len(docs), 1)
    for start in range(0, len(docs), batch_size):
        batch = [coerce_dates(d) for d in docs[start : start + batch_size]]
        ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in batch]
        res = coll.bulk_write(ops, ordered=False)
        written += (res.upserted_count or 0) + (res.modified_count or 0)
        if progress:
            progress(min((start + len(batch)) / total, 1.0),
                     f"Stored {start + len(batch)}/{len(docs)} chunks")
        if throttle:
            time.sleep(throttle)  # respect the free-tier 100 ops/sec ceiling
    ensure_btree_indexes(coll)
    return written


def ensure_btree_indexes(coll: Collection) -> None:
    """No unique index on chunk_id -- it *is* _id now, so uniqueness is free."""
    coll.create_index([("doc_id", ASCENDING), ("seq", ASCENDING)])
    coll.create_index([("plan_id", ASCENDING), ("plan_year", ASCENDING)])
    coll.create_index([("doc_id", ASCENDING), ("table_id", ASCENDING),
                       ("part_index", ASCENDING)])
    coll.create_index([("needs_review", ASCENDING)])
    coll.create_index([("pipeline_version", ASCENDING),
                       ("chunk_config_hash", ASCENDING)])


def attach_embeddings(
    chunk_docs: Sequence[dict],
    vectors: Sequence[Sequence[float]],
    model: str,
) -> List[dict]:
    """Stamp vectors plus lifecycle metadata onto chunk documents.

    `embedded_at` and `embed_model` are what let you find rows to re-embed after
    a model change, instead of re-embedding the whole corpus."""
    stamped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: List[dict] = []
    for doc, vec in zip(chunk_docs, vectors):
        d = dict(doc)
        d["embedding"] = list(vec)
        d["embed_model"] = model
        d["embed_dim"] = len(vec)
        d["embedded_at"] = stamped_at
        d["updated_at"] = stamped_at
        out.append(d)
    return out


def delete_document(coll: Collection, doc_id: str) -> int:
    return coll.delete_many({"doc_id": doc_id}).deleted_count


def chunks_needing_review(coll: Collection, doc_id: Optional[str] = None,
                          limit: int = 100) -> List[dict]:
    match: Dict = {"needs_review": True}
    if doc_id:
        match["doc_id"] = doc_id
    pipeline = [
        {"$match": match},
        {"$sort": {"doc_id": 1, "seq": 1}},
        {"$limit": int(limit)},
        {"$project": {**PROJECTION, "extraction_notes": 1}},
    ]
    return list(coll.aggregate(pipeline))


def stale_chunks(coll: Collection, pipeline_version: str,
                 config_hash: Optional[str] = None) -> int:
    """Count chunks produced by an older pipeline or a different chunk config."""
    clauses: List[Dict] = [{"pipeline_version": {"$ne": pipeline_version}}]
    if config_hash:
        clauses.append({"chunk_config_hash": {"$ne": config_hash}})
    return coll.count_documents({"$or": clauses})


def list_documents(coll: Collection) -> List[dict]:
    pipeline = [
        {"$group": {
            "_id": "$doc_id",
            "plan_id": {"$first": "$plan_id"},
            "plan_year": {"$first": "$plan_year"},
            "chunks": {"$sum": 1},
            "pages": {"$max": "$page_end"},
            "tables": {"$sum": {"$cond": [{"$eq": ["$type", "table"]}, 1, 0]}},
            "review": {"$sum": {"$cond": ["$needs_review", 1, 0]}},
        }},
        {"$sort": {"_id": 1}},
    ]
    return [
        {"doc_id": r["_id"], "plan_id": r.get("plan_id"), "plan_year": r.get("plan_year"),
         "chunks": r["chunks"], "pages": r["pages"], "tables": r["tables"],
         "needs_review": r.get("review", 0)}
        for r in coll.aggregate(pipeline)
    ]


# --------------------------------------------------------------------------- #
# Documents collection (one record per COC)
# --------------------------------------------------------------------------- #
def upsert_document_record(coll: Collection, record: dict) -> str:
    """Store the converted Markdown, table structures, and conversion settings.

    This is what makes "Markdown as a checkpoint" real: you can re-chunk with
    different parameters without re-parsing 200 pages, and the audit artifact
    lives next to the chunks it produced.

    Watch the 16 MB BSON document limit. A 200-page COC is typically 1-3 MB of
    Markdown, so there is ample headroom -- but if you approach it, gzip the
    markdown and store it as BinData instead.
    """
    doc = coerce_dates(record)
    doc.setdefault("ingested_at", datetime.now(timezone.utc))
    _id = doc.pop("doc_id", None) or doc.get("_id")
    doc["_id"] = _id
    coll.replace_one({"_id": _id}, doc, upsert=True)
    coll.create_index([("plan_id", ASCENDING), ("plan_year", ASCENDING)])
    coll.create_index([("source_sha256", ASCENDING)])
    return _id


def get_document_record(coll: Collection, doc_id: str,
                        include_markdown: bool = False) -> Optional[dict]:
    projection = None if include_markdown else {"markdown": 0, "tables": 0}
    return coll.find_one({"_id": doc_id}, projection)


def find_by_source_hash(coll: Collection, sha256: str) -> Optional[dict]:
    """Recognise a re-upload of a document already ingested."""
    return coll.find_one({"source_sha256": sha256}, {"markdown": 0, "tables": 0})


def log_query(coll: Collection, record: dict) -> None:
    """Optional. Retrieval logs are the raw material for the evaluation set --
    real questions, what was retrieved, and whether the answer held up."""
    doc = coerce_dates(record)
    doc.setdefault("asked_at", datetime.now(timezone.utc))
    coll.insert_one(doc)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def vector_search(
    coll: Collection,
    query_vector: Sequence[float],
    k: int = 8,
    num_candidates: Optional[int] = None,
    index_name: str = DEFAULT_VECTOR_INDEX,
    filters: Optional[Dict] = None,
) -> List[dict]:
    stage = {
        "index": index_name,
        "path": EMBEDDING_PATH,
        "queryVector": list(query_vector),
        "numCandidates": int(num_candidates or max(k * 20, 100)),
        "limit": int(k),
    }
    if filters:
        stage["filter"] = filters
    pipeline = [
        {"$vectorSearch": stage},
        {"$project": {**PROJECTION, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(coll.aggregate(pipeline))


def text_search(
    coll: Collection,
    query: str,
    k: int = 8,
    index_name: str = DEFAULT_TEXT_INDEX,
    doc_id: Optional[str] = None,
) -> List[dict]:
    must = [{"text": {"query": query, "path": ["embed_text", "breadcrumb"]}}]
    compound: Dict = {"must": must}
    if doc_id:
        compound["filter"] = [{"equals": {"path": "doc_id", "value": doc_id}}]
    pipeline = [
        {"$search": {"index": index_name, "compound": compound}},
        {"$limit": int(k)},
        {"$project": {**PROJECTION, "score": {"$meta": "searchScore"}}},
    ]
    return list(coll.aggregate(pipeline))


def reciprocal_rank_fusion(rankings: Sequence[Sequence[dict]], k: int = 60,
                           limit: int = 8) -> List[dict]:
    """Combine result lists client-side. Avoids depending on server-side
    $rankFusion availability, which varies by cluster tier and version."""
    scores: Dict[str, float] = {}
    docs: Dict[str, dict] = {}
    for results in rankings:
        for rank, doc in enumerate(results, start=1):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            docs.setdefault(cid, doc)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    out = []
    for cid, score in ordered:
        doc = dict(docs[cid])
        doc["score"] = round(score, 6)
        out.append(doc)
    return out


def hybrid_search(
    coll: Collection,
    query: str,
    query_vector: Sequence[float],
    k: int = 8,
    doc_id: Optional[str] = None,
    vector_index: str = DEFAULT_VECTOR_INDEX,
    text_index: str = DEFAULT_TEXT_INDEX,
) -> List[dict]:
    filters = {"doc_id": {"$eq": doc_id}} if doc_id else None
    dense = vector_search(coll, query_vector, k=k * 2, index_name=vector_index, filters=filters)
    try:
        sparse = text_search(coll, query, k=k * 2, index_name=text_index, doc_id=doc_id)
    except Exception:
        sparse = []
    if not sparse:
        return dense[:k]
    return reciprocal_rank_fusion([dense, sparse], limit=k)


def fetch_neighbors(coll: Collection, doc_id: str, seq: int, window: int = 1) -> List[dict]:
    """Pull adjacent chunks so an answer can see the sentence that ran over a
    chunk boundary."""
    pipeline = [
        {"$match": {"doc_id": doc_id, "seq": {"$gte": seq - window, "$lte": seq + window}}},
        {"$sort": {"seq": 1}},
        {"$project": PROJECTION},
    ]
    return list(coll.aggregate(pipeline))
