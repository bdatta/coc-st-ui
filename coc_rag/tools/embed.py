"""
Embed chunks and store them in MongoDB Atlas, from the command line.

Resumable by design: before embedding, the chunk IDs already present in the
collection for this document are read back and skipped. A run interrupted by a
network fault, an exhausted socket pool, or a rate limit can simply be repeated
without paying to embed the same text twice.

    python tools/embed.py coc.chunks.jsonl --doc-id acme_2026
    python tools/embed.py coc.chunks.jsonl --doc-id acme_2026 --dry-run
    python tools/embed.py coc.chunks.jsonl --doc-id acme_2026 \\
        --markdown coc.md --tables coc.tables.json
    python tools/embed.py coc.chunks.jsonl --config pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from coc.pipeline_config import build_config  # noqa: E402


def load_chunks(path: str, doc_id: Optional[str]) -> List[dict]:
    rows: List[dict] = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if doc_id:
            row["doc_id"] = doc_id
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed chunks and store them in Atlas.")
    ap.add_argument("chunks", help="JSONL from tools/chunk.py")
    ap.add_argument("--config", help="pipeline YAML")
    ap.add_argument("--doc-id")
    ap.add_argument("--markdown", help="store this Markdown in coc_documents")
    ap.add_argument("--tables", help="store this tables JSON in coc_documents")
    ap.add_argument("--embed-provider")
    ap.add_argument("--embed-model")
    ap.add_argument("--embed-dim", type=int)
    ap.add_argument("--collection")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen, embed nothing")
    ap.add_argument("--no-replace", action="store_true",
                    help="keep existing chunks (default replaces them)")
    ap.add_argument("--resume", action="store_true",
                    help="skip chunks already stored (implies --no-replace)")
    ap.add_argument("--create-text-index", action="store_true")
    args = ap.parse_args()

    cfg = build_config(args.config, {
        "doc_id": args.doc_id,
        "embed_provider": args.embed_provider,
        "embed_model": args.embed_model,
        "embed_dim": args.embed_dim,
        "mongodb_collection": args.collection,
        "create_text_index": True if args.create_text_index else None,
        "replace_existing": False if (args.no_replace or args.resume) else None,
    })

    chunks = load_chunks(args.chunks, cfg.doc_id)
    if not chunks:
        print("No chunks in that file.")
        sys.exit(1)
    doc_id = cfg.doc_id or chunks[0].get("doc_id")
    if not doc_id:
        print("No doc_id: pass --doc-id or set it in the YAML.")
        sys.exit(2)

    total_tokens = sum(int(c.get("n_tokens") or 0) for c in chunks)
    print(f"\ndocument      {doc_id}")
    print(f"chunks        {len(chunks)}")
    print(f"tokens        {total_tokens:,}")
    print(f"provider      {cfg.embed_provider} / {cfg.embed_model} @ {cfg.embed_dim} dims")
    print(f"destination   {cfg.mongodb_db}.{cfg.mongodb_collection}")
    print(f"storage       ~{len(chunks) * cfg.embed_dim * 8 / 1_048_576:.1f} MB of vectors")

    missing = cfg.missing_secrets(need_embedding=True)
    if missing:
        print(f"\n!! Missing from the environment: {', '.join(missing)}")
        print("   Put them in .env — they are deliberately not read from YAML.")
        if not args.dry_run:
            sys.exit(2)

    if args.dry_run:
        print("\nDry run: nothing embedded or written.")
        return

    from coc.embeddings import get_embedder
    from coc.vectorstore import (
        DEFAULT_DOC_COLLECTION,
        attach_embeddings,
        delete_document,
        ensure_text_index,
        ensure_vector_index,
        get_collection,
        index_state,
        list_documents,
        upsert_chunks,
        upsert_document_record,
    )

    coll = get_collection(cfg.mongodb_uri, cfg.mongodb_db, cfg.mongodb_collection)

    print("\n-- indexes --")
    state = ensure_vector_index(coll, cfg.embed_dim, name=cfg.vector_index)
    print(f"   vector index '{cfg.vector_index}': {state}")
    if cfg.create_text_index:
        print(f"   lexical index '{cfg.text_index}': "
              f"{ensure_text_index(coll, cfg.text_index)}")

    if cfg.replace_existing and not args.resume:
        removed = delete_document(coll, doc_id)
        print(f"   removed {removed} existing chunk(s) for '{doc_id}'")

    pending = chunks
    if args.resume:
        existing = {d["_id"] for d in coll.find({"doc_id": doc_id}, {"_id": 1})}
        pending = [c for c in chunks if c.get("_id") not in existing]
        print(f"   resuming: {len(existing)} already stored, {len(pending)} to do")
        if not pending:
            print("\nNothing left to embed.")
            return

    embedder = get_embedder(cfg.embed_provider, cfg.embed_model, cfg.embed_dim,
                            api_key=cfg.embed_api_key, sleep=cfg.embed_batch_sleep)

    print("\n-- embedding --")
    started = time.time()
    vectors = embedder.embed_documents(
        [c.get("embed_text") or c.get("text") or "" for c in pending],
        progress=lambda f, m: print(f"\r   {m:<50}", end="", flush=True),
    )
    print(f"\r   embedded {len(vectors)} chunk(s) in {time.time() - started:.1f}s{' ' * 20}")

    docs = attach_embeddings(pending, vectors, cfg.embed_model)

    print("\n-- writing --")
    written = upsert_chunks(coll, docs, batch_size=cfg.write_batch_size,
                            throttle=cfg.write_throttle,
                            progress=lambda f, m: print(f"\r   {m:<50}", end="", flush=True))
    print(f"\r   stored {written} chunk(s){' ' * 30}")

    if cfg.store_document_record and args.markdown:
        record: Dict = {
            "doc_id": doc_id,
            "title": pathlib.Path(args.markdown).name,
            "plan_id": cfg.plan_id,
            "group_number": cfg.group_number,
            "plan_year": cfg.plan_year,
            "effective_date": cfg.effective_date,
            "carrier": cfg.carrier,
            "market_segment": cfg.market_segment,
            "states": cfg.states,
            "markdown": pathlib.Path(args.markdown).read_text(encoding="utf-8"),
            "n_chunks": len(chunks),
            "embed_model": cfg.embed_model,
            "embed_dim": cfg.embed_dim,
            "status": "indexed",
        }
        tables = []
        if args.tables:
            tables = json.loads(pathlib.Path(args.tables).read_text(encoding="utf-8"))
        record["tables"] = tables
        record["tables_complete"] = bool(tables)
        doc_coll = get_collection(cfg.mongodb_uri, cfg.mongodb_db, DEFAULT_DOC_COLLECTION)
        upsert_document_record(doc_coll, record)
        print(f"   document record stored "
              f"({len(tables)} table structures"
              f"{'' if tables else ' — no audit artifact'})")
    elif cfg.store_document_record:
        print("   document record skipped: pass --markdown to store one")

    idx = index_state(coll, cfg.vector_index) or {}
    ready = idx.get("queryable") or idx.get("status") == "READY"
    print(f"\n-- state --")
    print(f"   vector index queryable: {ready}")
    if not ready:
        print("   still building; queries return nothing until it is READY")
        print(f"   check with: python tools/index_status.py")
    for row in list_documents(coll):
        print(f"   {row['doc_id']:<24} chunks={row['chunks']:<6} "
              f"tables={row['tables']:<5} review={row['needs_review']}")


if __name__ == "__main__":
    main()
