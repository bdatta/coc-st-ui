"""
Rename a document's `doc_id` across the chunk and document collections.

Embeddings are preserved -- this is a rename, not a re-index, so nothing is
re-embedded and nothing is paid for again.

Why it is not a simple `update_many`. A chunk's `_id` is derived from the
document it belongs to (`acme_2026#00042-a3f9c1d2`) and `_id` is immutable in
MongoDB. Updating only the `doc_id` field would leave every `_id` carrying the
old name, and the next pipeline run -- which computes IDs from the *new* name --
would insert a second copy of every chunk instead of upserting over it. So the
rename reinserts under freshly computed IDs and removes the originals.

    python tools/rename_doc.py --from OLD_ID --to NEW_ID            # dry run
    python tools/rename_doc.py --from OLD_ID --to NEW_ID --apply
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from coc.chunker import _mk_id  # noqa: E402
from coc.config import load_settings  # noqa: E402
from coc.vectorstore import DEFAULT_DOC_COLLECTION, get_collection  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Rename a document's doc_id.")
    ap.add_argument("--from", dest="old", required=True)
    ap.add_argument("--to", dest="new", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="perform the rename (otherwise report only)")
    args = ap.parse_args()

    if args.old == args.new:
        print("Old and new doc_id are the same.")
        sys.exit(2)

    settings = load_settings()
    chunks = get_collection(settings.mongodb_uri, settings.mongodb_db,
                            settings.mongodb_collection)
    documents = get_collection(settings.mongodb_uri, settings.mongodb_db,
                               DEFAULT_DOC_COLLECTION)

    n_old = chunks.count_documents({"doc_id": args.old})
    n_new = chunks.count_documents({"doc_id": args.new})
    has_old_doc = documents.count_documents({"_id": args.old})
    has_new_doc = documents.count_documents({"_id": args.new})

    print(f"\n{settings.mongodb_db}.{settings.mongodb_collection}")
    print(f"   chunks with doc_id '{args.old}' : {n_old}")
    print(f"   chunks with doc_id '{args.new}' : {n_new}")
    print(f"{settings.mongodb_db}.{DEFAULT_DOC_COLLECTION}")
    print(f"   record '{args.old}' : {has_old_doc}")
    print(f"   record '{args.new}' : {has_new_doc}")

    if not n_old and not has_old_doc:
        print(f"\nNothing found under '{args.old}'. Check the spelling with:")
        print("   python tools/index_status.py")
        sys.exit(1)
    if n_new or has_new_doc:
        print(f"\n!! '{args.new}' already exists. Renaming onto it would mix two "
              "documents.\n   Pick a different name, or delete the existing one "
              "first.")
        sys.exit(2)

    if not args.apply:
        print(f"\nDry run. Would move {n_old} chunk(s) and "
              f"{has_old_doc} document record to '{args.new}'.")
        print("Re-run with --apply to perform it. Embeddings are preserved.")
        return

    print(f"\nRenaming '{args.old}' -> '{args.new}'…")

    moved = 0
    batch = []
    for row in chunks.find({"doc_id": args.old}):
        seq = int(row.get("seq") or 0)
        row["_id"] = _mk_id(args.new, seq)      # ID must follow the new name
        row["doc_id"] = args.new
        batch.append(row)
        if len(batch) >= 100:
            chunks.insert_many(batch, ordered=False)
            moved += len(batch)
            batch = []
            print(f"\r   inserted {moved}/{n_old}", end="", flush=True)
    if batch:
        chunks.insert_many(batch, ordered=False)
        moved += len(batch)
    print(f"\r   inserted {moved}/{n_old} chunk(s) under the new id")

    removed = chunks.delete_many({"doc_id": args.old}).deleted_count
    print(f"   removed {removed} chunk(s) under the old id")

    if has_old_doc:
        record = documents.find_one({"_id": args.old})
        record["_id"] = args.new
        record["doc_id"] = args.new
        documents.insert_one(record)
        documents.delete_one({"_id": args.old})
        print("   moved the document record")

    final_new = chunks.count_documents({"doc_id": args.new})
    final_old = chunks.count_documents({"doc_id": args.old})
    print(f"\nDone. '{args.new}' now has {final_new} chunk(s); "
          f"'{args.old}' has {final_old}.")
    if final_new != n_old or final_old:
        print("!! Counts do not match — inspect before relying on this document.")
        sys.exit(1)
    print("\nEmbeddings were preserved; no re-embedding needed.")
    print("Verify with:")
    print(f"   python tools/diagnose_denial.py --batch cases.csv --atlas "
          f"--doc-id {args.new}")


if __name__ == "__main__":
    main()
