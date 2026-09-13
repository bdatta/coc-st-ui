"""Phase 2 — chunk the Markdown, embed it, and store it in MongoDB Atlas."""

from __future__ import annotations

import json
import pathlib
import re
import sys
import time

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from coc.chunker import (  # noqa: E402
    ChunkOptions,
    DocumentContext,
    chunk_markdown,
    sha256_bytes,
)
from coc.config import load_settings  # noqa: E402
from coc.embeddings import SUGGESTED_MODELS, get_embedder  # noqa: E402
from coc.vectorstore import (  # noqa: E402
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

st.set_page_config(page_title="Chunk & Embed", page_icon="🧩", layout="wide")

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
settings = st.session_state.settings

st.title("🧩 Phase 2 — Chunk, Embed, Store")
st.caption(
    "Tables are never split mid-table. Oversized tables are divided by rows with "
    "the header repeated on each part, and every chunk carries its heading "
    "breadcrumb so a stray copay row still knows which benefit it belongs to."
)

# --------------------------------------------------------------------------- #
# Source
# --------------------------------------------------------------------------- #
st.subheader("Source Markdown")
source = st.radio("Input", ["Use Markdown from Phase 1", "Upload a .md file"],
                  horizontal=True, label_visibility="collapsed")

markdown = None
default_doc_id = "coc"
if source.startswith("Use") and st.session_state.get("markdown"):
    markdown = st.session_state["markdown"]
    default_doc_id = pathlib.Path(st.session_state.get("markdown_name", "coc.md")).stem
    st.success(f"Using in-session Markdown ({len(markdown):,} characters).")
elif source.startswith("Use"):
    st.info("Nothing in session yet — convert a PDF on Phase 1 or upload a file here.")
else:
    mcol, tcol = st.columns(2)
    with mcol:
        up = st.file_uploader("Markdown file", type=["md", "markdown", "txt"])
        if up:
            markdown = up.getvalue().decode("utf-8", errors="replace")
            default_doc_id = pathlib.Path(up.name).stem
            st.session_state.source_filename = up.name
            st.session_state.source_sha256 = sha256_bytes(up.getvalue())
            st.success(f"Loaded {up.name} ({len(markdown):,} characters).")
    with tcol:
        tj = st.file_uploader(
            "Table structures (JSON) — optional but recommended", type=["json"],
            help="The 'Download table structures' file from Phase 1. Holds every "
                 "cell with its true rowspan/colspan. Without it the document "
                 "record is stored with no audit artifact.",
        )
        if tj:
            try:
                parsed = json.loads(tj.getvalue().decode("utf-8"))
                if not isinstance(parsed, list):
                    raise ValueError("expected a JSON array of table objects")
                bad = [t for t in parsed if not isinstance(t, dict) or "table_id" not in t]
                if bad:
                    raise ValueError(f"{len(bad)} entries are missing 'table_id'")
                st.session_state.tables_json = parsed
                st.success(f"Loaded {len(parsed)} table structures.")
            except Exception as exc:
                st.error(f"Could not read the table structures: {exc}")

# Reconcile the table IDs in the Markdown against the loaded structures. A
# mismatch means the two files came from different conversion runs, which would
# make the audit trail point at the wrong cells.
tables_json = st.session_state.get("tables_json") or []
if markdown:
    md_ids = set(re.findall(r"<!--\s*table\s+id=(\S+)", markdown))
    json_ids = {t.get("table_id") for t in tables_json}
    if md_ids and not json_ids:
        st.warning(
            f"The Markdown contains {len(md_ids)} tables but no table structures "
            "are loaded. You can still index, but the document record will have "
            "no cell-level audit artifact. Re-download it from Phase 1, or "
            "re-run the conversion.", icon="⚠️",
        )
    elif md_ids and json_ids:
        missing, extra = md_ids - json_ids, json_ids - md_ids
        if missing or extra:
            st.error(
                f"Table structures do not match this Markdown "
                f"({len(missing)} in Markdown only, {len(extra)} in JSON only). "
                "These files are from different conversion runs — using them "
                "together would point the audit trail at the wrong cells.",
                icon="🛑",
            )
            tables_json = []
        else:
            st.caption(f"✓ Table structures reconciled: {len(md_ids)} tables matched.")

# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Chunking")
    target_tokens = st.slider("Target tokens per text chunk", 200, 1500, 650, 50)
    max_tokens = st.slider("Hard max tokens", 300, 2000, 900, 50)
    overlap_tokens = st.slider("Overlap tokens", 0, 300, 90, 10)
    max_table_tokens = st.slider("Max tokens per table chunk", 400, 3000, 1100, 100)
    include_row_sentences = st.checkbox(
        "Append row sentences to table chunks", value=True,
        help="Adds 'Column: value' pairs per row. Materially improves recall on "
             "benefit grids because values stay bound to their column headers.",
    )
    prefix_breadcrumb = st.checkbox("Prefix section breadcrumb into embedded text", value=True)

    st.header("Embeddings")
    provider = st.selectbox("Provider", ["voyage", "openai", "local"],
                            index=["voyage", "openai", "local"].index(settings.embed_provider))
    model = st.selectbox("Model", SUGGESTED_MODELS[provider] + ["custom…"], index=0)
    if model == "custom…":
        model = st.text_input("Custom model name", value=settings.embed_model)
    dim = st.number_input("Dimensions", 64, 4096, int(settings.embed_dim), 64)
    throttle = st.slider("Write throttle (seconds per 100 docs)", 0.0, 2.0, 0.4, 0.1,
                         help="Atlas free clusters cap at 100 operations/second.")

    st.header("Atlas")
    create_text_index = st.checkbox(
        "Also create a lexical index (hybrid search)", value=False,
        help="Costs an extra search index. Worth it for exact-term lookups such "
             "as CPT codes or defined plan terms.",
    )

chunk_opts = ChunkOptions(
    target_tokens=target_tokens,
    max_tokens=max_tokens,
    overlap_tokens=overlap_tokens,
    max_table_tokens=max_table_tokens,
    include_row_sentences=include_row_sentences,
    prefix_breadcrumb=prefix_breadcrumb,
)

doc_id = st.text_input("Document ID", value=default_doc_id,
                       help="Namespaces the chunks so you can index several COCs "
                            "in one collection and filter at query time.")

st.subheader("Plan identity")
st.caption(
    "Stamped onto every chunk and declared as filters on the vector index. "
    "Answering from the wrong plan or plan year is the costliest failure mode "
    "this system has, so these are worth filling in before you load real data."
)
pi1, pi2, pi3, pi4 = st.columns(4)
plan_id = pi1.text_input("Plan ID", placeholder="ACME-PPO-2000")
group_number = pi2.text_input("Group number", placeholder="0084512")
plan_year = pi3.number_input("Plan year", 2000, 2100, 2026, 1)
carrier = pi4.text_input("Carrier", placeholder="Acme Health")

pi5, pi6, pi7, pi8 = st.columns(4)
effective_date = pi5.date_input("Effective date", value=None)
termination_date = pi6.date_input("Termination date", value=None)
market_segment = pi7.selectbox(
    "Market segment", ["", "large_group", "small_group", "individual", "medicare", "medicaid"]
)
states_raw = pi8.text_input("States (comma-separated)", placeholder="IL, IN")

doc_ctx = DocumentContext(
    doc_id=doc_id,
    plan_id=plan_id or None,
    group_number=group_number or None,
    plan_year=int(plan_year) if plan_year else None,
    effective_date=effective_date.isoformat() if effective_date else None,
    termination_date=termination_date.isoformat() if termination_date else None,
    carrier=carrier or None,
    market_segment=market_segment or None,
    states=[s.strip() for s in states_raw.split(",") if s.strip()],
    source_filename=st.session_state.get("source_filename"),
    source_sha256=st.session_state.get("source_sha256"),
)

# --------------------------------------------------------------------------- #
# Preview
# --------------------------------------------------------------------------- #
if markdown and st.button("Preview chunks"):
    with st.spinner("Chunking…"):
        chunks = chunk_markdown(markdown, doc_ctx, chunk_opts)
    st.session_state.chunks = chunks

chunks = st.session_state.get("chunks")
if chunks:
    st.subheader("Chunk preview")
    table_chunks = [c for c in chunks if c.type == "table"]
    tokens = [c.n_tokens for c in chunks]
    m = st.columns(5)
    m[0].metric("Chunks", len(chunks))
    m[1].metric("Table chunks", len(table_chunks))
    m[2].metric("Median tokens", int(sorted(tokens)[len(tokens) // 2]) if tokens else 0)
    m[3].metric("Max tokens", max(tokens) if tokens else 0)
    m[4].metric("Flagged for review", sum(1 for c in chunks if c.needs_review))

    st.dataframe(
        [{
            "seq": c.seq,
            "type": c.type,
            "pages": f"{c.page_start}-{c.page_end}",
            "tokens": c.n_tokens,
            "part": f"{c.part_index}/{c.part_total}",
            "review": "⚠" if c.needs_review else "",
            "section": c.breadcrumb[:90],
            "preview": " ".join(c.text.split())[:110],
        } for c in chunks],
        hide_index=True, width="stretch", height=340,
    )

    pick = st.number_input("Inspect chunk #", 0, len(chunks) - 1, 0)
    with st.expander("Embedded text for this chunk", expanded=True):
        st.code(chunks[pick].embed_text, language="markdown")

    st.download_button(
        "Download chunks (JSONL)",
        "\n".join(json.dumps(c.to_doc(), ensure_ascii=False) for c in chunks),
        file_name=f"{doc_id}_chunks.jsonl", mime="application/json",
    )

# --------------------------------------------------------------------------- #
# Embed + store
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Embed and store in MongoDB Atlas")

if not settings.mongodb_uri:
    st.warning("Set a MongoDB URI on the home page first.", icon="⚠️")

replace_existing = st.checkbox(f"Delete existing chunks for '{doc_id}' first", value=True)
store_document_record = st.checkbox(
    f"Also store the document record in '{DEFAULT_DOC_COLLECTION}'", value=True,
    help="Keeps the Markdown and table structures alongside the chunks, so you "
         "can re-chunk later without re-parsing the PDF.",
)
if store_document_record and markdown and not tables_json:
    proceed_without_tables = st.checkbox(
        "Store the document record without table structures (no audit artifact)",
        value=False,
        help="Leave unchecked to be stopped before writing an incomplete "
             "document record.",
    )
else:
    proceed_without_tables = True

if chunks and settings.mongodb_uri and st.button("Embed & store", type="primary"):
    api_key = settings.voyage_api_key if provider == "voyage" else settings.openai_api_key
    try:
        embedder = get_embedder(provider, model, int(dim), api_key=api_key)
    except Exception as exc:
        st.error(f"Could not initialise the embedder: {exc}")
        st.stop()

    try:
        coll = get_collection(settings.mongodb_uri, settings.mongodb_db,
                              settings.mongodb_collection)
    except Exception as exc:
        st.error(f"MongoDB connection failed: {exc}")
        st.stop()

    with st.status("Indexing…", expanded=True) as status:
        status.write("Ensuring vector search index…")
        try:
            state = ensure_vector_index(coll, int(dim), name=settings.vector_index)
            status.write(f"Vector index: {state}")
            if create_text_index:
                status.write(f"Lexical index: {ensure_text_index(coll, settings.text_index)}")
        except Exception as exc:
            status.write(f"Index creation FAILED: {exc}")
            st.error(
                "The vector index was not created. Chunks and embeddings will "
                "still be stored, but Phase 3 will return no results until an "
                "index exists. Re-run this page once the chunks are stored, or "
                "create the index from the Atlas UI (see SETUP_ATLAS.md).",
                icon="🛑",
            )

        if replace_existing:
            removed = delete_document(coll, doc_id)
            status.write(f"Removed {removed} existing chunks for '{doc_id}'.")

        status.write(f"Embedding {len(chunks)} chunks with {model}…")
        bar = st.progress(0.0)
        started = time.time()
        try:
            vectors = embedder.embed_documents(
                [c.embed_text for c in chunks],
                progress=lambda f, m: bar.progress(min(f, 1.0), text=m),
            )
        except Exception as exc:
            status.update(label="Embedding failed", state="error")
            st.exception(exc)
            st.stop()

        docs = attach_embeddings([c.to_doc() for c in chunks], vectors, model)

        status.write("Writing to Atlas…")
        written = upsert_chunks(
            coll, docs, throttle=throttle,
            progress=lambda f, m: bar.progress(min(f, 1.0), text=m),
        )
        if store_document_record and not tables_json and not proceed_without_tables:
            status.write(
                "Skipped the document record: no table structures loaded. "
                "Chunks and embeddings were stored normally."
            )
        elif store_document_record:
            status.write("Storing the document record (Markdown + table structures)…")
            doc_coll = get_collection(settings.mongodb_uri, settings.mongodb_db,
                                      DEFAULT_DOC_COLLECTION)
            record = doc_ctx.stamp()
            record.update({
                "doc_id": doc_id,
                "title": st.session_state.get("markdown_name", doc_id),
                "markdown": markdown,
                "tables": tables_json,
                "tables_complete": bool(tables_json),
                "conversion_stats": st.session_state.get("conversion_stats", {}),
                "chunk_options": chunk_opts.__dict__,
                "chunk_config_hash": chunk_opts.config_hash(),
                "n_chunks": len(chunks),
                "embed_model": model,
                "embed_dim": int(dim),
            })
            upsert_document_record(doc_coll, record)

        status.update(
            label=f"Stored {written} chunks in {time.time() - started:.1f}s",
            state="complete",
        )

    idx = index_state(coll, settings.vector_index) or {}
    if not (idx.get("queryable") or idx.get("status") == "READY"):
        st.info(
            "The vector index is still building. Atlas usually needs a minute or "
            "two before queries return results.", icon="⏳",
        )
    st.dataframe(list_documents(coll), hide_index=True, width="stretch")
    st.page_link("pages/3_Ask_Questions.py", label="Go to Phase 3 — Ask Questions", icon="💬")
