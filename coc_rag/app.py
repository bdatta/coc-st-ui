"""
COC Knowledge Base -- Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pathlib
import sys

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from coc.config import load_settings  # noqa: E402

st.set_page_config(page_title="COC Knowledge Base", page_icon="📘", layout="wide")

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()

settings = st.session_state.settings

st.title("Certificate of Coverage — Knowledge Base")
st.caption(
    "Convert a COC PDF to structured Markdown, chunk and embed it, then ask "
    "benefit questions against the indexed document."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1 · Convert")
    st.write("PDF → Markdown with section hierarchy and page-spanning tables preserved.")
    st.page_link("pages/1_PDF_to_Markdown.py", label="Open converter", icon="📄")
with col2:
    st.subheader("2 · Index")
    st.write("Markdown → section-aware chunks → embeddings → MongoDB Atlas.")
    st.page_link("pages/2_Chunk_and_Embed.py", label="Open indexer", icon="🧩")
with col3:
    st.subheader("3 · Ask")
    st.write("Semantic search over the COC with cited, grounded answers.")
    st.page_link("pages/3_Ask_Questions.py", label="Open Q&A", icon="💬")

st.divider()
st.subheader("Configuration")
st.caption(
    "Values come from environment variables or a `.env` file. Overrides here "
    "apply to this session only and are not written to disk."
)

with st.form("settings_form"):
    a, b = st.columns(2)
    with a:
        st.markdown("**MongoDB Atlas**")
        uri = st.text_input("Connection URI", value=settings.mongodb_uri, type="password",
                            placeholder="mongodb+srv://user:pass@cluster.mongodb.net/")
        db = st.text_input("Database", value=settings.mongodb_db)
        coll = st.text_input("Collection", value=settings.mongodb_collection)
        vindex = st.text_input("Vector index name", value=settings.vector_index)
        tindex = st.text_input("Text index name (hybrid, optional)", value=settings.text_index)
    with b:
        st.markdown("**Models**")
        provider = st.selectbox(
            "Embedding provider", ["voyage", "openai", "local"],
            index=["voyage", "openai", "local"].index(settings.embed_provider),
        )
        model = st.text_input("Embedding model", value=settings.embed_model)
        dim = st.number_input("Embedding dimensions", min_value=64, max_value=4096,
                              value=int(settings.embed_dim), step=64)
        voyage_key = st.text_input("Voyage API key", value=settings.voyage_api_key, type="password")
        openai_key = st.text_input("OpenAI API key", value=settings.openai_api_key, type="password")
        answer_provider = st.selectbox(
            "Answer provider", ["anthropic", "openai"],
            index=["anthropic", "openai"].index(settings.answer_provider),
        )
        answer_model = st.text_input("Answer model", value=settings.answer_model)
        anthropic_key = st.text_input("Anthropic API key", value=settings.anthropic_api_key,
                                      type="password")

    if st.form_submit_button("Save for this session", type="primary"):
        settings.mongodb_uri = uri
        settings.mongodb_db = db
        settings.mongodb_collection = coll
        settings.vector_index = vindex
        settings.text_index = tindex
        settings.embed_provider = provider
        settings.embed_model = model
        settings.embed_dim = int(dim)
        settings.voyage_api_key = voyage_key
        settings.openai_api_key = openai_key
        settings.answer_provider = answer_provider
        settings.answer_model = answer_model
        settings.anthropic_api_key = anthropic_key
        st.success("Settings updated.")

st.divider()
if st.button("Test MongoDB connection"):
    if not settings.mongodb_uri:
        st.error("No connection URI set.")
    else:
        try:
            from coc.vectorstore import get_collection, list_documents, list_search_indexes

            collection = get_collection(settings.mongodb_uri, settings.mongodb_db,
                                        settings.mongodb_collection)
            st.success(f"Connected. Collection holds {collection.estimated_document_count()} documents.")
            idx = list_search_indexes(collection)
            if idx:
                st.write("Search indexes:")
                st.dataframe(
                    [{"name": i.get("name"), "type": i.get("type"),
                      "status": i.get("status"), "queryable": i.get("queryable")} for i in idx],
                    hide_index=True, width="stretch",
                )
            else:
                st.info("No search indexes yet — create one on the Index page.")
            docs = list_documents(collection)
            if docs:
                st.dataframe(docs, hide_index=True, width="stretch")
        except Exception as exc:
            st.error(f"Connection failed: {exc}")

with st.expander("Session state"):
    st.write({
        "markdown_loaded": bool(st.session_state.get("markdown")),
        "markdown_source": st.session_state.get("markdown_name"),
        "chunks_prepared": len(st.session_state.get("chunks") or []),
    })
