"""Phase 3 — ask benefit questions against the vectorized COC."""

from __future__ import annotations

import pathlib
import sys

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from coc.config import load_settings  # noqa: E402
from coc.embeddings import get_embedder  # noqa: E402
from coc.qa import generate_answer, retrieve, rerank, to_sources  # noqa: E402
from coc.vectorstore import get_collection, list_documents  # noqa: E402

st.set_page_config(page_title="Ask the COC", page_icon="💬", layout="wide")

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
settings = st.session_state.settings

st.title("💬 Phase 3 — Ask a benefit question")

if not settings.mongodb_uri:
    st.warning("Set a MongoDB URI on the home page first.", icon="⚠️")
    st.stop()


@st.cache_resource(show_spinner=False)
def _collection(uri: str, db: str, coll: str):
    return get_collection(uri, db, coll)


@st.cache_data(show_spinner=False, max_entries=256)
def _cached_query_vector(question: str, provider: str, model: str, dim: int,
                         api_key: str) -> list:
    """Cache the query embedding.

    Re-asking a question during testing then costs no network call at all, which
    both speeds up iteration and keeps socket churn down — the practical trigger
    for Windows ephemeral-port exhaustion.
    """
    embedder = _cached_embedder(provider, model, dim, api_key)
    return list(embedder.embed_query(question))


class _PrecomputedQuery:
    """Adapter presenting a cached vector through the embedder interface."""

    def __init__(self, vector: list):
        self._vector = vector

    def embed_query(self, text: str) -> list:
        return self._vector


@st.cache_resource(show_spinner=False)
def _cached_embedder(provider: str, model: str, dim: int, api_key: str):
    """One embedder per configuration, reused across reruns.

    Streamlit re-executes the script on every interaction, so building a client
    per click opens a fresh TCP connection pool each time. On Windows those
    sockets sit in TIME_WAIT and the ephemeral port range runs out after a few
    dozen clicks, surfacing as WinError 10048.
    """
    return get_embedder(provider, model, dim, api_key=api_key)


try:
    collection = _collection(settings.mongodb_uri, settings.mongodb_db,
                             settings.mongodb_collection)
    documents = list_documents(collection)
except Exception as exc:
    st.error(f"Could not reach MongoDB: {exc}")
    st.stop()

if not documents:
    st.info("No indexed documents found. Run Phase 2 first.", icon="📭")
    st.stop()

with st.sidebar:
    st.header("Retrieval")
    doc_options = ["(all documents)"] + [d["doc_id"] for d in documents]
    selected = st.selectbox("Document", doc_options)
    doc_id = None if selected.startswith("(") else selected
    k = st.slider("Chunks to retrieve", 3, 20, 8)
    use_hybrid = st.checkbox("Hybrid (vector + keyword)", value=False,
                             help="Requires the lexical index from Phase 2. "
                                  "Helps with exact terms like CPT codes.")
    expand_neighbors = st.checkbox("Include neighbouring chunks", value=False,
                                   help="Pulls the chunks immediately before and "
                                        "after each hit, in case a sentence ran "
                                        "over a boundary.")
    do_rerank = st.checkbox("Rerank results", value=False,
                            help="Cross-encoder rerank via Voyage. Costs one extra "
                                 "API call, usually improves precision noticeably.")
    st.divider()
    st.header("Answering")
    answer_provider = st.selectbox("Provider", ["anthropic", "openai"],
                                   index=["anthropic", "openai"].index(settings.answer_provider))
    answer_model = st.text_input("Model", value=settings.answer_model)
    retrieval_only = st.checkbox("Retrieval only (skip generation)", value=False)

st.caption(
    "Answers are drawn only from the indexed COC and cite the section and page "
    "they came from. Verify anything you act on against the plan document itself."
)

examples = [
    "What is the out-of-pocket maximum for an individual in-network?",
    "Is bariatric surgery covered, and what conditions apply?",
    "How many physical therapy visits are allowed per year?",
    "What is the copay for a specialist visit out-of-network?",
]
# The question box is keyed so its contents survive the rerun that clicking
# "Ask" triggers. An earlier version popped a staged value into `value=`, which
# meant picking an example question and then pressing Ask silently did nothing:
# the pop had already consumed the text, so the box came back empty.
if "question_box" not in st.session_state:
    st.session_state.question_box = ""

cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, width="stretch"):
        st.session_state.question_box = ex
        st.rerun()

question = st.text_area(
    "Question", key="question_box",
    placeholder="e.g. What is the deductible for family coverage in-network?",
    height=90,
)

asked = st.button("Ask", type="primary")
if asked and not question.strip():
    st.warning("Type a question first, or pick one of the examples above.", icon="✍️")
if asked and question.strip():
    api_key = (settings.voyage_api_key if settings.embed_provider == "voyage"
               else settings.openai_api_key)
    try:
        qvec = _cached_query_vector(question, settings.embed_provider,
                                    settings.embed_model, int(settings.embed_dim),
                                    api_key or "")
        embedder = _PrecomputedQuery(qvec)
    except Exception as exc:
        st.error(f"Embedder unavailable: {exc}")
        if "10048" in str(exc):
            st.info(
                "Windows ran out of ephemeral ports — sockets from earlier calls "
                "are still in TIME_WAIT. Wait about two minutes and retry; the "
                "same question will then come from cache without a network call.",
                icon="🔌",
            )
        st.stop()

    with st.spinner("Searching the COC…"):
        try:
            hits = retrieve(
                collection, embedder, question, k=k, doc_id=doc_id,
                use_hybrid=use_hybrid, expand_neighbors=expand_neighbors,
                vector_index=settings.vector_index, text_index=settings.text_index,
            )
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            st.info("If the index was just created, give it a minute to finish building.")
            st.stop()

        if do_rerank:
            hits = rerank(settings.voyage_api_key, question, hits, top_n=min(k, 8))

    if not hits:
        st.warning("Nothing matched. Try the plan's own terminology, or widen the "
                   "retrieval count.", icon="🔍")
        st.stop()

    sources = to_sources(hits)

    if not retrieval_only:
        with st.spinner("Composing an answer…"):
            key = (settings.anthropic_api_key if answer_provider == "anthropic"
                   else settings.openai_api_key)
            try:
                answer = generate_answer(question, sources, provider=answer_provider,
                                         model=answer_model, api_key=key)
            except Exception as exc:
                answer = None
                st.error(f"Answer generation failed: {exc}")
        if answer:
            st.subheader("Answer")
            st.markdown(answer)

    st.subheader(f"Sources ({len(sources)})")
    for s in sources:
        icon = "▦" if s.type == "table" else "¶"
        with st.expander(f"[{s.index}] {icon} {s.label} — score {s.score:.4f}"):
            if s.type == "table":
                st.markdown(s.text, unsafe_allow_html=True)
                with st.popover("Raw chunk"):
                    st.code(s.text, language="markdown")
            else:
                st.markdown(s.text)
            st.caption(f"chunk_id: {s.chunk_id}"
                       + (f" · table: {s.table_id}" if s.table_id else ""))
