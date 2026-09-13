"""
Phase 5 - LLM-assisted denial support.  DEVELOPER TESTING ONLY.

Type a procedure code (optionally with a diagnosis code), have a language model
describe it in plain clinical English, then search the plan for the provision
that governs it.

Why this page is separate from Phase 4
--------------------------------------
Phase 4 refuses to infer what a code means: descriptors come from your licensed
code set, because a model's recollection of a code can be wrong in ways nobody
catches, and a wrong descriptor leads to a confidently wrong citation in a
regulated document.

This page relaxes that on purpose, for exploration -- to see what the plan says
about a service without first building a code table. Two things keep it honest:

  * The model is asked for a plain-language clinical description, NOT the
    official AMA descriptor. That avoids reproducing copyrighted text, and it is
    what the search needs anyway, since plan documents never use code-set
    vocabulary.
  * The description is shown and is editable before any search runs. The search
    uses what is in the box, not what the model said -- so a human always sits
    between the model's guess and the citation.
"""

from __future__ import annotations

import pathlib
import sys

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from coc.config import load_settings  # noqa: E402
from coc.denial_support import (  # noqa: E402
    rerank_provisions,
    DEFAULT_SCOPES,
    build_core_terms,
    MarkdownProvisionIndex,
    assess,
    build_query_terms,
    classify_code,
    describe_codes,
    format_citation_block,
    normalize_code,
    search_provisions_vector,
)

st.set_page_config(page_title="LLM Denial Support", page_icon="🧪", layout="wide")

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
settings = st.session_state.settings

st.title("🧪 Phase 5 — LLM-assisted denial support")
st.caption(
    "Code in, plain-English description out, then search the plan for the "
    "provision that governs it."
)

st.error(
    "**Developer testing only — not for producing denial letters.** Unlike "
    "Phase 4, this page lets a language model guess what a code means. Models "
    "misremember codes, and a wrong description leads to a confidently wrong "
    "citation. Read and correct the description before searching, and use "
    "Phase 4 with a licensed code table for anything that reaches a member.",
    icon="🧪",
)

@st.cache_resource(show_spinner=False)
def _cached_embedder(provider: str, model: str, dim: int, api_key: str):
    """One embedder per configuration, reused across reruns.

    Streamlit re-executes the whole script on every interaction, so building a
    client per click opens a fresh TCP connection pool each time. On Windows
    those sockets sit in TIME_WAIT and the ephemeral port range is exhausted
    within a few dozen clicks, surfacing as WinError 10048.
    """
    from coc.embeddings import get_embedder

    return get_embedder(provider, model, dim, api_key=api_key)


@st.cache_resource(show_spinner=False)
def _cached_collection(uri: str, db: str, name: str):
    """MongoClient maintains its own pool; one per process is correct."""
    from coc.vectorstore import get_collection

    return get_collection(uri, db, name)


@st.cache_data(show_spinner=False, ttl=120)
def _indexed_documents(uri: str, db: str, coll_name: str) -> list:
    """Documents available in Atlas, for the plan selector."""
    from coc.vectorstore import list_documents

    try:
        coll = _cached_collection(uri, db, coll_name)
        return list_documents(coll)
    except Exception:
        return []


def _document_selector(settings) -> str:
    """Choose which indexed plan to search.

    Required rather than optional. With several plans indexed, an unscoped
    search can return another plan's exclusion, and a denial letter citing the
    wrong plan document is worse than one citing nothing.
    """
    docs = _indexed_documents(settings.mongodb_uri, settings.mongodb_db,
                              settings.mongodb_collection)
    if not docs:
        st.sidebar.warning("No indexed documents found in Atlas.", icon="📭")
        return ""

    def label(d: dict) -> str:
        parts = [d["doc_id"]]
        if d.get("plan_id"):
            parts.append(str(d["plan_id"]))
        if d.get("plan_year"):
            parts.append(str(d["plan_year"]))
        return "  ·  ".join(parts) + f"   ({d['chunks']} chunks)"

    labels = [label(d) for d in docs]
    chosen = st.sidebar.selectbox(
        "Plan document", labels,
        help="Searches are scoped to this document. Every indexed plan has its "
             "own exclusions, so an unscoped search can surface the wrong "
             "plan's language.",
    )
    picked = docs[labels.index(chosen)]
    bits = [f"**{picked['doc_id']}**"]
    if picked.get("plan_id"):
        bits.append(picked["plan_id"])
    if picked.get("plan_year"):
        bits.append(str(picked["plan_year"]))
    st.sidebar.caption("Searching " + " · ".join(bits))
    return picked["doc_id"]


# --------------------------------------------------------------------------- #
# Step 1 - describe the codes
# --------------------------------------------------------------------------- #
st.subheader("1 · Codes")

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    proc_code = st.text_input("Procedure code", placeholder="15830")
    if proc_code:
        k = classify_code(proc_code)
        st.caption(f"Format: **{k}**" if k != "unknown" else "Format not recognised")
with c2:
    diag_code = st.text_input("Diagnosis code (optional)", placeholder="E66.01")
    if diag_code:
        k = classify_code(diag_code)
        st.caption(f"Format: **{k}**" if k != "unknown" else "Format not recognised")
with c3:
    provider = st.selectbox(
        "Description model", ["anthropic", "openai"],
        index=["anthropic", "openai"].index(settings.answer_provider),
    )
    model = st.text_input("Model", value=settings.answer_model)

if st.button("Describe these codes", type="primary", disabled=not proc_code):
    key = (settings.anthropic_api_key if provider == "anthropic"
           else settings.openai_api_key)
    with st.spinner("Asking the model what these codes represent…"):
        try:
            st.session_state.code_description = describe_codes(
                proc_code, diag_code or None, provider=provider,
                model=model or None, api_key=key,
            )
        except Exception as exc:
            st.error(f"Description failed: {exc}")

desc = st.session_state.get("code_description")

# --------------------------------------------------------------------------- #
# Step 2 - review and correct
# --------------------------------------------------------------------------- #
if desc:
    st.subheader("2 · Review the description")

    badge = {"high": ("Model reports high confidence", st.info),
             "medium": ("Model reports medium confidence — verify", st.warning),
             "low": ("Model reports LOW confidence — verify before searching", st.error)}
    label, fn = badge.get(desc.confidence, badge["low"])
    fn(f"**{label}**  ·  {desc.model}", icon="🔎")

    for c in desc.caveats:
        st.warning(c, icon="❗")

    st.caption(
        "Edit anything below. The search uses these boxes, not the model's "
        "original output."
    )
    e1, e2 = st.columns(2)
    with e1:
        procedure_description = st.text_area(
            f"What {desc.procedure_code} is, in plain English",
            value=desc.procedure_description, height=110,
        )
        diagnosis_description = st.text_area(
            f"What {desc.diagnosis_code or 'the diagnosis'} is (optional)",
            value=desc.diagnosis_description or "", height=90,
        )
    with e2:
        search_terms = st.text_area(
            "Search terms (one per line) — words a plan document would use",
            value="\n".join(desc.search_terms), height=110,
            help="Plans write 'hanging skin' and 'cosmetic', not 'panniculectomy'. "
                 "These matter more than the descriptor for recall.",
        )
        st.caption(f"**Clinical context:** {desc.clinical_context or '—'}")

    with st.expander("Raw model response"):
        st.code(desc.raw or "(empty)", language="json")

    # ---------------------------------------------------------------------- #
    # Step 3 - search plan language
    # ---------------------------------------------------------------------- #
    st.subheader("3 · Search the plan")

    s1, s2 = st.columns([2, 3])
    with s1:
        source = st.radio("Search against",
                          ["Atlas vector index", "Converted Markdown (lexical)"])
        selected_doc = ""
        if source.startswith("Atlas") and settings.mongodb_uri:
            selected_doc = _document_selector(settings)
    with s2:
        scopes = st.multiselect("Sections", list(DEFAULT_SCOPES),
                                default=list(DEFAULT_SCOPES))
        use_rerank = st.checkbox(
            "Rerank candidates", value=False,
            help="Reorders the shortlist with a cross-encoder. Ordering only — "
                 "it never changes the verdict.",
        )

    markdown = None
    if source.startswith("Converted"):
        if st.session_state.get("markdown"):
            markdown = st.session_state["markdown"]
            st.caption("Using the Markdown from Phase 1.")
        else:
            up = st.file_uploader("Converted COC Markdown", type=["md", "markdown", "txt"])
            if up:
                markdown = up.getvalue().decode("utf-8", errors="replace")

    if st.button("Find governing provision", type="primary"):
        descriptors = [d for d in (procedure_description, diagnosis_description) if d.strip()]
        if not descriptors:
            st.error("A description is required before searching.")
            st.stop()
        extra = [t.strip() for t in search_terms.splitlines() if t.strip()]
        terms = build_query_terms(descriptors, extra)
        core_terms = build_core_terms(descriptors, extra)
        codes = [normalize_code(desc.procedure_code)]
        if desc.diagnosis_code:
            codes.append(normalize_code(desc.diagnosis_code))

        provisions = []
        search_idf = {}
        if source.startswith("Converted"):
            if not markdown:
                st.error("No Markdown loaded.")
                st.stop()
            with st.spinner("Searching plan language…"):
                index = MarkdownProvisionIndex(markdown, scopes=scopes or DEFAULT_SCOPES)
                provisions = index.search(terms, limit=6, descriptors=descriptors + extra,
                                      core_terms=core_terms)
                search_idf = index.idf
            st.caption(f"Searched {len(index.units):,} citable sentences.")
        else:
            try:
                from coc.embeddings import get_embedder
                from coc.vectorstore import get_collection

                ekey = (settings.voyage_api_key if settings.embed_provider == "voyage"
                        else settings.openai_api_key)
                embedder = _cached_embedder(settings.embed_provider, settings.embed_model,
                                            int(settings.embed_dim), ekey or "")
                coll = _cached_collection(settings.mongodb_uri, settings.mongodb_db,
                                          settings.mongodb_collection)
                from coc.denial_support import scoped_section_values

                with st.spinner("Searching the vector index…"):
                    sections = scoped_section_values(coll, scopes or DEFAULT_SCOPES)
                    provisions = search_provisions_vector(
                        coll, embedder, " ".join(descriptors + extra), terms,
                        doc_id=selected_doc or None,
                        scopes=scopes or DEFAULT_SCOPES, limit=6,
                        descriptors=descriptors + extra,
                        core_terms=core_terms, idf_out=search_idf,
                    )
                st.caption(
                    (f"Scoped to **{selected_doc}**; " if selected_doc else
                     "**Not scoped to a document**. ")
                    + f"pre-filtered to {len(sections)} in-scope section(s); "
                    f"{len(provisions)} candidate sentence(s) scored."
                )
            except Exception as exc:
                st.error(f"Vector search unavailable: {exc}")
                st.stop()

        if use_rerank and provisions:
            with st.spinner("Reranking candidates…"):
                provisions = rerank_provisions(
                    provisions, " ".join(descriptors + extra),
                    api_key=settings.voyage_api_key, top_n=6)

        result = assess(codes, descriptors, terms, provisions,
                        idf=search_idf)

        banner = {
            "supported": ("Provision found — verify before citing", st.success),
            "weak": ("Weak match — do not cite without confirmation", st.warning),
            "not_found": ("No supporting plan provision found", st.error),
        }[result.outcome]
        banner[1](f"**{banner[0]}**", icon="⚖️")

        if desc.is_uncertain and result.outcome == "supported":
            st.warning(
                "The provision matched well, but the model was not confident about "
                "what the code represents. A confident match on a wrong description "
                "is still a wrong citation — confirm the service before relying on it.",
                icon="⚠️",
            )

        for note in result.notes:
            st.warning(note, icon="❗")

        if provisions:
            st.subheader("Candidate provisions")
            for i, p in enumerate(provisions, start=1):
                flags = ("" if not p.has_conflict else " · ⛔ CONFLICT") + \
                        ("" if not p.has_exception else " · ⚠️ exception")
                with st.expander(f"{i}. {p.citation} — score {p.score:.2f}"
                                 f"{f' · rerank {p.rerank_score:.3f}' if p.rerank_score is not None else ''}"
                                 f"{flags}",
                                 expanded=(i == 1)):
                    if p.lead_in:
                        st.caption("Context that makes this provision operative:")
                        for depth, part in enumerate(p.lead_in):
                            st.markdown(f"{'&nbsp;' * (depth * 4)}> {part}",
                                        unsafe_allow_html=True)
                        st.markdown(
                            f"{'&nbsp;' * (len(p.lead_in) * 4)}> **{p.text}**",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(f"> {p.text}")
                    m1, m2 = st.columns(2)
                    m1.caption(f"**Breadcrumb:** {p.breadcrumb}")
                    m1.caption(f"**Pages:** {p.page_start}–{p.page_end} · "
                               f"**Type:** {p.source_type}")
                    m2.caption(f"**Matched:** {', '.join(p.matched_terms) or '—'}")
                    m2.caption(f"**Distinctive:** {', '.join(p.distinctive_terms) or 'none'}")
                    for c in p.conflicts:
                        st.error(f"Qualifier conflict: {c}", icon="⛔")
                    if p.exception_clause:
                        st.warning(f"Exception clause: *{p.exception_clause}*", icon="⚠️")
                    if p.cross_references:
                        st.info("Cross-references: " + ", ".join(p.cross_references), icon="🔗")

            st.subheader("Draft citation block")
            block = format_citation_block(result)
            st.code(block, language="text")
            st.download_button(
                "Download citation block",
                (f"Plan document: {selected_doc}\n" if selected_doc else "")
                + f"Code(s): {', '.join(result.codes)}\n"
                f"Description (LLM-assisted, human-reviewed): "
                f"{'; '.join(result.descriptors)}\n"
                f"Description model: {desc.model} (confidence: {desc.confidence})\n"
                f"Outcome: {result.outcome}\n\n{block}\n\n"
                + "\n".join(f"NOTE: {n}" for n in result.notes)
                + "\n\nSOURCE OF DESCRIPTION: language model, not a licensed code "
                  "set. Verify before any external use.\n"
                  "\nReviewed by: ____________________  Date: __________\n",
                file_name=f"llm_provision_{'_'.join(result.codes)}.txt",
            )
        else:
            st.info(
                "Nothing matched. Try adding search terms in the plan's own "
                "vocabulary — plan language rarely mirrors code-set wording.",
                icon="🔍",
            )
