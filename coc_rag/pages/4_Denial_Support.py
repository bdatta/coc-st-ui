"""Phase 4 - Find the plan provision governing a denied service."""

from __future__ import annotations

import io
import pathlib
import sys

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from coc.config import load_settings  # noqa: E402
from coc.denial_support import (  # noqa: E402
    rerank_provisions,
    CodeResolver,
    scoped_section_values,
    build_core_terms,
    DEFAULT_SCOPES,
    MarkdownProvisionIndex,
    assess,
    build_query_terms,
    classify_code,
    format_citation_block,
    normalize_code,
    search_provisions_vector,
)

st.set_page_config(page_title="Denial Support", page_icon="⚖️", layout="wide")

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
settings = st.session_state.settings

st.title("⚖️ Phase 4 — Denial support: find the governing provision")
st.caption(
    "Locates plan language that governs a denied service and quotes it verbatim "
    "with its section breadcrumb and page, for review before use in an adverse "
    "benefit determination."
)

st.warning(
    "**Decision support only.** This tool retrieves and quotes plan language; it "
    "does not determine coverage. Every result requires review by a qualified "
    "reviewer before it goes into a letter. An adverse benefit determination must "
    "state the actual basis for the denial — if the real basis is medical "
    "necessity, documentation, network status, or an exhausted benefit maximum, "
    "citing an exclusion instead is incorrect regardless of how well it matches.",
    icon="⚠️",
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
# Source
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Provision source")
    selected_doc = ""
    source = st.radio(
        "Search against", ["Converted Markdown (lexical)", "Atlas vector index"],
        help="Lexical search works without a database and every hit traces to "
             "specific matched terms. The vector index adds recall for wording "
             "that differs from the code descriptor.",
    )
    if source.startswith("Atlas") and settings.mongodb_uri:
        selected_doc = _document_selector(settings)
    use_rerank = st.checkbox(
        "Rerank candidates", value=False,
        help="Reorders the shortlist with a cross-encoder (Voyage). Helps when "
             "the description's wording differs from the plan's. Ordering only "
             "— it never changes the supported/weak/not-found verdict, because "
             "a reranker always returns a ranking even when nothing applies.",
    )
    scopes = st.multiselect(
        "Sections to search", list(DEFAULT_SCOPES), default=list(DEFAULT_SCOPES),
        help="Denials are not only supported by the exclusions section: benefit "
             "limits sit in the Schedule of Benefits, and terms such as "
             "'Experimental or Investigational' are governed by their definition.",
    )
    st.divider()
    st.header("Code descriptors")
    st.caption(
        "Descriptors are never inferred. Upload an export from your licensed "
        "code set, or type the descriptor in directly."
    )
    code_csv = st.file_uploader("Code table (CSV: code, system, descriptor, synonyms)",
                                type=["csv"])
    st.caption(
        "CPT is AMA-copyrighted and needs a license. ICD-10-CM is published free "
        "by CMS/NCHS. No code descriptions ship with this application."
    )




resolver = None
if code_csv is not None:
    try:
        resolver = CodeResolver.from_bytes(code_csv.getvalue())
        if not resolver.entries:
            raise ValueError("no rows with a 'code' column were found")
        st.sidebar.success(f"Loaded {len(resolver.entries)} codes.")
    except Exception as exc:
        st.sidebar.error(f"Could not read the code table: {exc}")

markdown = None
if source.startswith("Converted"):
    if st.session_state.get("markdown"):
        markdown = st.session_state["markdown"]
        st.success("Using the Markdown converted in Phase 1.")
    else:
        up = st.file_uploader("Converted COC Markdown", type=["md", "markdown", "txt"])
        if up:
            markdown = up.getvalue().decode("utf-8", errors="replace")
            st.success(f"Loaded {up.name}.")

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
st.subheader("Denied service")

PROC_SYSTEMS = {"cpt", "hcpcs"}


def _label(entry) -> str:
    return f"{entry.code} — {entry.descriptor[:70]}" if entry.descriptor else entry.code


picked_proc = picked_diag = None
if resolver and resolver.entries:
    entries = list(resolver.entries.values())
    procs = [e for e in entries if e.system in PROC_SYSTEMS] or entries
    diags = [e for e in entries if e.system == "icd10"]

    mode = st.radio(
        "Code entry", ["Pick from the loaded code table", "Type a code manually"],
        horizontal=True, label_visibility="collapsed",
    )
    if mode.startswith("Pick"):
        proc_label = st.selectbox(
            "Procedure code", [_label(e) for e in procs],
            help="Loaded from your code table. Selecting one fills the descriptor "
                 "and synonyms below; both stay editable.",
        )
        picked_proc = procs[[_label(e) for e in procs].index(proc_label)]
        if diags:
            diag_labels = ["(none)"] + [_label(e) for e in diags]
            diag_label = st.selectbox("Diagnosis code (optional)", diag_labels)
            if diag_label != "(none)":
                picked_diag = diags[diag_labels.index(diag_label) - 1]
else:
    st.caption("Upload a code table in the sidebar to pick codes from a list.")

# Widget keys include the selected code so the fields reset when the code
# changes -- otherwise a descriptor from a previous selection silently persists
# and gets searched against the wrong code.
key_suffix = picked_proc.code if picked_proc else "manual"

c1, c2 = st.columns([1, 3])
with c1:
    code = st.text_input("Procedure / diagnosis code",
                         value=picked_proc.code if picked_proc else "",
                         placeholder="15830", key=f"code_{key_suffix}")
    kind = classify_code(code) if code else ""
    if code:
        st.caption(f"Detected format: **{kind}**" if kind != "unknown"
                   else "Format not recognised — descriptor is required.")
with c2:
    descriptor = st.text_input(
        "Service descriptor (required)",
        value=picked_proc.descriptor if picked_proc else "",
        placeholder="Excision, excess skin and subcutaneous tissue, abdomen",
        key=f"desc_{key_suffix}",
        help="From your licensed code set. This is what gets matched against "
             "plan language — a misremembered descriptor produces a confidently "
             "wrong citation.",
    )

d1, d2 = st.columns(2)
with d1:
    diag_code = st.text_input("Diagnosis code (optional)",
                              value=picked_diag.code if picked_diag else "",
                              placeholder="E66.01", key=f"dxc_{key_suffix}")
    diag_desc = st.text_input("Diagnosis descriptor (optional)",
                              value=picked_diag.descriptor if picked_diag else "",
                              placeholder="Morbid (severe) obesity due to excess calories",
                              key=f"dxd_{key_suffix}")
with d2:
    synonyms = st.text_area(
        "Clinical synonyms (optional, one per line)", height=118,
        value="\n".join(picked_proc.synonyms) if picked_proc else "",
        placeholder="abdominoplasty\npanniculectomy",
        key=f"syn_{key_suffix}",
        help="Plan language rarely uses code-set vocabulary. Adding the terms a "
             "plan would use materially improves recall.",
    )

run = st.button("Find governing provision", type="primary", disabled=not (code and descriptor))
if code and not descriptor:
    st.info("Enter the service descriptor to search.", icon="ℹ️")

# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
if run:
    descriptors = [descriptor] + ([diag_desc] if diag_desc else [])
    extra = [s.strip() for s in synonyms.splitlines() if s.strip()]
    terms = build_query_terms(descriptors, extra)
    core_terms = build_core_terms(descriptors, extra)
    codes = [normalize_code(code)] + ([normalize_code(diag_code)] if diag_code else [])

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

            key = (settings.voyage_api_key if settings.embed_provider == "voyage"
                   else settings.openai_api_key)
            embedder = _cached_embedder(settings.embed_provider, settings.embed_model,
                                        int(settings.embed_dim), key or "")
            coll = _cached_collection(settings.mongodb_uri, settings.mongodb_db,
                                      settings.mongodb_collection)
            with st.spinner("Searching plan language…"):
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
                 "**Not scoped to a document** — results may come from any "
                 "indexed plan. ")
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

    for note in result.notes:
        st.warning(note, icon="❗")

    if provisions:
        st.subheader("Candidate provisions")
        for i, p in enumerate(provisions, start=1):
            conflict = " · ⛔ CONFLICT" if p.has_conflict else ""
            exception = " · ⚠️ exception" if p.has_exception else ""
            with st.expander(
                f"{i}. {p.citation} — score {p.score:.2f}"
                f"{f' · rerank {p.rerank_score:.3f}' if p.rerank_score is not None else ''}"
                f"{conflict}{exception}",
                expanded=(i == 1),
            ):
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
                m1.caption(f"**Pages:** {p.page_start}–{p.page_end}  ·  "
                           f"**Type:** {p.source_type}")
                m2.caption(f"**Matched terms:** {', '.join(p.matched_terms) or '—'}")
                m2.caption(f"**Distinctive:** {', '.join(p.distinctive_terms) or 'none'}")
                for c in p.conflicts:
                    st.error(f"Qualifier conflict: {c}", icon="⛔")
                if p.exception_clause:
                    st.warning(f"Exception clause: *{p.exception_clause}*", icon="⚠️")
                if p.cross_references:
                    st.info("Cross-references: " + ", ".join(p.cross_references), icon="🔗")
                if p.context and p.context.strip() != p.text.strip():
                    with st.popover("Surrounding text"):
                        st.markdown(p.context)

        st.subheader("Draft citation block")
        st.caption("Verbatim plan language for reviewer verification. Not a letter.")
        block = format_citation_block(result)
        st.code(block, language="text")
        st.download_button(
            "Download citation block",
            (f"Plan document: {selected_doc}\n" if selected_doc else "")
            + f"Code(s): {', '.join(result.codes)}\n"
            f"Descriptor: {'; '.join(result.descriptors)}\n"
            f"Outcome: {result.outcome}\n\n{block}\n\n"
            + "\n".join(f"NOTE: {n}" for n in result.notes)
            + "\n\nReviewed by: ____________________  Date: __________\n",
            file_name=f"provision_{'_'.join(result.codes)}.txt",
        )
    else:
        st.info(
            "Nothing matched. Try adding the clinical synonyms a plan document "
            "would use — plan language rarely mirrors code-set vocabulary.",
            icon="🔍",
        )

    if result.needs_human_review:
        st.error(
            "**Escalate to a qualified reviewer before drafting.** Either the "
            "match is not clear-cut, or the provision carries an exception that "
            "may apply on these clinical facts.",
            icon="🛑",
        )
