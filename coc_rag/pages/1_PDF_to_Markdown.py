"""Phase 1 — upload a COC PDF and produce structured Markdown."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import time

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from coc.config import load_settings  # noqa: E402
from coc.pdf_to_markdown import ConversionOptions, convert_pdf  # noqa: E402

st.set_page_config(page_title="PDF → Markdown", page_icon="📄", layout="wide")

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
settings = st.session_state.settings

st.title("📄 Phase 1 — COC PDF to Markdown")
st.caption(
    "Tables are rebuilt from ruling-line geometry, so merged cells keep their "
    "row/column spans and tables continuing onto the next page are stitched "
    "back together before anything is written out."
)

uploaded = st.file_uploader("Certificate of Coverage (PDF)", type=["pdf"])

with st.sidebar:
    st.header("Conversion options")

    st.subheader("Tables")
    table_strategy = st.selectbox(
        "Detection strategy", ["auto", "lines", "lines_strict", "hybrid", "text"],
        help=(
            "`auto` tries ruled-line detection then a stricter variant. Use "
            "`text` only if the COC's tables have no ruling lines at all — it "
            "is much more prone to false positives on prose."
        ),
    )
    table_format = st.selectbox(
        "Markdown table format", ["pipe", "html"],
        help=(
            "`pipe` propagates merged values into every cell they cover, which "
            "reads better and embeds better. `html` keeps literal rowspan / "
            "colspan attributes for full structural fidelity."
        ),
    )
    emit_row_sentences = st.checkbox(
        "Also emit row sentences", value=False,
        help="Adds a 'Label: value | Label: value' line per row inside each table "
             "block. Improves retrieval on wide benefit grids at the cost of size.",
    )
    stitch_tables = st.checkbox("Stitch tables across page breaks", value=True)
    min_table_fill = st.slider("Minimum cell fill ratio", 0.0, 0.6, 0.15, 0.05,
                               help="Rejects sparse grids that are really prose.")

    st.subheader("Headings")
    heading_size_ratio = st.slider(
        "Heading font-size ratio", 1.00, 1.40, 1.08, 0.01,
        help="A line qualifies as a heading when its font is this much larger "
             "than body text. Lower it if headings are being missed.",
    )
    treat_allcaps = st.checkbox("Treat ALL CAPS lines as headings", value=True)
    treat_bold = st.checkbox("Treat bold short lines as headings", value=True)
    force_patterns_raw = st.text_area(
        "Force-heading regexes (one per line)",
        value=r"^SECTION\s+\d+" + "\n" + r"^ARTICLE\s+[IVX\d]+",
        height=90,
    )

    st.subheader("Page furniture")
    drop_running = st.checkbox("Remove running headers/footers", value=True)
    header_band = st.slider("Header/footer band (page fraction)", 0.04, 0.20, 0.10, 0.01)

    st.subheader("Range")
    use_range = st.checkbox("Limit page range", value=False,
                            help="Skip front matter, or convert a short range while tuning.")
    detected = st.session_state.get("page_numbering")
    first_default = detected.first_numbered_page if detected and detected.found else 1
    last_default = st.session_state.get("pdf_page_count", 20)
    first_page = st.number_input("First page", min_value=1, value=int(first_default),
                                 disabled=not use_range)
    last_page = st.number_input("Last page", min_value=1, value=int(last_default),
                                disabled=not use_range)

if uploaded and st.button("Detect where printed page numbering starts"):
    import pdfplumber

    from coc.layout import detect_page_numbering

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getbuffer())
        probe_path = tmp.name
    try:
        with st.spinner("Reading page footers…"):
            with pdfplumber.open(probe_path) as probe:
                st.session_state.pdf_page_count = len(probe.pages)
                st.session_state.page_numbering = detect_page_numbering(probe)
    finally:
        os.unlink(probe_path)

    det = st.session_state.page_numbering
    if det.found:
        st.success(
            f"Printed page 1 is **PDF page {det.first_numbered_page}** "
            f"(offset +{det.offset}), agreed by {det.consistent_pages} of "
            f"{det.pages_examined} pages — {det.confidence:.0%} confidence.\n\n"
            f"Tick **Limit page range** and set First page to "
            f"**{det.first_numbered_page}** to skip the cover and contents while "
            f"keeping every page of plan content.",
            icon="🔢",
        )
        if det.roman_pages:
            st.caption(f"Roman numerals also seen on PDF page(s) "
                       f"{', '.join(map(str, det.roman_pages))} — usually incidental.")
        st.caption(
            "Remember the offset when citing: an answer citing PDF page N refers "
            f"to printed page N−{det.offset} in the member's document."
        )
    else:
        st.warning(
            "No consistent page numbering found. The footers may be images, or "
            "the document may not carry printed numbers. Set the range manually.",
            icon="🔢",
        )

st.info(
    "Tip: run a 15–20 page range first and check the output, then tune the "
    "heading ratio and table strategy before committing to all 200 pages.",
    icon="💡",
)

if uploaded and st.button("Convert to Markdown", type="primary"):
    options = ConversionOptions(
        table_strategy=table_strategy,
        table_format=table_format,
        heading_size_ratio=heading_size_ratio,
        treat_allcaps_as_heading=treat_allcaps,
        treat_bold_as_heading=treat_bold,
        drop_running_lines=drop_running,
        header_band=header_band,
        stitch_tables=stitch_tables,
        min_table_fill=min_table_fill,
        emit_row_sentences=emit_row_sentences,
        force_heading_patterns=[p.strip() for p in force_patterns_raw.splitlines() if p.strip()],
        page_range=(int(first_page), int(last_page)) if use_range else None,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    bar = st.progress(0.0, text="Starting…")
    started = time.time()
    try:
        result = convert_pdf(
            tmp_path,
            options,
            progress=lambda frac, msg: bar.progress(min(max(frac, 0.0), 1.0), text=msg),
            doc_title=uploaded.name,
        )
    except Exception as exc:
        bar.empty()
        st.exception(exc)
        st.stop()
    finally:
        os.unlink(tmp_path)

    bar.progress(1.0, text=f"Done in {time.time() - started:.1f}s")

    st.session_state.markdown = result.markdown
    st.session_state.markdown_name = pathlib.Path(uploaded.name).stem + ".md"
    st.session_state.tables_json = result.tables_json
    st.session_state.conversion_stats = result.stats
    st.session_state.conversion_warnings = result.warnings

md = st.session_state.get("markdown")
if md:
    stats = st.session_state.get("conversion_stats", {})
    st.subheader("Result")

    c = st.columns(5)
    c[0].metric("Pages", stats.get("pages", 0))
    c[1].metric("Headings", stats.get("headings", 0))
    c[2].metric("Tables", stats.get("tables", 0))
    c[3].metric("With merged cells", stats.get("tables_with_spans", 0))
    c[4].metric("Spanning pages", stats.get("tables_spanning_pages", 0))

    for warning in st.session_state.get("conversion_warnings", []):
        st.warning(warning, icon="⚠️")

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download Markdown", md,
        file_name=st.session_state.get("markdown_name", "coc.md"),
        mime="text/markdown", type="primary", width="stretch",
    )
    d2.download_button(
        "Download table structures (JSON)",
        json.dumps(st.session_state.get("tables_json", []), indent=2),
        file_name="coc_tables.json", mime="application/json", width="stretch",
        help="Every cell with its true rowspan/colspan — useful for auditing "
             "fidelity against the source PDF.",
    )

    # Roughly 50 lines of text. Both previews scroll inside a fixed-height box so
    # a 450,000-character document cannot push the diagnostics off the screen.
    PREVIEW_HEIGHT = 900

    with st.expander("Preview the converted Markdown", expanded=False):
        tab_render, tab_raw = st.tabs(["Rendered", "Raw Markdown"])
        with tab_render:
            limit = st.slider("Preview characters", 2000, 60000, 12000, 2000)
            with st.container(height=PREVIEW_HEIGHT, border=False):
                st.markdown(md[:limit], unsafe_allow_html=True)
            if len(md) > limit:
                st.caption(f"Showing {limit:,} of {len(md):,} characters.")
        with tab_raw:
            st.code(md[:40000], language="markdown", height=PREVIEW_HEIGHT)
            if len(md) > 40000:
                st.caption(f"Showing 40,000 of {len(md):,} characters. "
                           "Download the file for the full text.")

    with st.container():
        st.subheader("Diagnostics")
        st.json(stats, expanded=False)
        tables = st.session_state.get("tables_json", [])
        if tables:
            st.dataframe(
                [{
                    "table_id": t["table_id"],
                    "pages": ", ".join(map(str, t["pages"])),
                    "rows": t["n_rows"],
                    "cols": t["n_cols"],
                    "merged cells": t["has_spans"],
                    "columns": " | ".join(t["column_labels"])[:120],
                } for t in tables],
                hide_index=True, width="stretch",
            )

    st.success("Markdown is held in session — continue on the Index page.", icon="✅")
    st.page_link("pages/2_Chunk_and_Embed.py", label="Go to Phase 2 — Chunk & Embed", icon="🧩")
