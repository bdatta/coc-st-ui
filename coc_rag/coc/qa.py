"""
Phase 3: retrieval-augmented Q&A over the vectorized COC.

The answering prompt is deliberately conservative. Benefit questions have real
consequences for the person asking, so the model is told to answer only from
retrieved COC text, to quote plan language for anything numeric, and to say when
the document does not settle the question rather than inferring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .vectorstore import (
    DEFAULT_TEXT_INDEX,
    DEFAULT_VECTOR_INDEX,
    fetch_neighbors,
    hybrid_search,
    vector_search,
)

SYSTEM_PROMPT = """You answer questions about a health benefit plan using ONLY the \
excerpts provided from its Certificate of Coverage (COC).

Rules:
- Ground every statement in the excerpts. Never use outside knowledge about how \
plans "usually" work.
- Cite the source number(s) inline like [1] or [2][3] after each claim.
- For any dollar amount, percentage, day/visit limit, or deductible, reproduce the \
plan's exact figure and state the network tier (in-network vs out-of-network) it \
applies to.
- Tables often encode the answer. Read column headers carefully; a value is \
meaningless without its column.
- If the excerpts do not contain the answer, say so plainly and name the section \
of the COC the user should check. Do not guess.
- Note when an answer depends on conditions the excerpts mention but do not \
resolve (prior authorization, medical necessity, deductible status).
- Close by reminding the reader that plan administration governs and they should \
confirm with the plan administrator.
"""


@dataclass
class Source:
    index: int
    chunk_id: str
    breadcrumb: str
    text: str
    page_start: int
    page_end: int
    type: str
    score: float = 0.0
    table_id: Optional[str] = None
    form_code: Optional[str] = None
    printed_page_start: Optional[int] = None
    printed_page_end: Optional[int] = None

    @property
    def label(self) -> str:
        """Cite the printed page and its form code when the source is a bound
        compilation: several documents inside one PDF restart at page 1, so a
        page number alone does not identify a provision."""
        if self.printed_page_start:
            pages = (f"p. {self.printed_page_start}"
                     if self.printed_page_start == self.printed_page_end
                     else f"pp. {self.printed_page_start}-{self.printed_page_end}")
        else:
            pages = (f"p. {self.page_start}" if self.page_start == self.page_end
                     else f"pp. {self.page_start}-{self.page_end}")
        form = f"{self.form_code}, " if self.form_code else ""
        return f"{self.breadcrumb or 'Untitled section'} ({form}{pages})"


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[Source] = field(default_factory=list)
    model: str = ""
    used_hybrid: bool = False


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def retrieve(
    coll,
    embedder,
    question: str,
    k: int = 8,
    doc_id: Optional[str] = None,
    use_hybrid: bool = False,
    expand_neighbors: bool = False,
    vector_index: str = DEFAULT_VECTOR_INDEX,
    text_index: str = DEFAULT_TEXT_INDEX,
) -> List[dict]:
    qvec = embedder.embed_query(question)
    if use_hybrid:
        hits = hybrid_search(coll, question, qvec, k=k, doc_id=doc_id,
                             vector_index=vector_index, text_index=text_index)
    else:
        filters = {"doc_id": {"$eq": doc_id}} if doc_id else None
        hits = vector_search(coll, qvec, k=k, index_name=vector_index, filters=filters)

    if not expand_neighbors:
        return hits

    seen = {h["chunk_id"] for h in hits}
    expanded = list(hits)
    for h in hits:
        if h.get("type") == "table":
            continue  # tables are already self-contained
        for nb in fetch_neighbors(coll, h["doc_id"], h["seq"], window=1):
            if nb["chunk_id"] not in seen:
                nb["score"] = h.get("score", 0.0) * 0.5
                seen.add(nb["chunk_id"])
                expanded.append(nb)
    return expanded


def rerank(embedder_api_key: Optional[str], question: str, hits: Sequence[dict],
           top_n: int = 6, model: str = "rerank-2.5") -> List[dict]:
    """Optional cross-encoder rerank via Voyage. Silently no-ops if unavailable."""
    if not hits:
        return list(hits)
    try:
        import voyageai

        client = voyageai.Client(api_key=embedder_api_key or os.environ.get("VOYAGE_API_KEY"))
        docs = [h.get("embed_text") or h.get("text", "") for h in hits]
        res = client.rerank(question, docs, model=model, top_k=min(top_n, len(docs)))
        out = []
        for r in res.results:
            hit = dict(hits[r.index])
            hit["score"] = float(r.relevance_score)
            out.append(hit)
        return out
    except Exception:
        return list(hits)[:top_n]


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
def to_sources(hits: Sequence[dict]) -> List[Source]:
    return [
        Source(
            index=i,
            chunk_id=h.get("chunk_id", ""),
            breadcrumb=h.get("breadcrumb", ""),
            text=h.get("embed_text") or h.get("text", ""),
            page_start=int(h.get("page_start") or 0),
            page_end=int(h.get("page_end") or 0),
            type=h.get("type", "text"),
            score=float(h.get("score") or 0.0),
            table_id=h.get("table_id"),
            form_code=h.get("form_code"),
            printed_page_start=h.get("printed_page_start"),
            printed_page_end=h.get("printed_page_end"),
        )
        for i, h in enumerate(hits, start=1)
    ]


def build_context(sources: Sequence[Source], max_chars: int = 24_000) -> str:
    parts: List[str] = []
    used = 0
    for s in sources:
        block = f"[{s.index}] {s.label}\n{s.text.strip()}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n---\n".join(parts)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_answer(
    question: str,
    sources: Sequence[Source],
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 1200,
) -> str:
    context = build_context(sources)
    if not context.strip():
        return ("I could not find anything in the indexed COC that addresses that "
                "question. Try rephrasing it using the plan's own wording, or "
                "confirm the document was embedded.")

    user_msg = (
        f"COC excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above, with inline [n] citations."
    )

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=model or "claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model or "gpt-4.1-mini",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content or ""

    raise ValueError(f"Unknown answer provider: {provider}")


def ask(
    coll,
    embedder,
    question: str,
    *,
    k: int = 8,
    doc_id: Optional[str] = None,
    use_hybrid: bool = False,
    expand_neighbors: bool = False,
    do_rerank: bool = False,
    rerank_top_n: int = 6,
    answer_provider: str = "anthropic",
    answer_model: Optional[str] = None,
    answer_api_key: Optional[str] = None,
    voyage_api_key: Optional[str] = None,
) -> Answer:
    hits = retrieve(coll, embedder, question, k=k, doc_id=doc_id,
                    use_hybrid=use_hybrid, expand_neighbors=expand_neighbors)
    if do_rerank:
        hits = rerank(voyage_api_key, question, hits, top_n=rerank_top_n)
    sources = to_sources(hits)
    text = generate_answer(question, sources, provider=answer_provider,
                           model=answer_model, api_key=answer_api_key)
    return Answer(question=question, answer=text, sources=sources,
                  model=answer_model or "", used_hybrid=use_hybrid)
