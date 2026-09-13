"""
Embedding providers.

All default providers are HTTP API clients, so no model weights are downloaded
at runtime -- that was an explicit requirement. A local sentence-transformers
option is included for air-gapped deployments, but it expects the model to have
been baked into the image ahead of time (see README).

Voyage is the default because MongoDB owns Voyage AI, so it is the natural
pairing with Atlas Vector Search. `voyage-context-3` is worth benchmarking for
this corpus: it produces contextualized chunk embeddings, meaning each chunk's
vector also encodes its surrounding document context -- useful when a benefit
table row only makes sense under its section heading.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

# Rough per-request budgets so we stay inside provider limits.
MAX_BATCH_ITEMS = 96
MAX_BATCH_CHARS = 220_000

#: Transient network faults worth retrying rather than surfacing. WinError 10048
#: ("Only one usage of each socket address...") is Windows ephemeral-port
#: exhaustion: sockets linger in TIME_WAIT for two minutes, so a short backoff
#: usually clears it where an immediate retry will not.
TRANSIENT_MARKERS = (
    "10048", "10054", "10061", "connection reset", "connection aborted",
    "max retries exceeded", "temporarily unavailable", "timed out",
    "newconnectionerror", "remote end closed",
)
RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 2.0


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def with_retries(fn, *args, attempts: int = RETRY_ATTEMPTS,
                 base_delay: float = RETRY_BASE_DELAY, **kwargs):
    """Call `fn`, backing off on transient connection failures."""
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - provider SDKs wrap many types
            if not _is_transient(exc) or attempt == attempts - 1:
                raise
            last = exc
            time.sleep(base_delay * (2 ** attempt))
    if last:
        raise last


@dataclass
class EmbedResult:
    vectors: List[List[float]]
    model: str
    dim: int


def _batches(texts: Sequence[str], max_items: int = MAX_BATCH_ITEMS,
             max_chars: int = MAX_BATCH_CHARS):
    batch: List[str] = []
    size = 0
    for t in texts:
        t = t or " "
        if batch and (len(batch) >= max_items or size + len(t) > max_chars):
            yield batch
            batch, size = [], 0
        batch.append(t)
        size += len(t)
    if batch:
        yield batch


class BaseEmbedder:
    provider = "base"

    def __init__(self, model: str, dim: Optional[int] = None):
        self.model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_documents(
        self,
        texts: Sequence[str],
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Voyage
# --------------------------------------------------------------------------- #
class VoyageEmbedder(BaseEmbedder):
    provider = "voyage"
    #: models that accept an explicit output_dimension
    FLEXIBLE_DIMS = {
        "voyage-3-large", "voyage-3.5", "voyage-3.5-lite", "voyage-code-3",
        "voyage-context-3", "voyage-4-large", "voyage-4", "voyage-4-lite",
    }

    def __init__(self, model: str = "voyage-3.5", dim: Optional[int] = 1024,
                 api_key: Optional[str] = None, sleep: float = 0.0):
        super().__init__(model, dim)
        import voyageai  # noqa: F401  (imported lazily so the app starts without it)

        self.client = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))
        self.sleep = sleep
        self.contextual = model.startswith("voyage-context")

    def _kwargs(self) -> dict:
        if self._dim and self.model in self.FLEXIBLE_DIMS:
            return {"output_dimension": self._dim}
        return {}

    def _embed(self, texts: Sequence[str], input_type: str) -> List[List[float]]:
        if self.contextual:
            res = self.client.contextualized_embed(
                inputs=[list(texts)], model=self.model,
                input_type=input_type, **self._kwargs()
            )
            results = getattr(res, "results", None) or res["results"]
            first = results[0]
            return list(getattr(first, "embeddings", None) or first["embeddings"])
        res = self.client.embed(
            list(texts), model=self.model, input_type=input_type, **self._kwargs()
        )
        return list(getattr(res, "embeddings", None) or res["embeddings"])

    def embed_documents(self, texts, progress=None) -> List[List[float]]:
        out: List[List[float]] = []
        total = max(len(texts), 1)
        for batch in _batches(texts):
            out.extend(with_retries(self._embed, batch, "document"))
            if progress:
                progress(len(out) / total, f"Embedded {len(out)}/{total}")
            if self.sleep:
                time.sleep(self.sleep)
        return out

    def embed_query(self, text: str) -> List[float]:
        return with_retries(self._embed, [text], "query")[0]


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAIEmbedder(BaseEmbedder):
    provider = "openai"

    def __init__(self, model: str = "text-embedding-3-small",
                 dim: Optional[int] = 1536, api_key: Optional[str] = None,
                 sleep: float = 0.0):
        super().__init__(model, dim)
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.sleep = sleep

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        kwargs = {"dimensions": self._dim} if self._dim else {}
        res = self.client.embeddings.create(model=self.model, input=list(texts), **kwargs)
        return [d.embedding for d in sorted(res.data, key=lambda d: d.index)]

    def embed_documents(self, texts, progress=None) -> List[List[float]]:
        out: List[List[float]] = []
        total = max(len(texts), 1)
        for batch in _batches(texts):
            out.extend(with_retries(self._embed, batch))
            if progress:
                progress(len(out) / total, f"Embedded {len(out)}/{total}")
            if self.sleep:
                time.sleep(self.sleep)
        return out

    def embed_query(self, text: str) -> List[float]:
        return with_retries(self._embed, [text])[0]


# --------------------------------------------------------------------------- #
# Local (offline sentence-transformers)
# --------------------------------------------------------------------------- #
class LocalEmbedder(BaseEmbedder):
    """Expects a model directory already present on disk. Set
    HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 to guarantee no network use."""

    provider = "local"

    def __init__(self, model: str, dim: Optional[int] = None, **_):
        super().__init__(model, dim)
        from sentence_transformers import SentenceTransformer

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.st = SentenceTransformer(model)
        self._dim = dim or self.st.get_sentence_embedding_dimension()

    def embed_documents(self, texts, progress=None) -> List[List[float]]:
        vecs = self.st.encode(
            list(texts), batch_size=16, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        if progress:
            progress(1.0, f"Embedded {len(texts)}")
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        return self.st.encode([text], normalize_embeddings=True,
                              convert_to_numpy=True)[0].tolist()


PROVIDERS = {
    "voyage": VoyageEmbedder,
    "openai": OpenAIEmbedder,
    "local": LocalEmbedder,
}

SUGGESTED_MODELS = {
    "voyage": ["voyage-3.5", "voyage-context-3", "voyage-3-large", "voyage-3.5-lite"],
    "openai": ["text-embedding-3-small", "text-embedding-3-large"],
    "local": ["/models/bge-small-en-v1.5", "/models/all-MiniLM-L6-v2"],
}


def get_embedder(provider: str, model: str, dim: Optional[int] = None,
                 api_key: Optional[str] = None, sleep: float = 0.0) -> BaseEmbedder:
    cls = PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown embedding provider: {provider}")
    if provider == "local":
        return cls(model=model, dim=dim)
    return cls(model=model, dim=dim, api_key=api_key, sleep=sleep)
