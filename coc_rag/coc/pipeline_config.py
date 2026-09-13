"""
Pipeline configuration.

Resolution order, lowest priority first: built-in defaults, environment (and
`.env`), the YAML file, then command-line arguments. YAML sits above environment
so a per-document file can pin conversion settings that differ from the
machine's defaults, while CLI flags still win for one-off runs.

**Secrets never live in YAML.** Connection URIs and API keys are read from the
environment only. A YAML file describes how a document should be processed and
is safe to commit next to the document; a `.env` holds credentials and is not.
Any secret-looking key found in YAML is rejected rather than silently used, so a
mistake surfaces at load time instead of in a commit.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    # Search the working directory upward first, so a per-project .env wins,
    # then fall back to the one beside this package. Without the fallback,
    # running a CLI tool from an unrelated folder silently ignores the .env
    # sitting in the repository and reports missing credentials.
    load_dotenv()
    _repo_env = pathlib.Path(__file__).resolve().parents[1] / ".env"
    if _repo_env.exists():
        load_dotenv(_repo_env, override=False)
except Exception:  # pragma: no cover
    pass

#: Keys that must come from the environment. Present in YAML, they are an error.
FORBIDDEN_YAML_KEYS = {
    "mongodb_uri", "voyage_api_key", "openai_api_key", "anthropic_api_key",
    "api_key", "password", "secret", "token",
}


@dataclass
class PipelineConfig:
    # --- source -----------------------------------------------------------
    pdf: Optional[str] = None
    outdir: str = "build"
    doc_id: Optional[str] = None
    toc: Optional[str] = None
    first_page: Optional[int] = None
    last_page: Optional[int] = None

    # --- conversion -------------------------------------------------------
    treat_bold_as_heading: bool = False
    treat_allcaps_as_heading: bool = False
    heading_size_ratio: float = 1.08
    table_strategy: str = "auto"
    table_format: str = "pipe"
    stitch_tables: bool = True
    detect_form_codes: bool = True
    emit_row_sentences: bool = False
    force_heading_patterns: List[str] = field(default_factory=list)

    # --- chunking ---------------------------------------------------------
    target_tokens: int = 650
    max_tokens: int = 900
    overlap_tokens: int = 90
    max_table_tokens: int = 1100
    include_row_sentences: bool = True
    prefix_breadcrumb: bool = True

    # --- plan identity ----------------------------------------------------
    plan_id: Optional[str] = None
    group_number: Optional[str] = None
    plan_year: Optional[int] = None
    effective_date: Optional[str] = None
    termination_date: Optional[str] = None
    carrier: Optional[str] = None
    market_segment: Optional[str] = None
    states: List[str] = field(default_factory=list)

    # --- embedding --------------------------------------------------------
    embed_provider: str = "voyage"
    embed_model: str = "voyage-3.5"
    embed_dim: int = 1024
    embed_batch_sleep: float = 0.0
    write_batch_size: int = 100
    write_throttle: float = 0.4

    # --- storage ----------------------------------------------------------
    mongodb_db: str = "coc_rag"
    mongodb_collection: str = "coc_chunks"
    vector_index: str = "coc_vector_index"
    text_index: str = "coc_text_index"
    create_text_index: bool = False
    store_document_record: bool = True
    replace_existing: bool = True

    # --- gates ------------------------------------------------------------
    #: Below this score the pipeline refuses to embed. Embedding a document that
    #: failed validation costs money and produces an index nobody should trust.
    min_validation_score: float = 90.0
    allow_review: bool = False
    require_full_toc_coverage: bool = True

    # --- secrets (environment only) --------------------------------------
    mongodb_uri: str = ""
    voyage_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ------------------------------------------------------------------ #
    @property
    def embed_api_key(self) -> str:
        return (self.voyage_api_key if self.embed_provider == "voyage"
                else self.openai_api_key)

    def missing_secrets(self, need_embedding: bool = True) -> List[str]:
        missing = []
        if not self.mongodb_uri:
            missing.append("MONGODB_URI")
        if need_embedding and self.embed_provider != "local" and not self.embed_api_key:
            missing.append("VOYAGE_API_KEY" if self.embed_provider == "voyage"
                           else "OPENAI_API_KEY")
        return missing

    def describe(self) -> str:
        redacted = {"mongodb_uri", "voyage_api_key", "openai_api_key",
                    "anthropic_api_key"}
        lines = []
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name in redacted:
                value = "(set)" if value else "(not set)"
            lines.append(f"   {f.name:<26} {value}")
        return "\n".join(lines)


def _from_env() -> Dict[str, Any]:
    def flag(name: str) -> Optional[bool]:
        raw = os.environ.get(name)
        return None if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")

    values: Dict[str, Any] = {
        "mongodb_uri": os.environ.get("MONGODB_URI", ""),
        "mongodb_db": os.environ.get("MONGODB_DB"),
        "mongodb_collection": os.environ.get("MONGODB_COLLECTION"),
        "vector_index": os.environ.get("VECTOR_INDEX_NAME"),
        "text_index": os.environ.get("TEXT_INDEX_NAME"),
        "embed_provider": os.environ.get("EMBED_PROVIDER"),
        "embed_model": os.environ.get("EMBED_MODEL"),
        "embed_dim": os.environ.get("EMBED_DIM"),
        "voyage_api_key": os.environ.get("VOYAGE_API_KEY", ""),
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "outdir": os.environ.get("WORK_DIR"),
    }
    text_index = flag("CREATE_TEXT_INDEX")
    if text_index is not None:
        values["create_text_index"] = text_index
    return {k: v for k, v in values.items() if v not in (None, "")}


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    data = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")

    # Flatten one level of grouping so the file can be organised into sections.
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value

    offending = sorted(k for k in flat if k.lower() in FORBIDDEN_YAML_KEYS)
    if offending:
        raise ValueError(
            f"{path} contains secret(s) {offending}. Put credentials in .env; a "
            "YAML config describes how to process a document and should be safe "
            "to commit alongside it."
        )

    known = {f.name for f in dataclasses.fields(PipelineConfig)}
    unknown = sorted(set(flat) - known)
    if unknown:
        raise ValueError(f"{path} has unknown setting(s): {unknown}")
    return flat


def build_config(yaml_path: Optional[str] = None,
                 overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    values: Dict[str, Any] = {}
    values.update(_from_env())
    if yaml_path:
        values.update(load_yaml(yaml_path))
    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = value

    known = {f.name: f for f in dataclasses.fields(PipelineConfig)}
    coerced: Dict[str, Any] = {}
    for key, value in values.items():
        spec = known.get(key)
        if spec is None:
            continue
        if spec.type in ("int", "Optional[int]") and value is not None:
            coerced[key] = int(value)
        elif spec.type in ("float", "Optional[float]") and value is not None:
            coerced[key] = float(value)
        elif spec.type == "List[str]" and isinstance(value, str):
            coerced[key] = [v.strip() for v in value.split(",") if v.strip()]
        else:
            coerced[key] = value
    return PipelineConfig(**coerced)
