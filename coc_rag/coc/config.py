"""Environment-backed settings shared by all three Streamlit pages."""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Optional

try:  # optional convenience
    from dotenv import load_dotenv

    # Working directory first, then the .env beside this package, so a tool run
    # from an unrelated folder still finds the repository's credentials.
    load_dotenv()
    _repo_env = pathlib.Path(__file__).resolve().parents[1] / ".env"
    if _repo_env.exists():
        load_dotenv(_repo_env, override=False)
except Exception:  # pragma: no cover
    pass


@dataclass
class Settings:
    mongodb_uri: str = ""
    mongodb_db: str = "coc_rag"
    mongodb_collection: str = "coc_chunks"
    vector_index: str = "coc_vector_index"
    text_index: str = "coc_text_index"

    embed_provider: str = "voyage"
    embed_model: str = "voyage-3.5"
    embed_dim: int = 1024
    voyage_api_key: str = ""
    openai_api_key: str = ""

    answer_provider: str = "anthropic"
    answer_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""

    work_dir: str = "./data"


def load_settings() -> Settings:
    s = Settings(
        mongodb_uri=os.environ.get("MONGODB_URI", ""),
        mongodb_db=os.environ.get("MONGODB_DB", "coc_rag"),
        mongodb_collection=os.environ.get("MONGODB_COLLECTION", "coc_chunks"),
        vector_index=os.environ.get("VECTOR_INDEX_NAME", "coc_vector_index"),
        text_index=os.environ.get("TEXT_INDEX_NAME", "coc_text_index"),
        embed_provider=os.environ.get("EMBED_PROVIDER", "voyage"),
        embed_model=os.environ.get("EMBED_MODEL", "voyage-3.5"),
        embed_dim=int(os.environ.get("EMBED_DIM", "1024") or 1024),
        voyage_api_key=os.environ.get("VOYAGE_API_KEY", ""),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        answer_provider=os.environ.get("ANSWER_PROVIDER", "anthropic"),
        answer_model=os.environ.get("ANSWER_MODEL", "claude-sonnet-4-6"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        work_dir=os.environ.get("WORK_DIR", "./data"),
    )
    os.makedirs(s.work_dir, exist_ok=True)
    return s
