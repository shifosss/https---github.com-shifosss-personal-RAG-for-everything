from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONTEXTD_",
        env_file=None,
        frozen=True,
        extra="ignore",
    )

    data_root: Path = Field(
        default_factory=lambda: Path.home() / ".contextd",
        alias="CONTEXTD_HOME",
    )
    default_corpus: str = "personal"
    log_level: str = "INFO"
    schema_version: int = 1

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16

    retrieval_default_limit: int = 10
    retrieval_dense_top_k: int = 50
    retrieval_sparse_top_k: int = 50
    retrieval_rrf_k: int = 60
    retrieval_rerank_top_k: int = 50
    retrieval_rewrite_enabled: bool = False  # PRD D-30: off by default in v0.1
    retrieval_rerank_enabled: bool = True
    retrieval_rewrite_timeout_ms: int = 3000
    retrieval_rerank_timeout_ms: int = 5000

    reranker_provider: str = "anthropic"
    reranker_model: str = "claude-haiku-4-5"
    rewriter_model: str = "claude-haiku-4-5"

    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8787


@lru_cache
def get_settings() -> Settings:
    return Settings()
