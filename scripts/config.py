"""Shared runtime configuration loaded from .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")

    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


def env_value(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Missing required setting in .env: {name}")
    return value


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    name: str
    user: str
    password: str


def database_settings() -> DatabaseSettings:
    load_env()
    return DatabaseSettings(
        host=env_value("TULIP_SOURCE_DB_HOST"),
        port=int(env_value("TULIP_SOURCE_DB_PORT", "5432")),
        name=env_value("TULIP_SOURCE_DB_NAME"),
        user=env_value("TULIP_SOURCE_DB_USER"),
        password=env_value("TULIP_SOURCE_DB_PASSWORD"),
    )


def artifacts_dir() -> Path:
    load_env()
    return BASE_DIR / env_value("CATEGORIZATION_ARTIFACTS_DIR", "categorization_artifacts")


def embedding_model_name() -> str:
    load_env()
    return env_value("EMBEDDING_MODEL", "BAAI/bge-m3")


def embedding_device() -> str:
    load_env()
    return env_value("EMBEDDING_DEVICE", "auto").strip().lower()


def duplicate_retrieval_threshold() -> float:
    load_env()
    return float(env_value("DUPLICATE_RETRIEVAL_THRESHOLD", "0.35"))


def duplicate_match_threshold() -> float:
    """Score at which an embedding-only match is treated as a duplicate."""
    load_env()
    return float(env_value("DUPLICATE_MATCH_THRESHOLD", "0.88"))


def duplicate_review_threshold() -> float:
    """Score at which an embedding-only match requires manual review."""
    load_env()
    return float(env_value("DUPLICATE_REVIEW_THRESHOLD", "0.78"))


def duplicate_window_days() -> int:
    load_env()
    return int(env_value("DUPLICATE_WINDOW_DAYS", "60"))


def max_candidate_products() -> int:
    load_env()
    return int(env_value("DUPLICATE_MAX_CANDIDATES", "100"))


def rerank_model_name() -> str:
    load_env()
    return env_value("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual").strip()


def rerank_enabled() -> bool:
    load_env()
    return env_value("RERANK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def rerank_device() -> str:
    load_env()
    return env_value("RERANK_DEVICE", "cpu").strip().lower()


def rerank_top_k() -> int:
    load_env()
    return int(env_value("RERANK_TOP_K", "20"))


def rerank_match_threshold() -> float:
    load_env()
    return float(env_value("RERANK_MATCH_THRESHOLD", "0.85"))


def rerank_review_threshold() -> float:
    load_env()
    return float(env_value("RERANK_REVIEW_THRESHOLD", "0.70"))


def semantic_category_match_enabled() -> bool:
    """Enable semantic verification whenever its API endpoint is configured."""
    return bool(semantic_api_url())


def semantic_api_url() -> str:
    load_env()
    return os.environ.get("SEMANTIC_API_URL", "").strip()


def semantic_api_key() -> str:
    load_env()
    return os.environ.get("SEMANTIC_API_KEY", "").strip()


def semantic_model() -> str:
    load_env()
    return env_value("SEMANTIC_MODEL", "Qwen3-30B-A3B-Instruct").strip()


def semantic_timeout_seconds() -> int:
    load_env()
    return int(env_value("SEMANTIC_TIMEOUT_SECONDS", "5"))


def openai_api_key() -> str:
    load_env()
    return os.environ.get("OPENAI_API_KEY", "").strip()


def openai_model() -> str:
    load_env()
    return env_value("OPENAI_MODEL", "gpt-5-mini").strip()


def openai_timeout_seconds() -> int:
    load_env()
    return int(env_value("OPENAI_TIMEOUT_SECONDS", "30"))


def llm_max_candidates() -> int:
    """Maximum reranked candidates sent to an LLM for final verification."""
    load_env()
    return max(1, int(env_value("LLM_MAX_CANDIDATES", "5")))
