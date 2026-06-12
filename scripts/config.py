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


def duplicate_api_url() -> str:
    load_env()
    return env_value("DUPLICATE_API_URL", "http://localhost:5000").rstrip("/")


def embedding_model_name() -> str:
    load_env()
    return env_value("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def duplicate_match_threshold() -> float:
    load_env()
    return float(env_value("DUPLICATE_MATCH_THRESHOLD", "0.88"))


def duplicate_review_threshold() -> float:
    load_env()
    return float(env_value("DUPLICATE_REVIEW_THRESHOLD", "0.78"))


def duplicate_window_days() -> int:
    load_env()
    return int(env_value("DUPLICATE_WINDOW_DAYS", "60"))


def max_candidate_products() -> int:
    load_env()
    return int(env_value("DUPLICATE_MAX_CANDIDATES", "5000"))
