"""Product-name cleanup for semantic duplicate detection.

The project does not rely on a category column or hand-maintained synonym
dictionaries. This module only removes generic noise from product_name; semantic
meaning is left to the configured embedding model.
"""

from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[a-z0-9]+")
SPACE_RE = re.compile(r"\s+")

NOISE_TOKENS = {
    "and",
    "by",
    "for",
    "from",
    "in",
    "of",
    "or",
    "the",
    "to",
    "with",
    "without",
    "unbranded",
    "generic",
    "compatible",
    "type",
    "model",
    "series",
    "new",
}

def _drop_token(token: str) -> bool:
    if token in NOISE_TOKENS:
        return True
    if len(token) == 1 and not token.isdigit():
        return True
    return False


def normalize_product_text(product_name: str) -> str:
    """Return cleaned text derived only from product_name."""
    text = product_name.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = SPACE_RE.sub(" ", text).strip()

    original_tokens = TOKEN_RE.findall(text)
    normalized_tokens: list[str] = []

    for token in original_tokens:
        if _drop_token(token):
            continue
        normalized_tokens.append(token)

    deduped_tokens = list(dict.fromkeys(normalized_tokens))
    return " ".join(deduped_tokens) or product_name.strip().lower()
