"""Auditable text preparation and generic product-specification extraction."""

from __future__ import annotations

import re
SEMANTIC_NORMALIZATION_VERSION = "semantic-v2"
LEXICAL_NORMALIZATION_VERSION = "lexical-v2"
SPACE_RE = re.compile(r"\s+")
UNIT_ALIASES = {
    "litre": "l", "litres": "l", "liter": "l", "liters": "l", "ltr": "l", "ltrs": "l",
    "gigabyte": "gb", "gigabytes": "gb", "terabyte": "tb", "terabytes": "tb",
    "volt": "v", "volts": "v", "horsepower": "hp",
    "piece": "pcs", "pieces": "pcs", "pc": "pcs",
}
UNIT_PATTERN = "|".join(sorted((re.escape(value) for value in UNIT_ALIASES), key=len, reverse=True))
CANONICAL_UNITS = "mm|cm|m|ml|l|mg|g|kg|kb|mb|gb|tb|w|kw|v|kv|hp|hz|mah|ah|pcs"


def _normalize_units(text: str) -> str:
    text = re.sub(
        rf"(?<=\d)\s*(?:{UNIT_PATTERN})\b",
        lambda match: UNIT_ALIASES[match.group(0).strip()],
        text,
    )
    return re.sub(rf"(?<=\d)\s*({CANONICAL_UNITS})\b", r"\1", text)


def prepare_embedding_text(product_name: str) -> str:
    """Prepare model input while retaining identifiers and specification values."""
    original = product_name.strip().lower()
    text = original.replace("&", " and ").replace("×", "x")
    text = re.sub(r"(?<![a-z0-9])-|-(?![a-z0-9])", " ", text)
    text = re.sub(r"[^a-z0-9-]+", " ", text)
    text = _normalize_units(text)
    return SPACE_RE.sub(" ", text).strip() or original


def prepare_lexical_text(product_name: str) -> str:
    """Prepare fuzzy/token input, making identifier punctuation comparable."""
    text = prepare_embedding_text(product_name)
    lexical_text = SPACE_RE.sub(" ", re.sub(r"[^a-z0-9]+", " ", text)).strip()
    # A valid source name can contain only symbols or non-ASCII characters.
    # Preserve it rather than storing an unusable empty trigram value.
    return lexical_text or text


def normalize_product_text(product_name: str) -> str:
    """Backward-compatible alias for semantic preparation."""
    return prepare_embedding_text(product_name)

