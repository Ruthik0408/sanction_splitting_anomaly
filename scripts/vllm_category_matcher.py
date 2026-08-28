"""OpenAI-compatible vLLM same-product matching for duplicate candidates."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from scripts.config import semantic_api_key, semantic_api_url, semantic_model, semantic_timeout_seconds


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class VllmCategorySettings:
    api_url: str
    api_key: str
    model: str
    timeout_seconds: int


def chat_completions_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed = urllib.parse.urlsplit(url)
    if not parsed.hostname:
        raise ValueError(f"Invalid SEMANTIC_API_URL: {value!r}")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def configured_vllm_category_settings() -> VllmCategorySettings | None:
    api_url = chat_completions_url(semantic_api_url())
    if not api_url:
        return None
    return VllmCategorySettings(
        api_url=api_url,
        api_key=semantic_api_key(),
        model=semantic_model(),
        timeout_seconds=semantic_timeout_seconds(),
    )


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(text)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("vLLM response must be a JSON object")
    return data


def bounded_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def same_product_match(
    *,
    settings: VllmCategorySettings,
    input_product_name: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    product_candidates = [
        {
            "rank": index + 1,
            "bill_id": candidate.get("bill_id"),
            "product_name": candidate.get("product_name"),
            "similarity_score": candidate.get("similarity_score"),
            "semantic_rank": candidate.get("semantic_rank"),
        }
        for index, candidate in enumerate(candidates[:20])
    ]
    payload = {
        "model": settings.model,
        "temperature": 0,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "system",
                        "content": (
                            "You are a strict procurement duplicate auditor. Compare the input product "
                            "name with every candidate product name. Mark same_product true only when "
                            "they describe the same underlying item or a clear renamed equivalent. "
                            "Do not mark broad category matches, substitutes, accessories, different "
                            "dosage/strength, size, pack quantity, capacity, duration, or product type "
                            "as the same. Return one decision for every candidate and JSON only."
                        ),
            },
            {
                "role": "user",
                "content": (
                    "Return exactly this JSON shape: "
                    "{\"matches\":[{\"rank\":1,\"same_product\":true,"
                    "\"confidence\":0.0,\"reason\":\"short reason\"}]}. "
                    f"Data: {json.dumps({'input_product_name': input_product_name, 'candidates': product_candidates}, ensure_ascii=True)}"
                ),
            },
        ],
    }
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    request = urllib.request.Request(
        settings.api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
        response_data = json.loads(response.read().decode("utf-8", "replace"))

    content = response_data["choices"][0]["message"]["content"]
    data = extract_json_object(content)
    matches = data.get("matches")
    if not isinstance(matches, list):
        raise ValueError("vLLM response field 'matches' must be a list")
    return {
        "available": True,
        "model": settings.model,
        "matches": [
            {
                "rank": int(item.get("rank")),
                "same_product": is_true(item.get("same_product")),
                "confidence": round(bounded_confidence(item.get("confidence")), 4),
                "reason": str(item.get("reason") or "").strip(),
            }
            for item in matches
            if isinstance(item, dict) and str(item.get("rank", "")).isdigit()
        ],
    }


def unavailable_category_match(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "matches": [],
        "reason": reason,
        "model": "",
    }
