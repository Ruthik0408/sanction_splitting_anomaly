"""LLM providers and fallback routing for same-product matching."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from scripts.config import (
    llm_max_candidates,
    openai_api_key,
    openai_model,
    openai_timeout_seconds,
    semantic_api_key,
    semantic_api_url,
    semantic_model,
    semantic_timeout_seconds,
)


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class VllmCategorySettings:
    api_url: str
    api_key: str
    model: str
    timeout_seconds: int


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    model: str
    timeout_seconds: int
    api_url: str = "https://api.openai.com/v1/responses"


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
    if url.endswith("/v1/models"):
        return f"{url[:-len('/models')]}/chat/completions"
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


def configured_openai_settings() -> OpenAISettings | None:
    api_key = openai_api_key()
    if not api_key:
        return None
    return OpenAISettings(
        api_key=api_key,
        model=openai_model(),
        timeout_seconds=openai_timeout_seconds(),
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


def _product_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index + 1,
            "bill_id": candidate.get("bill_id"),
            "product_name": candidate.get("product_name"),
            "similarity_score": candidate.get("similarity_score"),
            "semantic_rank": candidate.get("semantic_rank"),
        }
        for index, candidate in enumerate(candidates[:llm_max_candidates()])
    ]


def _system_prompt() -> str:
    return (
        "You are a strict procurement duplicate auditor. Compare the input product name "
        "with every candidate product name. Mark same_product true only when they describe "
        "the same underlying item or a clear renamed equivalent. Do not mark broad category "
        "matches, substitutes, accessories, different dosage/strength, size, pack quantity, "
        "capacity, duration, or product type as the same. Return one decision per candidate."
    )


def _user_prompt(input_product_name: str, candidates: list[dict[str, Any]]) -> str:
    return (
        "Return exactly this JSON shape: "
        '{"matches":[{"rank":1,"same_product":true,'
        '"confidence":0.0,"reason":"short reason"}]}. '
        f"Data: {json.dumps({'input_product_name': input_product_name, 'candidates': candidates}, ensure_ascii=True)}"
    )


def _normalized_result(
    data: dict[str, Any], *, provider: str, model: str, expected_count: int
) -> dict[str, Any]:
    matches = data.get("matches")
    if not isinstance(matches, list):
        raise ValueError(f"{provider} response field 'matches' must be a list")
    normalized_matches = [
        {
            "rank": int(item.get("rank")),
            "same_product": is_true(item.get("same_product")),
            "confidence": round(bounded_confidence(item.get("confidence")), 4),
            "reason": str(item.get("reason") or "").strip(),
        }
        for item in matches
        if isinstance(item, dict) and str(item.get("rank", "")).isdigit()
    ]
    actual_ranks = [item["rank"] for item in normalized_matches]
    expected_ranks = list(range(1, expected_count + 1))
    if sorted(actual_ranks) != expected_ranks:
        raise ValueError(
            f"{provider} must return exactly one decision for ranks 1 through {expected_count}"
        )
    return {
        "available": True,
        "provider": provider,
        "model": model,
        "candidate_count": expected_count,
        "matches": normalized_matches,
    }


def vllm_same_product_match(
    *,
    settings: VllmCategorySettings,
    input_product_name: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    product_candidates = _product_candidates(candidates)
    payload = {
        "model": settings.model,
        "temperature": 0,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "system",
                "content": _system_prompt() + " Return JSON only.",
            },
            {
                "role": "user",
                "content": _user_prompt(input_product_name, product_candidates),
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
    return _normalized_result(
        data,
        provider="vllm",
        model=settings.model,
        expected_count=len(product_candidates),
    )


def openai_same_product_match(
    *,
    settings: OpenAISettings,
    input_product_name: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    product_candidates = _product_candidates(candidates)
    payload = {
        "model": settings.model,
        "store": False,
        "input": [
            {"role": "developer", "content": _system_prompt()},
            {
                "role": "user",
                "content": _user_prompt(input_product_name, product_candidates),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "same_product_matches",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "matches": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "rank": {"type": "integer"},
                                    "same_product": {"type": "boolean"},
                                    "confidence": {"type": "number"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["rank", "same_product", "confidence", "reason"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["matches"],
                    "additionalProperties": False,
                },
            }
        },
    }
    request = urllib.request.Request(
        settings.api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
        response_data = json.loads(response.read().decode("utf-8", "replace"))

    output_text = next(
        (
            part.get("text", "")
            for output in response_data.get("output", [])
            if output.get("type") == "message"
            for part in output.get("content", [])
            if part.get("type") == "output_text"
        ),
        "",
    )
    if not output_text:
        raise ValueError("OpenAI response did not contain output text")
    return _normalized_result(
        extract_json_object(output_text),
        provider="openai",
        model=settings.model,
        expected_count=len(product_candidates),
    )


def same_product_match(
    *,
    provider: str,
    input_product_name: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run one provider, or vLLM then OpenAI when provider is auto."""
    providers = ("vllm", "openai") if provider == "auto" else (provider,)
    failures: list[str] = []

    for candidate_provider in providers:
        try:
            if candidate_provider == "vllm":
                settings = configured_vllm_category_settings()
                if settings is None:
                    raise RuntimeError("SEMANTIC_API_URL is not configured")
                result = vllm_same_product_match(
                    settings=settings,
                    input_product_name=input_product_name,
                    candidates=candidates,
                )
            elif candidate_provider == "openai":
                settings = configured_openai_settings()
                if settings is None:
                    raise RuntimeError("OPENAI_API_KEY is not configured")
                result = openai_same_product_match(
                    settings=settings,
                    input_product_name=input_product_name,
                    candidates=candidates,
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {candidate_provider}")

            result["requested_provider"] = provider
            result["fallback_reason"] = "; ".join(failures)
            return result
        except Exception as exc:
            failures.append(f"{candidate_provider}: {exc}")

    return unavailable_category_match(
        "; ".join(failures) or "No LLM provider is configured.",
        requested_provider=provider,
    )


def unavailable_category_match(reason: str, requested_provider: str = "auto") -> dict[str, Any]:
    return {
        "available": False,
        "candidate_count": 0,
        "matches": [],
        "reason": reason,
        "model": "",
        "provider": "",
        "requested_provider": requested_provider,
        "fallback_reason": "",
    }
