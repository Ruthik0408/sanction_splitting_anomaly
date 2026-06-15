"""Audit same-unit purchases with an OpenAI-compatible LLM judge.

The script is intentionally separate from the FastAPI duplicate checker. It
filters existing data by hard fraud conditions first, then asks an LLM whether
candidate product names are the same underlying product or a renamed equivalent.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.config import BASE_DIR, duplicate_window_days, load_env


DEFAULT_BILL_CSV = BASE_DIR / "exports" / "gem_bill.csv"
DEFAULT_PRODUCT_CSV = BASE_DIR / "exports" / "gem_product.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "exports" / "llm_same_product_audit.csv"
DEFAULT_MODEL = "Qwen3-30B-A3B-Instruct"
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
OUTPUT_COLUMNS = [
    "row_id",
    "transaction_id",
    "order_id",
    "supply_order_date",
    "days_from_input",
    "fk_central_unit",
    "candidate_product_name",
    "product_brand",
    "quantity_ordered",
    "unit_price",
    "total_value",
    "quantity_unit_type",
    "vendor_name",
    "bill_amount",
    "amount_passed",
    "same_underlying_product",
    "renamed_equivalent",
    "close_substitute",
    "canonical_concept",
    "material_difference",
    "material_difference_reason",
    "fraud_relevance",
    "confidence",
    "reason",
    "local_validation_reason",
]
CONTRADICTION_RE = re.compile(
    r"\b("
    r"no match|not (?:a )?match|unrelated|different product type|"
    r"different use case|not relevant|no relevant match|clearly different"
    r")\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
GENERIC_TOKENS = {
    "and",
    "for",
    "from",
    "in",
    "of",
    "or",
    "the",
    "to",
    "up",
    "with",
    "without",
}
PRODUCT_CATEGORY_KEYWORDS = {
    "pen": {"pen", "gel", "ball", "rollerball", "ballpoint"},
    "marker": {"marker", "markers", "paint"},
    "printer_consumable": {"cartridge", "toner", "ribbon", "inkjet"},
    "paper": {"paper", "gsm", "sheet", "sheets", "ream"},
    "computer": {"computer", "desktop", "workstation", "pc"},
    "laptop": {"laptop", "notebook"},
    "software_support": {
        "software",
        "license",
        "licence",
        "subscription",
        "support",
        "amc",
        "warranty",
        "office",
        "eoffice",
    },
    "medical_test": {
        "elisa",
        "hbsag",
        "test",
        "kit",
        "kits",
        "reagent",
        "diagnostic",
        "diagnostics",
    },
}


@dataclass(frozen=True)
class AuditArgs:
    product_name: str
    fk_central_unit: int
    supply_order_date: date
    order_id: str
    transaction_id: str
    bill_csv: Path
    product_csv: Path
    output_csv: Path
    api_url: str
    api_key: str
    model: str
    window_days: int
    batch_size: int
    max_candidates: int
    timeout: int
    dry_run: bool


def parse_args() -> AuditArgs:
    env = load_env()
    parser = argparse.ArgumentParser(
        description=(
            "Find existing purchases from the same fk_central_unit/date window "
            "and use a vLLM/OpenAI-compatible model to judge renamed same products."
        )
    )
    parser.add_argument("--product-name", required=True)
    parser.add_argument("--fk-central-unit", type=int, required=True)
    parser.add_argument("--supply-order-date", type=date.fromisoformat, required=True)
    parser.add_argument("--order-id", default="")
    parser.add_argument("--transaction-id", default="")
    parser.add_argument("--bill-csv", type=Path, default=DEFAULT_BILL_CSV)
    parser.add_argument("--product-csv", type=Path, default=DEFAULT_PRODUCT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--api-url", default=env.get("SEMANTIC_API_URL", ""))
    parser.add_argument("--api-key", default=env.get("SEMANTIC_API_KEY", ""))
    parser.add_argument("--model", default=env.get("SEMANTIC_MODEL", DEFAULT_MODEL))
    parser.add_argument("--window-days", type=int, default=duplicate_window_days())
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write filtered candidates without calling the LLM.",
    )
    parsed = parser.parse_args()
    return AuditArgs(
        product_name=parsed.product_name.strip(),
        fk_central_unit=parsed.fk_central_unit,
        supply_order_date=parsed.supply_order_date,
        order_id=parsed.order_id.strip(),
        transaction_id=parsed.transaction_id.strip(),
        bill_csv=parsed.bill_csv,
        product_csv=parsed.product_csv,
        output_csv=parsed.output_csv,
        api_url=chat_completions_url(parsed.api_url),
        api_key=parsed.api_key.strip(),
        model=parsed.model.strip() or DEFAULT_MODEL,
        window_days=parsed.window_days,
        batch_size=parsed.batch_size,
        max_candidates=parsed.max_candidates,
        timeout=parsed.timeout,
        dry_run=parsed.dry_run,
    )


def chat_completions_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed = urllib.parse.urlsplit(url)
    if not parsed.hostname:
        raise ValueError(f"Invalid API URL: {value!r}")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def normalized_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def approved_mask(series: pd.Series) -> pd.Series:
    return normalized_text(series).str.lower().isin({"t", "true", "1", "yes"})


def load_candidates(args: AuditArgs) -> pd.DataFrame:
    bills = pd.read_csv(
        args.bill_csv,
        usecols=[
            "id",
            "transaction_id",
            "order_id",
            "supply_order_date",
            "record_status",
            "approved",
            "fk_central_unit",
            "vendor_name",
            "bill_amount",
            "amount_passed",
        ],
        low_memory=False,
    )
    products = pd.read_csv(
        args.product_csv,
        usecols=[
            "fk_gem_bill",
            "product_name",
            "product_brand",
            "quantity_ordered",
            "unit_price",
            "total_value",
            "quantity_unit_type",
        ],
        low_memory=False,
    )

    target_date = pd.Timestamp(args.supply_order_date)
    window_start = target_date - pd.Timedelta(days=args.window_days)
    window_end = target_date + pd.Timedelta(days=args.window_days)

    bills["supply_order_date"] = pd.to_datetime(bills["supply_order_date"], errors="coerce")
    bills["order_id"] = normalized_text(bills["order_id"])
    bills["transaction_id"] = normalized_text(bills["transaction_id"])
    bills = bills[
        (bills["record_status"] == "V")
        & approved_mask(bills["approved"])
        & (bills["fk_central_unit"] == args.fk_central_unit)
        & bills["supply_order_date"].between(window_start, window_end, inclusive="both")
        & (bills["transaction_id"] != "")
    ].copy()

    if args.order_id:
        bills = bills[bills["order_id"] != args.order_id]
    if args.transaction_id:
        bills = bills[bills["transaction_id"] != args.transaction_id]

    products = products[products["fk_gem_bill"].notna()].copy()
    products["bill_id"] = products["fk_gem_bill"].astype("int64")
    products["candidate_product_name"] = normalized_text(products["product_name"])
    products = products[products["candidate_product_name"] != ""]

    bills["bill_id"] = bills["id"].astype("int64")
    joined = products.merge(
        bills[
            [
                "bill_id",
                "transaction_id",
                "order_id",
                "supply_order_date",
                "fk_central_unit",
                "vendor_name",
                "bill_amount",
                "amount_passed",
            ]
        ],
        on="bill_id",
        how="inner",
    )
    joined["days_from_input"] = (
        joined["supply_order_date"] - target_date
    ).dt.days.abs()
    joined = joined.sort_values(
        ["days_from_input", "transaction_id", "candidate_product_name"]
    )
    joined = joined.drop_duplicates(
        ["transaction_id", "order_id", "candidate_product_name"]
    )
    if args.max_candidates > 0:
        joined = joined.head(args.max_candidates)
    return joined.reset_index(drop=True)


def batch_rows(candidates: pd.DataFrame, batch_size: int) -> list[list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for row_id, row in candidates.iterrows():
        records.append(
            {
                "row_id": int(row_id),
                "transaction_id": str(row["transaction_id"]),
                "order_id": str(row["order_id"]),
                "supply_order_date": row["supply_order_date"].date().isoformat(),
                "days_from_input": int(row["days_from_input"]),
                "product_name": str(row["candidate_product_name"]),
                "brand": "" if pd.isna(row.get("product_brand")) else str(row["product_brand"]),
                "quantity_ordered": safe_number(row.get("quantity_ordered")),
                "unit_price": safe_number(row.get("unit_price")),
                "total_value": safe_number(row.get("total_value")),
                "quantity_unit_type": ""
                if pd.isna(row.get("quantity_unit_type"))
                else str(row["quantity_unit_type"]),
                "vendor_name": "" if pd.isna(row.get("vendor_name")) else str(row["vendor_name"]),
            }
        )
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def safe_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def product_tokens(value: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(value.lower()) if token not in GENERIC_TOKENS}


def product_categories(tokens: set[str]) -> set[str]:
    return {
        category
        for category, keywords in PRODUCT_CATEGORY_KEYWORDS.items()
        if tokens & keywords
    }


def has_category_conflict(input_categories: set[str], candidate_categories: set[str]) -> bool:
    if not input_categories or not candidate_categories:
        return False
    if input_categories & candidate_categories:
        return False
    hard_conflicts = [
        {"pen", "marker"},
        {"computer", "laptop"},
        {"software_support", "medical_test"},
        {"pen", "medical_test"},
        {"marker", "medical_test"},
        {"paper", "medical_test"},
    ]
    return any(
        input_categories <= conflict or candidate_categories <= conflict
        for conflict in hard_conflicts
    )


def same_product_gate(input_product: str, candidate_product: str) -> tuple[bool, str]:
    input_clean = " ".join(TOKEN_RE.findall(input_product.lower()))
    candidate_clean = " ".join(TOKEN_RE.findall(candidate_product.lower()))
    input_tokens = product_tokens(input_product)
    candidate_tokens = product_tokens(candidate_product)
    if not input_tokens or not candidate_tokens:
        return False, "empty normalized product name"

    input_categories = product_categories(input_tokens)
    candidate_categories = product_categories(candidate_tokens)
    if has_category_conflict(input_categories, candidate_categories):
        return False, "candidate belongs to a conflicting product category"

    overlap = input_tokens & candidate_tokens
    overlap_ratio = len(overlap) / min(len(input_tokens), len(candidate_tokens))
    jaccard = len(overlap) / len(input_tokens | candidate_tokens)
    sequence_ratio = difflib.SequenceMatcher(None, input_clean, candidate_clean).ratio()

    if sequence_ratio >= 0.82:
        return True, "high normalized name similarity"
    if overlap_ratio >= 0.65 and jaccard >= 0.45:
        return True, "strong token overlap"
    if input_categories & candidate_categories and overlap_ratio >= 0.5 and sequence_ratio >= 0.62:
        return True, "same category with supporting token overlap"
    return False, (
        "insufficient same-product evidence "
        f"(overlap={overlap_ratio:.2f}, jaccard={jaccard:.2f}, name_similarity={sequence_ratio:.2f})"
    )


def llm_candidate_gate(input_product: str, candidate_product: str) -> tuple[bool, str]:
    strict_match, strict_reason = same_product_gate(input_product, candidate_product)
    if strict_match:
        return True, strict_reason

    input_clean = " ".join(TOKEN_RE.findall(input_product.lower()))
    candidate_clean = " ".join(TOKEN_RE.findall(candidate_product.lower()))
    input_tokens = product_tokens(input_product)
    candidate_tokens = product_tokens(candidate_product)
    input_categories = product_categories(input_tokens)
    candidate_categories = product_categories(candidate_tokens)
    if has_category_conflict(input_categories, candidate_categories):
        return False, "candidate belongs to a conflicting product category"

    overlap = input_tokens & candidate_tokens
    overlap_ratio = len(overlap) / min(len(input_tokens), len(candidate_tokens))
    sequence_ratio = difflib.SequenceMatcher(None, input_clean, candidate_clean).ratio()
    if overlap_ratio >= 0.35 or sequence_ratio >= 0.55 or input_categories & candidate_categories:
        return True, "possible same-product candidate for LLM review"
    return False, "too little product-name evidence for LLM review"


def filter_candidates_for_llm(input_product: str, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    keep_mask = candidates["candidate_product_name"].map(
        lambda candidate: llm_candidate_gate(input_product, str(candidate))[0]
    )
    return candidates[keep_mask].reset_index(drop=True)


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(text)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    return data


def call_llm(args: AuditArgs, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt_payload = {
        "input_product": args.product_name,
        "fk_central_unit": args.fk_central_unit,
        "input_supply_order_date": args.supply_order_date.isoformat(),
        "candidate_rows": rows,
    }
    payload = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict procurement same-product auditor. Judge whether each "
                    "candidate GeM product name is the same underlying product as the input "
                    "product or a renamed equivalent of that product. Do not flag broad "
                    "category matches, unrelated products, or merely possible substitutes. "
                    "Different product type, use case, consumable family, medical item, "
                    "software/service family, capacity, duration, pack size, or license count "
                    "is a material difference unless the names still clearly describe the "
                    "same item. If the candidate is only a close substitute, set "
                    "same_underlying_product=false and renamed_equivalent=false. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return this exact JSON shape: "
                    "{\"matches\":[{\"row_id\":0,\"same_underlying_product\":true,"
                    "\"renamed_equivalent\":true,\"close_substitute\":false,"
                    "\"canonical_concept\":\"desktop_computer\","
                    "\"material_difference\":false,"
                    "\"material_difference_reason\":\"\","
                    "\"fraud_relevance\":\"high\",\"confidence\":0.95,"
                    "\"reason\":\"short reason\"}]}. "
                    "Only include rows with fraud_relevance high or medium where "
                    "same_underlying_product or renamed_equivalent is true. Exclude rows if "
                    "your reason would say unrelated, different product type, different use "
                    "case, or no match. "
                    f"Data: {json.dumps(prompt_payload, ensure_ascii=True)}"
                ),
            },
        ],
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        args.api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        raw = response.read().decode("utf-8", "replace")
    response_data = json.loads(raw)
    content = response_data["choices"][0]["message"]["content"]
    data = extract_json_object(content)
    matches = data.get("matches", [])
    if not isinstance(matches, list):
        raise ValueError("LLM response field 'matches' must be a list")
    return [match for match in matches if isinstance(match, dict)]


def normalize_fraud_relevance(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"high", "medium", "low"} else "low"


def normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def validated_llm_matches(
    input_product: str,
    candidates: pd.DataFrame,
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid_rows: list[dict[str, Any]] = []
    seen_row_ids: set[int] = set()
    candidate_names = candidates["candidate_product_name"].to_dict()

    for match in matches:
        try:
            row_id = int(match.get("row_id"))
        except (TypeError, ValueError):
            continue
        if row_id in seen_row_ids or row_id not in candidate_names:
            continue

        fraud_relevance = normalize_fraud_relevance(match.get("fraud_relevance"))
        if fraud_relevance not in {"high", "medium"}:
            continue

        reason = str(match.get("reason") or "").strip()
        if CONTRADICTION_RE.search(reason):
            continue

        same_product = is_true(match.get("same_underlying_product"))
        renamed_equivalent = is_true(match.get("renamed_equivalent"))
        material_difference = is_true(match.get("material_difference"))
        if not (same_product or renamed_equivalent) or material_difference:
            continue

        gate_passed, gate_reason = same_product_gate(input_product, str(candidate_names[row_id]))
        if not gate_passed:
            continue

        cleaned = dict(match)
        cleaned["row_id"] = row_id
        cleaned["fraud_relevance"] = fraud_relevance
        cleaned["confidence"] = normalize_confidence(match.get("confidence"))
        cleaned["local_validation_reason"] = gate_reason
        valid_rows.append(cleaned)
        seen_row_ids.add(row_id)

    return valid_rows


def enrich_matches(
    input_product: str,
    candidates: pd.DataFrame,
    matches: list[dict[str, Any]],
) -> pd.DataFrame:
    matches = validated_llm_matches(input_product, candidates, matches)
    if not matches:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    llm = pd.DataFrame(matches)
    if "row_id" not in llm.columns:
        raise ValueError("LLM matches must include row_id")
    llm["row_id"] = pd.to_numeric(llm["row_id"], errors="coerce")
    llm = llm[llm["row_id"].notna()].copy()
    llm["row_id"] = llm["row_id"].astype("int64")

    base = candidates.reset_index(names="row_id")
    selected_columns = [
        "row_id",
        "transaction_id",
        "order_id",
        "supply_order_date",
        "days_from_input",
        "fk_central_unit",
        "candidate_product_name",
        "product_brand",
        "quantity_ordered",
        "unit_price",
        "total_value",
        "quantity_unit_type",
        "vendor_name",
        "bill_amount",
        "amount_passed",
    ]
    output = base[selected_columns].merge(llm, on="row_id", how="inner")
    relevance_rank = {"high": 0, "medium": 1}
    output["_relevance_rank"] = output["fraud_relevance"].map(relevance_rank).fillna(9)
    return output.sort_values(
        ["_relevance_rank", "confidence", "days_from_input"],
        ascending=[True, False, True],
    ).drop(columns=["_relevance_rank"])


def main() -> int:
    try:
        args = parse_args()
    except ValueError as exc:
        print(f"Invalid arguments: {exc}", file=sys.stderr)
        return 2

    if not args.product_name:
        print("--product-name cannot be empty", file=sys.stderr)
        return 2

    scanned_candidates = load_candidates(args)
    candidates = filter_candidates_for_llm(args.product_name, scanned_candidates)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        candidates.to_csv(args.output_csv, index=False)
        print(f"Wrote filtered candidates without LLM: {args.output_csv}")
        print(f"Scanned candidate rows: {len(scanned_candidates)}")
        print(f"LLM candidate rows: {len(candidates)}")
        return 0

    if not args.api_url:
        print("Missing --api-url or SEMANTIC_API_URL for LLM judging.", file=sys.stderr)
        return 2

    all_matches: list[dict[str, Any]] = []
    try:
        for batch in batch_rows(candidates, args.batch_size):
            all_matches.extend(call_llm(args, batch))
    except (
        KeyError,
        TimeoutError,
        urllib.error.URLError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"LLM audit failed: {exc}", file=sys.stderr)
        return 3

    output = enrich_matches(args.product_name, candidates, all_matches)
    output.to_csv(args.output_csv, index=False)
    print(f"Scanned candidate rows: {len(scanned_candidates)}")
    print(f"LLM candidate rows: {len(candidates)}")
    print(f"LLM suspicious matches: {len(output)}")
    print(f"Output: {args.output_csv}")
    if not output.empty:
        print(
            output[
                [
                    "transaction_id",
                    "supply_order_date",
                    "days_from_input",
                    "candidate_product_name",
                    "fraud_relevance",
                    "confidence",
                    "reason",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )
    else:
        print("No suspicious same-product matches after validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
