#!/usr/bin/env python3
"""Find transaction_ids for semantically similar GeM products.

This script does Step 1 only:
1. Pull broad keyword candidates from public.gem_product.
2. Optionally refine candidates through an OpenAI-compatible vLLM chat API.

The API base URL can be configured as `SEMANTIC_API_URL` or the existing
`api_key` value in .env. Both base URLs and full /chat/completions URLs work.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path



DEFAULT_MODEL = "Qwen3-30B-A3B-Instruct"
PLACEHOLDER_PATTERN = re.compile(r"\b(?:YOUR_|CHANGE_ME|TODO|REPLACE_ME)", re.IGNORECASE)


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def regex_from_terms(value: str) -> str:
    terms = [term.strip().lower() for term in re.split(r"[,|]", value) if term.strip()]
    if not terms:
        return ""
    escaped = [re.escape(term).replace(r"\ ", r"\s+") for term in terms]
    return r"\m(" + "|".join(escaped) + r")\M"


def run_psql_copy(env: dict[str, str], target_regex: str, exclude_regex: str, output: Path) -> None:
    required_keys = (
        "TULIP_SOURCE_DB_HOST",
        "TULIP_SOURCE_DB_PORT",
        "TULIP_SOURCE_DB_USER",
        "TULIP_SOURCE_DB_NAME",
        "TULIP_SOURCE_DB_PASSWORD",
    )
    missing_keys = [key for key in required_keys if not env.get(key)]
    if missing_keys:
        raise RuntimeError(f"Missing database settings in .env: {', '.join(missing_keys)}")

    schema = env.get("TULIP_SOURCE_DB_SCHEMA", "public")
    target_regex = target_regex.replace("'", "''")
    exclude_regex = exclude_regex.replace("'", "''")
    exclude_filter = f"and text_to_match !~ '{exclude_regex}'" if exclude_regex else ""
    sql = f"""
copy (
    with normalized as (
        select
            transaction_id,
            product_name,
            lower(product_name) as text_to_match
        from "{schema}".gem_product
        where transaction_id is not null
          and btrim(transaction_id) <> ''
    )
    select distinct transaction_id, product_name
    from normalized
    where text_to_match ~ '{target_regex}'
      {exclude_filter}
    order by transaction_id, product_name
) to stdout with csv header
"""
    cmd = [
        "psql",
        "-h",
        env["TULIP_SOURCE_DB_HOST"],
        "-p",
        env["TULIP_SOURCE_DB_PORT"],
        "-U",
        env["TULIP_SOURCE_DB_USER"],
        "-d",
        env["TULIP_SOURCE_DB_NAME"],
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        sql,
    ]
    merged_env = os.environ.copy()
    merged_env["PGPASSWORD"] = env["TULIP_SOURCE_DB_PASSWORD"]
    with tempfile.NamedTemporaryFile("w", delete=False, dir=output.parent, newline="") as handle:
        temp_output = Path(handle.name)
        try:
            subprocess.run(
                cmd,
                check=True,
                env=merged_env,
                stdout=handle,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            temp_output.unlink(missing_ok=True)
            message = (exc.stderr or "").strip() or f"psql exited with status {exc.returncode}"
            raise RuntimeError(message) from exc
    temp_output.replace(output)


def candidate_batches(path: Path, batch_size: int) -> list[list[dict[str, str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def chat_completions_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if not url:
        return ""
    if PLACEHOLDER_PATTERN.search(url):
        raise ValueError(f"API URL contains an unresolved placeholder: {value!r}")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed = urllib.parse.urlsplit(url)
    if not parsed.hostname:
        raise ValueError(f"API URL is missing a host: {value!r}")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"API URL has an invalid port: {value!r}") from exc
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def extract_json_object(text: str) -> dict[str, object]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"transaction_ids": data}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("No JSON object found in model response", text, 0)
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise json.JSONDecodeError("JSON response is not an object", text, 0)
    return data


def call_category_api(
    api_url: str,
    model: str,
    target: str,
    rows: list[dict[str, str]],
    timeout: int,
) -> set[str]:
    compact_rows = [
        {"transaction_id": row["transaction_id"], "product_name": row["product_name"]}
        for row in rows
    ]
    user_content = {
        "target_category": target,
        "items": compact_rows,
    }
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You classify GeM product names for fraud/sanction-splitting analysis. "
                    "Return JSON only with key transaction_ids. Include products that are "
                    "the target product category or clear synonyms/alternate names. Exclude "
                    "accessories, consumables, spare parts, services, software, printers, "
                    "batteries, UPS, paper, RAM, motherboards, and peripherals."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(user_content, ensure_ascii=True),
            },
        ],
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", "replace")
    response_data = json.loads(raw)
    content = response_data["choices"][0]["message"]["content"]
    data = extract_json_object(content)
    values = data.get("transaction_ids") or data.get("matches") or []
    return {str(value) for value in values}


def refine_with_api(
    input_csv: Path,
    output_csv: Path,
    api_url: str,
    model: str,
    target: str,
    batch_size: int,
    timeout: int,
) -> None:
    matched_ids: set[str] = set()
    for batch in candidate_batches(input_csv, batch_size):
        matched_ids.update(call_category_api(api_url, model, target, batch, timeout))

    with input_csv.open(newline="") as in_handle, output_csv.open("w", newline="") as out_handle:
        reader = csv.DictReader(in_handle)
        writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["transaction_id"] in matched_ids:
                writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="exports/step1")
    parser.add_argument("--api-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--target",
        required=True,
        help="Comma- or pipe-separated product terms to include, for example 'computer,desktop,pc'.",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma- or pipe-separated product terms to exclude from broad candidates.",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args()

    env = load_env(Path(args.env_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_csv = output_dir / "computer_like_transaction_ids_pre_api.csv"
    refined_csv = output_dir / "computer_like_transaction_ids_api_refined.csv"

    try:
        run_psql_copy(env, regex_from_terms(args.target), regex_from_terms(args.exclude), candidates_csv)
    except RuntimeError as exc:
        print(f"Database candidate export failed: {exc}", file=sys.stderr)
        return 1

    if args.skip_api:
        print(f"Wrote broad candidates: {candidates_csv}")
        return 0

    try:
        api_url = chat_completions_url(
            args.api_url or env.get("SEMANTIC_API_URL", "") or env.get("api_key", "")
        )
    except ValueError as exc:
        print(f"Wrote broad candidates: {candidates_csv}")
        print(f"Invalid API URL configuration: {exc}", file=sys.stderr)
        print("Set SEMANTIC_API_URL to a real OpenAI-compatible base URL, or rerun with --skip-api.")
        return 2
    model = args.model or env.get("SEMANTIC_MODEL", "") or DEFAULT_MODEL

    if not api_url:
        print(f"Wrote broad candidates: {candidates_csv}")
        print("No API URL found. Set SEMANTIC_API_URL or pass --api-url for API refinement.")
        return 2

    try:
        refine_with_api(candidates_csv, refined_csv, api_url, model, args.target, args.batch_size, args.timeout)
    except (KeyError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        print(f"Wrote broad candidates: {candidates_csv}")
        print(f"API refinement failed: {exc}", file=sys.stderr)
        return 3

    print(f"Wrote API-refined candidates: {refined_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
