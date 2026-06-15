"""Find similar products already supplied within a central unit date window.

This is an offline audit over exported gem_bill/gem_product CSV files. It is
intended to complement the API path, which checks one new purchase at a time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from scripts.config import artifacts_dir, duplicate_review_threshold, duplicate_window_days


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BILL_CSV = BASE_DIR / "exports" / "gem_bill.csv"
DEFAULT_PRODUCT_CSV = BASE_DIR / "exports" / "gem_product.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "exports" / "existing_similar_purchases.csv"


@dataclass(frozen=True)
class AuditConfig:
    bill_csv: Path
    product_csv: Path
    output_csv: Path
    threshold: float
    window_days: int
    top_k_neighbors: int
    max_results: int
    fk_central_unit: int | None


def parse_args() -> AuditConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Audit existing data for semantically similar products supplied to "
            "the same fk_central_unit within a date window."
        )
    )
    parser.add_argument("--bill-csv", type=Path, default=DEFAULT_BILL_CSV)
    parser.add_argument("--product-csv", type=Path, default=DEFAULT_PRODUCT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--threshold",
        type=float,
        default=duplicate_review_threshold(),
        help="Minimum cosine similarity to report. Defaults to DUPLICATE_REVIEW_THRESHOLD.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=duplicate_window_days(),
        help="Date window in days. Defaults to DUPLICATE_WINDOW_DAYS.",
    )
    parser.add_argument(
        "--top-k-neighbors",
        type=int,
        default=25,
        help="Nearest semantic neighbors to keep for each distinct product.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=500,
        help="Maximum suspicious pairs to write.",
    )
    parser.add_argument(
        "--fk-central-unit",
        type=int,
        default=None,
        help="Optional single central unit to audit.",
    )
    args = parser.parse_args()
    return AuditConfig(
        bill_csv=args.bill_csv,
        product_csv=args.product_csv,
        output_csv=args.output_csv,
        threshold=args.threshold,
        window_days=args.window_days,
        top_k_neighbors=args.top_k_neighbors,
        max_results=args.max_results,
        fk_central_unit=args.fk_central_unit,
    )


def normalized_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def load_existing_purchases(config: AuditConfig) -> pd.DataFrame:
    bills = pd.read_csv(
        config.bill_csv,
        usecols=[
            "id",
            "transaction_id",
            "order_id",
            "supply_order_date",
            "record_status",
            "approved",
            "fk_central_unit",
        ],
        low_memory=False,
    )
    products = pd.read_csv(
        config.product_csv,
        usecols=["fk_gem_bill", "product_name", "product_category_name"],
        low_memory=False,
    )

    approved = normalized_text(bills["approved"]).str.lower().isin({"t", "true", "1"})
    bills = bills[
        (bills["record_status"] == "V")
        & approved
        & bills["fk_central_unit"].notna()
        & bills["supply_order_date"].notna()
        & bills["transaction_id"].notna()
    ].copy()
    if config.fk_central_unit is not None:
        bills = bills[bills["fk_central_unit"].astype("Int64") == config.fk_central_unit]

    bills["supply_order_date"] = pd.to_datetime(bills["supply_order_date"], errors="coerce")
    bills = bills[bills["supply_order_date"].notna()].copy()
    bills["fk_central_unit"] = bills["fk_central_unit"].astype("int64")
    bills["bill_id"] = bills["id"].astype("int64")
    bills["bill_transaction_id"] = normalized_text(bills["transaction_id"])
    bills["order_id"] = normalized_text(bills["order_id"])

    products = products[products["fk_gem_bill"].notna()].copy()
    products["bill_id"] = products["fk_gem_bill"].astype("int64")
    products["product_name"] = normalized_text(products["product_name"])
    products = products[products["product_name"] != ""]

    category_count = normalized_text(products["product_category_name"]).replace("", pd.NA).notna().sum()
    if category_count:
        print(
            "product_category_name is sparse in this export; "
            f"using product_name embeddings. Non-empty category rows: {category_count}"
        )

    joined = products.merge(
        bills[
            [
                "bill_id",
                "bill_transaction_id",
                "order_id",
                "supply_order_date",
                "fk_central_unit",
            ]
        ],
        on="bill_id",
        how="inner",
    )
    joined = joined.drop_duplicates(
        [
            "bill_id",
            "bill_transaction_id",
            "fk_central_unit",
            "supply_order_date",
            "product_name",
        ]
    )
    joined = joined.sort_values(["fk_central_unit", "supply_order_date", "bill_id"])
    return joined.reset_index(drop=True)


def embed_products(product_names: list[str]) -> np.ndarray:
    model_path = artifacts_dir() / "embedder"
    model_source = str(model_path) if model_path.exists() else "all-MiniLM-L6-v2"
    print(f"Loading embedding model: {model_source}")
    embedder = SentenceTransformer(model_source)
    embeddings = embedder.encode(product_names, batch_size=512, show_progress_bar=True)
    return normalize(np.asarray(embeddings, dtype=np.float32))


def build_neighbor_map(
    product_names: list[str],
    embeddings: np.ndarray,
    threshold: float,
    top_k_neighbors: int,
) -> dict[str, dict[str, float]]:
    neighbor_count = min(len(product_names), top_k_neighbors + 1)
    nn = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings, return_distance=True)

    neighbors: dict[str, dict[str, float]] = {}
    for source_index, source_name in enumerate(product_names):
        source_neighbors: dict[str, float] = {}
        for distance, target_index in zip(distances[source_index], indices[source_index]):
            if source_index == target_index:
                continue
            similarity = 1.0 - float(distance)
            if similarity >= threshold:
                source_neighbors[product_names[target_index]] = round(similarity, 4)
        if source_neighbors:
            neighbors[source_name] = source_neighbors
    return neighbors


def suspicious_pairs_for_unit(
    group: pd.DataFrame,
    neighbors: dict[str, dict[str, float]],
    config: AuditConfig,
    remaining_slots: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    records = group.to_dict("records")
    window = pd.Timedelta(days=config.window_days)
    start = 0

    for end, current in enumerate(records):
        current_date = current["supply_order_date"]
        while current_date - records[start]["supply_order_date"] > window:
            start += 1

        current_neighbors = neighbors.get(current["product_name"])
        if not current_neighbors:
            continue

        for previous in records[start:end]:
            if previous["bill_transaction_id"] == current["bill_transaction_id"]:
                continue
            if previous["order_id"] and previous["order_id"] == current["order_id"]:
                continue

            similarity = current_neighbors.get(previous["product_name"])
            if similarity is None:
                continue

            results.append(
                {
                    "fk_central_unit": int(current["fk_central_unit"]),
                    "similarity_score": similarity,
                    "days_between": int(abs((current_date - previous["supply_order_date"]).days)),
                    "transaction_id_1": previous["bill_transaction_id"],
                    "transaction_id_2": current["bill_transaction_id"],
                    "order_id_1": previous["order_id"],
                    "order_id_2": current["order_id"],
                    "supply_order_date_1": previous["supply_order_date"].date().isoformat(),
                    "supply_order_date_2": current_date.date().isoformat(),
                    "bill_id_1": int(previous["bill_id"]),
                    "bill_id_2": int(current["bill_id"]),
                    "product_name_1": previous["product_name"],
                    "product_name_2": current["product_name"],
                }
            )
            if len(results) >= remaining_slots:
                return results

    return results


def main() -> None:
    config = parse_args()
    purchases = load_existing_purchases(config)
    if purchases.empty:
        raise RuntimeError("No eligible purchases found after applying bill conditions.")

    product_names = sorted(purchases["product_name"].unique())
    print(f"Eligible product rows: {len(purchases)}")
    print(f"Distinct product names: {len(product_names)}")
    print(
        "Conditions: record_status='V', approved=true, same fk_central_unit, "
        f"different transaction/order, within {config.window_days} days, "
        f"similarity >= {config.threshold}"
    )

    embeddings = embed_products(product_names)
    embedding_by_product = {
        product_name: embeddings[index] for index, product_name in enumerate(product_names)
    }

    results: list[dict[str, Any]] = []
    for fk_central_unit, group in purchases.groupby("fk_central_unit", sort=False):
        unit_product_names = sorted(group["product_name"].unique())
        if len(unit_product_names) < 2:
            continue

        unit_embeddings = np.asarray(
            [embedding_by_product[product_name] for product_name in unit_product_names],
            dtype=np.float32,
        )
        neighbors = build_neighbor_map(
            unit_product_names,
            unit_embeddings,
            threshold=config.threshold,
            top_k_neighbors=config.top_k_neighbors,
        )
        if not neighbors:
            continue

        remaining_slots = config.max_results - len(results)
        results.extend(suspicious_pairs_for_unit(group, neighbors, config, remaining_slots))
        if len(results) >= config.max_results:
            break

    results = sorted(
        results,
        key=lambda item: (-item["similarity_score"], item["days_between"], item["fk_central_unit"]),
    )
    output = pd.DataFrame(results)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.output_csv, index=False)

    print(f"Suspicious pairs written: {len(output)}")
    print(f"Output: {config.output_csv}")
    if not output.empty:
        print("\nTop fk_central_unit counts:")
        print(output["fk_central_unit"].value_counts().head(20).to_string())
        print("\nTop matches:")
        print(
            output[
                [
                    "fk_central_unit",
                    "similarity_score",
                    "days_between",
                    "transaction_id_1",
                    "transaction_id_2",
                    "product_name_1",
                    "product_name_2",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
