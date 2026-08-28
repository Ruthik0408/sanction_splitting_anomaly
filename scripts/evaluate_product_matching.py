"""Evaluate current product-name embedding and reranker behavior on labeled pairs.

This script is intentionally separate from production matching code. It reuses
the production config, device resolution, and product-name normalization, but it
does not change thresholds, models, database tables, or matching behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize

from scripts.config import (
    BASE_DIR,
    artifacts_dir,
    embedding_model_name,
    rerank_device,
    rerank_model_name,
)
from scripts.embedding_runtime import resolved_embedding_device
from scripts.product_nlp import normalize_product_text


DEFAULT_DATASET = BASE_DIR / "evaluation" / "product_pair_eval.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "evaluation" / "results"
DEFAULT_THRESHOLDS = "0.50,0.60,0.70,0.78,0.82,0.85,0.88,0.90,0.95"
RETRIEVAL_KS = (1, 5, 10, 20, 50)
MODEL_METADATA_FILE = "embedding_model.json"


@dataclass(frozen=True)
class EvalArgs:
    dataset: Path
    output_dir: Path
    thresholds: list[float]
    batch_size: int
    skip_reranker: bool
    require_reranker: bool


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    for threshold in thresholds:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"Threshold must be between 0 and 1: {threshold}")
    return thresholds


def parse_args() -> EvalArgs:
    parser = argparse.ArgumentParser(
        description="Benchmark the current product-name embedding model and reranker."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Evaluate embeddings only. Useful when reranker model is not available locally.",
    )
    parser.add_argument(
        "--require-reranker",
        action="store_true",
        help="Fail if the configured CrossEncoder cannot be loaded.",
    )
    args = parser.parse_args()
    return EvalArgs(
        dataset=args.dataset,
        output_dir=args.output_dir,
        thresholds=parse_thresholds(args.thresholds),
        batch_size=args.batch_size,
        skip_reranker=args.skip_reranker,
        require_reranker=args.require_reranker,
    )


def read_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"product_a", "product_b", "label", "case_type"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    df = df[["product_a", "product_b", "label", "case_type"]].copy()
    df["product_a"] = df["product_a"].fillna("").astype(str).str.strip()
    df["product_b"] = df["product_b"].fillna("").astype(str).str.strip()
    df["case_type"] = df["case_type"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    invalid_labels = sorted(set(df["label"]) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Labels must be 0 or 1. Found: {invalid_labels}")
    if (df["product_a"] == "").any() or (df["product_b"] == "").any():
        raise ValueError("product_a and product_b cannot be empty.")
    return df.reset_index(drop=True)


def model_source() -> str:
    artifact_path = artifacts_dir()
    local_model = artifact_path / "embedder"
    metadata_path = artifact_path / MODEL_METADATA_FILE
    configured_model = embedding_model_name()
    if local_model.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            metadata = {}
        if metadata.get("model_name") == configured_model:
            return str(local_model)
    return configured_model


def encode_texts(model: SentenceTransformer, texts: Iterable[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return normalize(np.asarray(embeddings, dtype=np.float32))


def bounded_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def score_pairs(args: EvalArgs, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["normalized_a"] = df["product_a"].map(normalize_product_text)
    df["normalized_b"] = df["product_b"].map(normalize_product_text)

    device = resolved_embedding_device()
    source = model_source()
    print(f"Loading embedding model: {source}")
    print(f"Embedding device: {device}")
    embedder = SentenceTransformer(source, device=device)

    unique_texts = sorted(set(df["normalized_a"]) | set(df["normalized_b"]))
    embeddings = encode_texts(embedder, unique_texts, args.batch_size)
    embedding_by_text = dict(zip(unique_texts, embeddings))

    df["embedding_cosine"] = [
        round(float(np.dot(embedding_by_text[row.normalized_a], embedding_by_text[row.normalized_b])), 6)
        for row in df.itertuples(index=False)
    ]

    if args.skip_reranker:
        df["cross_encoder_score"] = np.nan
        return df

    reranker_model = rerank_model_name()
    device = rerank_device()
    print(f"Loading reranker model: {reranker_model}")
    print(f"Reranker device: {device}")
    try:
        reranker = CrossEncoder(reranker_model, device=device)
    except Exception as exc:
        if args.require_reranker:
            raise
        print(f"Reranker unavailable; writing embedding-only results. Reason: {exc}")
        df["cross_encoder_score"] = np.nan
        return df

    pairs = [[row.product_a, row.product_b] for row in df.itertuples(index=False)]
    raw_scores = reranker.predict(pairs)
    df["cross_encoder_score"] = [round(bounded_sigmoid(float(score)), 6) for score in raw_scores]
    return df


def metrics_at_thresholds(
    df: pd.DataFrame,
    score_column: str,
    thresholds: list[float],
) -> pd.DataFrame:
    y_true = df["label"].astype(int).to_numpy()
    rows: list[dict[str, float | str]] = []
    for threshold in thresholds:
        y_pred = (df[score_column].astype(float).to_numpy() >= threshold).astype(int)
        rows.append(
            {
                "score": score_column,
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "predicted_positive": int(y_pred.sum()),
            }
        )
    return pd.DataFrame(rows)


def best_threshold(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score, group in metrics.groupby("score", sort=False):
        best = group.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def error_rows(df: pd.DataFrame, score_column: str, threshold: float, expected: int) -> pd.DataFrame:
    predicted = (df[score_column].astype(float) >= threshold).astype(int)
    mask = (df["label"] == expected) & (predicted != expected)
    columns = [
        "product_a",
        "product_b",
        "label",
        "case_type",
        "normalized_a",
        "normalized_b",
        score_column,
    ]
    return df.loc[mask, columns].sort_values(score_column, ascending=expected == 1)


def retrieval_metrics(df: pd.DataFrame) -> pd.DataFrame:
    products = sorted(set(df["product_a"]) | set(df["product_b"]))
    rows = []
    for product_a, group in df.groupby("product_a", sort=False):
        ranked = (
            df[df["product_a"] == product_a]
            .sort_values("embedding_cosine", ascending=False)
            .reset_index(drop=True)
        )
        candidate_rank = {
            row.product_b: index + 1 for index, row in enumerate(ranked.itertuples(index=False))
        }
        for row in group.itertuples(index=False):
            rank = candidate_rank.get(row.product_b)
            rows.append(
                {
                    "product_a": row.product_a,
                    "product_b": row.product_b,
                    "label": row.label,
                    "case_type": row.case_type,
                    "retrieval_rank_within_eval_candidates": rank,
                    **{f"hit_at_{k}": bool(rank is not None and rank <= k) for k in RETRIEVAL_KS},
                    "candidate_pool_size_for_product_a": len(ranked),
                    "global_unique_product_count": len(products),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(args: EvalArgs, scored: pd.DataFrame) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_scores_path = args.output_dir / "pair_scores.csv"
    scored.to_csv(pair_scores_path, index=False, quoting=csv.QUOTE_MINIMAL)

    metric_frames = [metrics_at_thresholds(scored, "embedding_cosine", args.thresholds)]
    if scored["cross_encoder_score"].notna().any():
        metric_frames.append(metrics_at_thresholds(scored, "cross_encoder_score", args.thresholds))
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics_path = args.output_dir / "threshold_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    best = best_threshold(metrics)
    best_path = args.output_dir / "best_thresholds.csv"
    best.to_csv(best_path, index=False)

    false_positive_frames = []
    false_negative_frames = []
    for row in best.itertuples(index=False):
        score_column = str(row.score)
        threshold = float(row.threshold)
        fps = error_rows(scored, score_column, threshold, expected=0)
        fns = error_rows(scored, score_column, threshold, expected=1)
        fps.insert(0, "score", score_column)
        fps.insert(1, "threshold", threshold)
        fns.insert(0, "score", score_column)
        fns.insert(1, "threshold", threshold)
        false_positive_frames.append(fps)
        false_negative_frames.append(fns)

    false_positives = pd.concat(false_positive_frames, ignore_index=True)
    false_negatives = pd.concat(false_negative_frames, ignore_index=True)
    false_positives.to_csv(args.output_dir / "false_positives.csv", index=False)
    false_negatives.to_csv(args.output_dir / "false_negatives.csv", index=False)

    retrieval = retrieval_metrics(scored)
    retrieval.to_csv(args.output_dir / "retrieval_metrics.csv", index=False)

    print(f"Pair scores: {pair_scores_path}")
    print(f"Threshold metrics: {metrics_path}")
    print(f"Best thresholds: {best_path}")
    print(f"False positives: {args.output_dir / 'false_positives.csv'}")
    print(f"False negatives: {args.output_dir / 'false_negatives.csv'}")
    print(f"Retrieval metrics: {args.output_dir / 'retrieval_metrics.csv'}")
    print()
    print("Best thresholds by F1:")
    print(best.to_string(index=False))


def main() -> None:
    args = parse_args()
    pairs = read_pairs(args.dataset)
    print(f"Loaded evaluation pairs: {len(pairs)}")
    scored = score_pairs(args, pairs)
    write_outputs(args, scored)


if __name__ == "__main__":
    main()
