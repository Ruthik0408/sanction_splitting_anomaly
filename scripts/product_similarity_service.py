"""Historical product retrieval using semantic embedding similarity."""

from __future__ import annotations

import json
import logging
import math
from time import perf_counter
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from scripts.config import (
    artifacts_dir,
    database_settings,
    duplicate_match_threshold,
    duplicate_retrieval_threshold,
    duplicate_review_threshold,
    duplicate_window_days,
    embedding_model_name,
    max_candidate_products,
    rerank_device,
    rerank_enabled,
    rerank_match_threshold,
    rerank_model_name,
    rerank_review_threshold,
    rerank_top_k,
)
from scripts.embedding_runtime import resolved_embedding_device
from scripts.product_nlp import normalize_product_text
from scripts.reranker_runtime import load_reranker
from scripts.vllm_category_matcher import (
    same_product_match,
    unavailable_category_match,
)

MODEL_METADATA_FILE = "embedding_model.json"
logger = logging.getLogger("uvicorn.error")


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


@dataclass(frozen=True)
class SimilaritySettings:
    match_threshold: float
    review_threshold: float
    retrieval_threshold: float
    window_days: int
    max_candidates: int
    use_reranker: bool
    reranker_model: str
    reranker_device: str
    reranker_top_k: int
    reranker_match_threshold: float
    reranker_review_threshold: float


class ProductSimilarityService:
    def __init__(
        self,
        db_host: str,
        db_name: str,
        db_user: str,
        db_pass: str,
        db_port: int = 5432,
        model_dir: str | Path | None = None,
        settings: SimilaritySettings | None = None,
    ):
        """Load the embedding model and connect to the database."""
        self.settings = settings or SimilaritySettings(
            match_threshold=duplicate_match_threshold(),
            review_threshold=duplicate_review_threshold(),
            retrieval_threshold=duplicate_retrieval_threshold(),
            window_days=duplicate_window_days(),
            max_candidates=max_candidate_products(),
            use_reranker=rerank_enabled(),
            reranker_model=rerank_model_name(),
            reranker_device=rerank_device(),
            reranker_top_k=rerank_top_k(),
            reranker_match_threshold=rerank_match_threshold(),
            reranker_review_threshold=rerank_review_threshold(),
        )
        if self.settings.review_threshold > self.settings.match_threshold:
            raise ValueError("DUPLICATE_REVIEW_THRESHOLD must be <= DUPLICATE_MATCH_THRESHOLD")
        if self.settings.reranker_review_threshold > self.settings.reranker_match_threshold:
            raise ValueError("RERANK_REVIEW_THRESHOLD must be <= RERANK_MATCH_THRESHOLD")

        artifact_path = Path(model_dir) if model_dir else artifacts_dir()
        configured_model = embedding_model_name()
        model_source = self._model_source(artifact_path, configured_model)
        device = resolved_embedding_device()

        print(f"Loading embedding model: {model_source}")
        print(f"Embedding device: {device}")
        self.model_source = model_source
        try:
            self.embedder = SentenceTransformer(model_source, device=device)
        except Exception as exc:
            if not device.startswith("cuda"):
                raise
            print(f"CUDA embedding initialization failed; falling back to CPU: {exc}")
            self.embedder = SentenceTransformer(model_source, device="cpu")
            device = "cpu"
        self.embedding_device = device
        self.reranker = None
        self.reranker_error = ""
        if self.settings.use_reranker:
            print(
                "Loading reranker model: "
                f"{self.settings.reranker_model} on {self.settings.reranker_device}"
            )
            try:
                self.reranker = load_reranker(
                    self.settings.reranker_model,
                    self.settings.reranker_device,
                    local_files_only=True,
                )
            except Exception as exc:
                self.reranker_error = str(exc)
                print(f"Reranker disabled because loading failed: {exc}")
        self.conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_pass,
        )
        print("Embedding model loaded and DB connected")

    @staticmethod
    def _model_source(artifact_path: Path, configured_model: str) -> str:
        local_model = artifact_path / "embedder"
        metadata_path = artifact_path / MODEL_METADATA_FILE
        if not local_model.exists():
            return configured_model

        if not metadata_path.exists():
            print(
                "Local embedding model exists but metadata is missing; "
                f"using configured model instead: {configured_model}"
            )
            return configured_model

        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            print(
                "Local embedding metadata is invalid; "
                f"using configured model instead: {configured_model}"
            )
            return configured_model

        if metadata.get("model_name") != configured_model:
            print(
                "Local embedding model does not match EMBEDDING_MODEL "
                f"({metadata.get('model_name')} != {configured_model}); "
                "using configured model instead."
            )
            return configured_model
        return str(local_model)

    def _embed(self, product_name: str) -> np.ndarray:
        semantic_text = normalize_product_text(product_name)
        try:
            embedding = self.embedder.encode([semantic_text], show_progress_bar=False)
        except Exception as exc:
            if not self.embedding_device.startswith("cuda"):
                raise
            print(f"CUDA embedding failed; reloading the model on CPU: {exc}")
            self.embedder = SentenceTransformer(self.model_source, device="cpu")
            self.embedding_device = "cpu"
            embedding = self.embedder.encode([semantic_text], show_progress_bar=False)
        return normalize(np.asarray(embedding, dtype=np.float32))[0]

    @staticmethod
    def _to_pgvector(embedding: np.ndarray) -> str:
        return "[" + ",".join(f"{float(value):.8f}" for value in embedding) + "]"

    @staticmethod
    def _bounded_sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def _similar_candidates(
        self,
        product_name: str,
        fk_central_unit: int,
        order_id: str,
        supply_order_date: str,
    ) -> tuple[list[tuple[Any, ...]], int, dict[str, float]]:
        embed_started_at = perf_counter()
        query_vector = self._to_pgvector(self._embed(product_name))
        embed_ms = _elapsed_ms(embed_started_at)

        purchase_date = pd.to_datetime(supply_order_date)
        window_start = purchase_date - timedelta(days=self.settings.window_days)
        window_end = purchase_date + timedelta(days=self.settings.window_days)

        max_distance = 1.0 - self.settings.retrieval_threshold
        retrieval_pool = max(self.settings.max_candidates, 100)

        database_started_at = perf_counter()
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                WITH vector_candidates AS (
                    SELECT
                        b.id AS bill_id,
                        b.transaction_id,
                        b.order_id,
                        b.supply_order_date,
                        trim(p.product_name) AS product_name,
                        pe.semantic_text,
                        1 - (pe.embedding <=> %s::vector) AS similarity_score
                    FROM gem_bill b
                    JOIN gem_product p
                        ON p.fk_gem_bill = b.id
                    JOIN product_embedding pe
                        ON pe.product_name = trim(p.product_name)
                    WHERE b.fk_central_unit = %s
                      AND COALESCE(b.order_id, '') <> %s
                      AND b.supply_order_date >= %s
                      AND b.supply_order_date <= %s
                      AND b.record_status = 'V'
                      AND b.approved IS TRUE
                      AND p.product_name IS NOT NULL
                      AND trim(p.product_name) <> ''
                      AND (pe.embedding <=> %s::vector) <= %s
                    ORDER BY pe.embedding <=> %s::vector ASC
                    LIMIT %s
                ),

                lexical_candidates AS (
                    SELECT
                        b.id AS bill_id,
                        b.transaction_id,
                        b.order_id,
                        b.supply_order_date,
                        trim(p.product_name) AS product_name,
                        pe.semantic_text,
                        1 - (pe.embedding <=> %s::vector) AS similarity_score,
                        ts_rank_cd(
                            to_tsvector('simple', trim(p.product_name)),
                            websearch_to_tsquery('simple', %s)
                        ) AS lexical_score
                    FROM gem_bill b
                    JOIN gem_product p
                        ON p.fk_gem_bill = b.id
                    JOIN product_embedding pe
                        ON pe.product_name = trim(p.product_name)
                    WHERE b.fk_central_unit = %s
                      AND COALESCE(b.order_id, '') <> %s
                      AND b.supply_order_date >= %s
                      AND b.supply_order_date <= %s
                      AND b.record_status = 'V'
                      AND b.approved IS TRUE
                      AND p.product_name IS NOT NULL
                      AND trim(p.product_name) <> ''
                      AND to_tsvector('simple', trim(p.product_name)) @@ websearch_to_tsquery('simple', %s)
                    ORDER BY lexical_score DESC
                    LIMIT %s
                ),

                combined AS (
                    SELECT bill_id, transaction_id, order_id, supply_order_date, product_name, semantic_text, similarity_score
                    FROM vector_candidates
                    UNION
                    SELECT bill_id, transaction_id, order_id, supply_order_date, product_name, semantic_text, similarity_score
                    FROM lexical_candidates
                ),

                deduplicated AS (
                    SELECT DISTINCT ON (bill_id, order_id, product_name)
                        bill_id, transaction_id, order_id, supply_order_date, product_name, semantic_text, similarity_score
                    FROM combined
                    ORDER BY bill_id, order_id, product_name, similarity_score DESC
                )

                SELECT bill_id, transaction_id, order_id, supply_order_date, product_name, semantic_text, similarity_score
                FROM deduplicated
                ORDER BY similarity_score DESC
                LIMIT %s
                """,
                (
                    # vector_candidates (9 parameters)
                    query_vector,
                    fk_central_unit,
                    order_id,
                    window_start.date(),
                    window_end.date(),
                    query_vector,
                    max_distance,
                    query_vector,
                    retrieval_pool,
                    # lexical_candidates (7 parameters)
                    query_vector,
                    product_name,
                    fk_central_unit,
                    order_id,
                    window_start.date(),
                    window_end.date(),
                    product_name,
                    retrieval_pool,
                    # final pool (1 parameter)
                    self.settings.max_candidates,
                ),
            )

            rows = cursor.fetchall()

        return rows, len(rows), {
            "embedding_ms": embed_ms,
            "database_search_ms": _elapsed_ms(database_started_at),
        }

    def readiness_check(self) -> None:
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT product_name, semantic_text, embedding FROM product_embedding LIMIT 1")
            cursor.fetchone()

    def search_existing_purchases(
        self,
        query: str = "",
        fk_central_unit: int | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        cleaned_query = query.strip()
        bounded_limit = max(1, min(limit, 100))

        filters = [
            "b.record_status = 'V'",
            "b.approved IS TRUE",
            "p.product_name IS NOT NULL",
            "trim(p.product_name) <> ''",
        ]
        params: list[Any] = []

        if cleaned_query:
            filters.append("p.product_name ILIKE %s")
            params.append(f"%{cleaned_query}%")
        if fk_central_unit is not None:
            filters.append("b.fk_central_unit = %s")
            params.append(fk_central_unit)

        params.append(bounded_limit)
        where_clause = " AND ".join(filters)

        with self.conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    b.id AS bill_id,
                    b.transaction_id,
                    b.order_id,
                    b.supply_order_date,
                    b.fk_central_unit,
                    p.product_name
                FROM gem_bill b
                JOIN gem_product p ON p.fk_gem_bill = b.id
                WHERE {where_clause}
                ORDER BY b.supply_order_date DESC NULLS LAST, b.id DESC
                LIMIT %s
                """,
                params,
            )
            return [
                {
                    "bill_id": row[0],
                    "transaction_id": row[1],
                    "order_id": row[2],
                    "supply_order_date": str(row[3]),
                    "fk_central_unit": row[4],
                    "product_name": row[5],
                }
                for row in cursor.fetchall()
            ]

    def _rerank_matches(self, product_name: str, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.reranker is None or not matches:
            return matches

        rerank_limit = max(1, min(self.settings.reranker_top_k, len(matches)))
        rerank_candidates = matches[:rerank_limit]
        pairs = [[product_name, candidate["product_name"]] for candidate in rerank_candidates]
        raw_scores = self.reranker.predict(pairs)

        for candidate, raw_score in zip(rerank_candidates, raw_scores):
            rerank_score = self._bounded_sigmoid(float(raw_score))
            candidate["embedding_similarity_score"] = candidate["similarity_score"]
            candidate["rerank_score"] = round(rerank_score, 4)
            candidate["similarity_score"] = round(rerank_score, 4)

        reranked = sorted(rerank_candidates, key=lambda item: item["rerank_score"], reverse=True)
        remaining = matches[rerank_limit:]
        for candidate in remaining:
            candidate["embedding_similarity_score"] = candidate["similarity_score"]
            candidate["rerank_score"] = None
        return reranked + remaining

    def check_purchase(
        self,
        product_name: str,
        fk_central_unit: int,
        order_id: str,
        supply_order_date: str,
        llm_provider: str = "auto",
    ) -> dict[str, Any]:
        total_started_at = perf_counter()
        rows, candidate_count, timings = self._similar_candidates(
            product_name,
            fk_central_unit,
            order_id,
            supply_order_date,
        )
        matches = [
            {
                "bill_id": row[0],
                "transaction_id": row[1],
                "order_id": row[2],
                "supply_order_date": str(row[3]),
                "product_name": row[4],
                "semantic_text": row[5],
                "similarity_score": round(float(row[6]), 4),
                "semantic_rank": index,
            }
            for index, row in enumerate(rows, start=1)
        ]
        matches.sort(key=lambda item: item["similarity_score"], reverse=True)
        rerank_started_at = perf_counter()
        matches = self._rerank_matches(product_name, matches)
        timings["reranking_ms"] = _elapsed_ms(rerank_started_at)
        llm_match = unavailable_category_match(
            "No candidate products were found for LLM comparison.", llm_provider
        )
        llm_started_at = perf_counter()
        if matches:
            llm_match = same_product_match(
                provider=llm_provider,
                input_product_name=product_name,
                candidates=matches,
            )
            if llm_match.get("available"):
                decisions = {
                        item["rank"]: item
                        for item in llm_match.get("matches", [])
                        if isinstance(item, dict)
                }
                confirmed_matches = []
                for rank, candidate in enumerate(matches, start=1):
                    decision = decisions.get(rank)
                    candidate["llm_same_product"] = bool(
                        decision and decision.get("same_product") is True
                    )
                    candidate["llm_confidence"] = (
                        decision.get("confidence", 0.0) if decision else 0.0
                    )
                    candidate["llm_reason"] = decision.get("reason", "") if decision else ""
                    if candidate["llm_same_product"]:
                        confirmed_matches.append(candidate)
                matches = confirmed_matches
        timings["llm_verification_ms"] = _elapsed_ms(llm_started_at)

        best_score = matches[0]["similarity_score"] if matches else 0.0
        reranker_active = self.reranker is not None
        duplicate_threshold = (
            self.settings.reranker_match_threshold
            if reranker_active
            else self.settings.match_threshold
        )
        review_threshold = (
            self.settings.reranker_review_threshold
            if reranker_active
            else self.settings.review_threshold
        )
        score_label = "rerank" if reranker_active else "similarity"

        if best_score >= duplicate_threshold:
            decision = "duplicate"
            flagged = True
            reason = (
                f"Matched historical product with {score_label} score {best_score}. "
                f"Threshold for duplicate is {duplicate_threshold}."
            )
        elif best_score >= review_threshold:
            decision = "manual_review"
            flagged = True
            reason = (
                f"Similar historical product found with {score_label} score {best_score}. "
                f"Manual review threshold is {review_threshold}."
            )
        else:
            decision = "clear"
            flagged = False
            reason = "No semantically similar approved purchase found in the configured window."

        timings["total_ms"] = _elapsed_ms(total_started_at)
        stage_timings = {
            **timings,
            "slowest_stage": max(
                (
                    "embedding_ms",
                    "database_search_ms",
                    "reranking_ms",
                    "llm_verification_ms",
                ),
                key=timings.__getitem__,
            ),
        }
        logger.info(
            "purchase_check_timing product_length=%d candidates=%d timings=%s",
            len(product_name),
            candidate_count,
            stage_timings,
        )

        return {
            "flagged": flagged,
            "decision": decision,
            "best_similarity_score": best_score,
            "llm_same_product": llm_match,
            "thresholds": {
                "duplicate": duplicate_threshold,
                "manual_review": review_threshold,
                "retrieval": self.settings.retrieval_threshold,
            },
            "rerank": {
                "enabled": reranker_active,
                "requested": self.settings.use_reranker,
                "model": self.settings.reranker_model if reranker_active else "",
                "device": self.settings.reranker_device if reranker_active else "",
                "top_k": self.settings.reranker_top_k if reranker_active else 0,
                "error": self.reranker_error,
            },
            "reason": reason,
            "conflicting_bills": matches,
            "candidate_count": candidate_count,
            "timings": stage_timings,
        }

    def close(self) -> None:
        self.conn.close()


if __name__ == "__main__":
    db = database_settings()
    detector = ProductSimilarityService(
        db_host=db.host,
        db_port=db.port,
        db_name=db.name,
        db_user=db.user,
        db_pass=db.password,
    )

    try:
        result = detector.check_purchase(
            product_name="Unbranded Computer Paper, GSM 70",
            fk_central_unit=1263,
            order_id="ORD_2024_00999",
            supply_order_date="2024-03-15",
        )
        print(result)
    finally:
        detector.close()
