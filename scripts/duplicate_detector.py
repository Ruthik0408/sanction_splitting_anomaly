"""Duplicate-purchase detection using semantic embedding similarity."""

from __future__ import annotations

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
    duplicate_review_threshold,
    duplicate_window_days,
    embedding_model_name,
    max_candidate_products,
)


@dataclass(frozen=True)
class DuplicateDetectorSettings:
    match_threshold: float
    review_threshold: float
    window_days: int
    max_candidates: int


class DuplicatePurchaseDetector:
    def __init__(
        self,
        db_host: str,
        db_name: str,
        db_user: str,
        db_pass: str,
        db_port: int = 5432,
        model_dir: str | Path | None = None,
        settings: DuplicateDetectorSettings | None = None,
    ):
        """Load the embedding model and connect to the database."""
        self.settings = settings or DuplicateDetectorSettings(
            match_threshold=duplicate_match_threshold(),
            review_threshold=duplicate_review_threshold(),
            window_days=duplicate_window_days(),
            max_candidates=max_candidate_products(),
        )
        if self.settings.review_threshold > self.settings.match_threshold:
            raise ValueError("DUPLICATE_REVIEW_THRESHOLD must be <= DUPLICATE_MATCH_THRESHOLD")

        artifact_path = Path(model_dir) if model_dir else artifacts_dir()
        local_model = artifact_path / "embedder"
        model_source = str(local_model) if local_model.exists() else embedding_model_name()

        print(f"Loading embedding model: {model_source}")
        self.embedder = SentenceTransformer(model_source)
        self.conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_pass,
        )
        print("Embedding model loaded and DB connected")

    def _embed(self, product_name: str) -> np.ndarray:
        embedding = self.embedder.encode([product_name], show_progress_bar=False)
        return normalize(np.asarray(embedding, dtype=np.float32))[0]

    @staticmethod
    def _to_pgvector(embedding: np.ndarray) -> str:
        return "[" + ",".join(f"{float(value):.8f}" for value in embedding) + "]"

    def _similar_candidates(
        self,
        product_name: str,
        fk_central_unit: int,
        order_id: str,
        supply_order_date: str,
    ) -> tuple[list[tuple[Any, ...]], int]:
        query_vector = self._to_pgvector(self._embed(product_name))
        purchase_date = pd.to_datetime(supply_order_date)
        window_start = purchase_date - timedelta(days=self.settings.window_days)
        window_end = purchase_date + timedelta(days=self.settings.window_days)
        max_distance = 1.0 - self.settings.review_threshold

        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*)
                FROM gem_bill b
                JOIN gem_product p ON p.fk_gem_bill = b.id
                JOIN product_embedding pe ON pe.product_name = p.product_name
                WHERE b.fk_central_unit = %s
                  AND COALESCE(b.order_id, '') <> %s
                  AND b.supply_order_date >= %s
                  AND b.supply_order_date <= %s
                  AND b.record_status = 'V'
                  AND b.approved IS TRUE
                  AND p.product_name IS NOT NULL
                """,
                (
                    fk_central_unit,
                    order_id,
                    window_start.date(),
                    window_end.date(),
                ),
            )
            candidate_count = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT
                    b.id AS bill_id,
                    b.transaction_id,
                    b.order_id,
                    b.supply_order_date,
                    p.product_name,
                    1 - (pe.embedding <=> %s::vector) AS similarity_score
                FROM gem_bill b
                JOIN gem_product p ON p.fk_gem_bill = b.id
                JOIN product_embedding pe ON pe.product_name = p.product_name
                WHERE b.fk_central_unit = %s
                  AND COALESCE(b.order_id, '') <> %s
                  AND b.supply_order_date >= %s
                  AND b.supply_order_date <= %s
                  AND b.record_status = 'V'
                  AND b.approved IS TRUE
                  AND p.product_name IS NOT NULL
                  AND (pe.embedding <=> %s::vector) <= %s
                ORDER BY pe.embedding <=> %s::vector ASC
                LIMIT %s
                """,
                (
                    query_vector,
                    fk_central_unit,
                    order_id,
                    window_start.date(),
                    window_end.date(),
                    query_vector,
                    max_distance,
                    query_vector,
                    self.settings.max_candidates,
                ),
            )
            return cursor.fetchall(), candidate_count

    def readiness_check(self) -> None:
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM product_embedding LIMIT 1")
            cursor.fetchone()

    def check_purchase(
        self,
        product_name: str,
        fk_central_unit: int,
        order_id: str,
        supply_order_date: str,
    ) -> dict[str, Any]:
        rows, candidate_count = self._similar_candidates(
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
                "similarity_score": round(float(row[5]), 4),
            }
            for row in rows
        ]
        matches.sort(key=lambda item: item["similarity_score"], reverse=True)

        best_score = matches[0]["similarity_score"] if matches else 0.0

        if best_score >= self.settings.match_threshold:
            decision = "duplicate"
            flagged = True
            reason = (
                f"Matched historical product with similarity {best_score}. "
                f"Threshold for duplicate is {self.settings.match_threshold}."
            )
        elif best_score >= self.settings.review_threshold:
            decision = "manual_review"
            flagged = True
            reason = (
                f"Similar historical product found with similarity {best_score}. "
                f"Manual review threshold is {self.settings.review_threshold}."
            )
        else:
            decision = "clear"
            flagged = False
            reason = "No semantically similar approved purchase found in the configured window."

        return {
            "flagged": flagged,
            "decision": decision,
            "best_similarity_score": best_score,
            "thresholds": {
                "duplicate": self.settings.match_threshold,
                "manual_review": self.settings.review_threshold,
            },
            "reason": reason,
            "conflicting_bills": matches[:20],
            "candidate_count": candidate_count,
        }

    def close(self) -> None:
        self.conn.close()


if __name__ == "__main__":
    db = database_settings()
    detector = DuplicatePurchaseDetector(
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
