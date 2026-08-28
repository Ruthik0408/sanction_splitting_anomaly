"""Build product embeddings used by duplicate detection.

Run this once after loading or refreshing product data.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from scripts.config import artifacts_dir, database_settings, embedding_model_name
from scripts.embedding_runtime import resolved_embedding_device
from scripts.product_nlp import normalize_product_text


BATCH_SIZE = 512
VECTOR_INDEX_LISTS = 100
MODEL_METADATA_FILE = "embedding_model.json"


def to_pgvector(embedding: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in embedding) + "]"


def fetch_products(cursor) -> list[str]:
    cursor.execute(
        """
        SELECT DISTINCT trim(product_name) AS product_name
        FROM gem_product
        WHERE product_name IS NOT NULL
          AND trim(product_name) <> ''
        ORDER BY trim(product_name)
        """
    )
    return [row[0] for row in cursor.fetchall()]


def recreate_embedding_table(cursor, embedding_dimension: int) -> None:
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cursor.execute("DROP TABLE IF EXISTS product_embedding")
    cursor.execute(
        sql.SQL(
            """
            CREATE TABLE product_embedding (
                product_name TEXT PRIMARY KEY,
                semantic_text TEXT NOT NULL,
                embedding vector({dimension}) NOT NULL
            )
            """
        ).format(dimension=sql.Literal(embedding_dimension))
    )
    cursor.execute("CREATE INDEX idx_product_embedding_name ON product_embedding(product_name)")
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_gem_product_bill_product
            ON gem_product(fk_gem_bill, product_name)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_gem_product_bill_trimmed_product
            ON gem_product(fk_gem_bill, trim(product_name))
        """
    )


def create_runtime_indexes(cursor) -> None:
    cursor.execute(
        """
        CREATE INDEX idx_product_embedding_vector_cosine
            ON product_embedding
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = %s)
        """,
        (VECTOR_INDEX_LISTS,),
    )
    cursor.execute("ANALYZE product_embedding")
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_gem_bill_duplicate_lookup
            ON gem_bill(fk_central_unit, supply_order_date, order_id)
            WHERE record_status = 'V' AND approved IS TRUE
        """
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pgvector product embeddings.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Products to embed per batch. Default: {BATCH_SIZE}.",
    )
    return parser.parse_args()


def write_model_metadata(
    artifact_path,
    model_name: str,
    device: str,
    embedding_dimension: int,
    product_count: int,
) -> None:
    metadata = {
        "model_name": model_name,
        "device": device,
        "embedding_dimension": embedding_dimension,
        "product_count": product_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage": "product_embedding",
        "search": "pgvector_cosine_product_name_nlp",
    }
    (artifact_path / MODEL_METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    db = database_settings()
    artifact_path = artifacts_dir()
    artifact_path.mkdir(parents=True, exist_ok=True)
    model_name = embedding_model_name()
    device = resolved_embedding_device()

    conn = psycopg2.connect(
        host=db.host,
        port=db.port,
        database=db.name,
        user=db.user,
        password=db.password,
    )
    cursor = conn.cursor()

    try:
        print("=" * 60)
        print("STEP 1: Fetching distinct products")
        print("=" * 60)
        products = fetch_products(cursor)
        print(f"Loaded {len(products)} products")

        print("\n" + "=" * 60)
        print(f"STEP 2: Loading embedding model: {model_name}")
        print(f"Embedding device: {device}")
        print("=" * 60)
        embedder = SentenceTransformer(model_name, device=device)
        sample_embedding = embedder.encode(["dimension probe"], show_progress_bar=False)
        embedding_dimension = int(np.asarray(sample_embedding).shape[1])
        print(f"Embedding dimension: {embedding_dimension}")

        print("\n" + "=" * 60)
        print("STEP 3: Recreating product_embedding table")
        print("=" * 60)
        recreate_embedding_table(cursor, embedding_dimension)
        conn.commit()

        total_batches = math.ceil(len(products) / args.batch_size)
        inserted = 0

        print("\n" + "=" * 60)
        print("STEP 4: NLP-normalizing, embedding, and inserting products")
        print("=" * 60)
        for batch_index, start in enumerate(range(0, len(products), args.batch_size), start=1):
            batch = products[start : start + args.batch_size]
            semantic_texts = [normalize_product_text(product_name) for product_name in batch]
            embeddings = embedder.encode(semantic_texts, show_progress_bar=False)
            embeddings = normalize(np.asarray(embeddings, dtype=np.float32))
            records = [
                (product_name, semantic_text, to_pgvector(embedding))
                for product_name, semantic_text, embedding in zip(batch, semantic_texts, embeddings)
            ]
            execute_values(
                cursor,
                """
                INSERT INTO product_embedding (product_name, semantic_text, embedding)
                VALUES %s
                """,
                records,
                page_size=1000,
            )
            conn.commit()
            inserted += len(records)

            if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
                print(f"  Batch {batch_index}/{total_batches}: inserted {inserted}/{len(products)}")

        print("\n" + "=" * 60)
        print("STEP 5: Creating vector index")
        print("=" * 60)
        create_runtime_indexes(cursor)
        conn.commit()

        embedder.save(str(artifact_path / "embedder"))
        write_model_metadata(
            artifact_path=artifact_path,
            model_name=model_name,
            device=device,
            embedding_dimension=embedding_dimension,
            product_count=inserted,
        )

        print("\n" + "=" * 60)
        print("EMBEDDING BUILD COMPLETE")
        print("=" * 60)
        print(f"Total products embedded: {inserted}")
        print("Database table created: product_embedding")
        print(f"Local model saved to: {artifact_path / 'embedder'}")
        print(f"Model metadata saved to: {artifact_path / MODEL_METADATA_FILE}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
