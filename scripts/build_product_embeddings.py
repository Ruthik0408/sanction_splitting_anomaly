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
import torch
from psycopg2 import sql
from psycopg2.extras import execute_values

from scripts.config import (
    artifacts_dir,
    database_settings,
    embedding_model_name,
)
from scripts.embedding_runtime import load_embedding_model, resolved_embedding_device
from scripts.product_nlp import (
    LEXICAL_NORMALIZATION_VERSION,
    SEMANTIC_NORMALIZATION_VERSION,
    prepare_embedding_text,
    prepare_lexical_text,
)


BATCH_SIZE = 512
VECTOR_INDEX_LISTS = 100
MODEL_METADATA_FILE = "embedding_model.json"
LIVE_TABLE = "product_embedding"
STAGING_TABLE = "product_embedding_staging"


def embedding_build_settings(model_name: str, embedding_dimension: int) -> dict[str, object]:
    return {
        "model_name": model_name,
        "embedding_dimension": embedding_dimension,
        "embedding_normalized": True,
        "distance_metric": "cosine",
        "semantic_normalization_version": SEMANTIC_NORMALIZATION_VERSION,
        "lexical_normalization_version": LEXICAL_NORMALIZATION_VERSION,
    }


def to_pgvector(embedding: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in embedding) + "]"


def encode_normalized_batch(
    embedder,
    semantic_texts: list[str],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, int]:
    """Encode with native normalization and recover from oversized CUDA batches."""
    effective_batch_size = min(batch_size, max(1, len(semantic_texts)))
    while True:
        try:
            embeddings = embedder.encode(
                semantic_texts,
                batch_size=effective_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(embeddings, dtype=np.float32), effective_batch_size
        except torch.OutOfMemoryError:
            if not device.startswith("cuda"):
                raise
            if effective_batch_size == 1:
                raise RuntimeError(
                    "BGE-M3 could not encode even one product on the configured GPU. "
                    "Set EMBEDDING_DEVICE=cpu and resume the staging build with --resume."
                ) from None
            reduced_batch_size = max(1, effective_batch_size // 2)
            print(
                "  CUDA memory exhausted at embedding batch size "
                f"{effective_batch_size}; retrying with {reduced_batch_size}."
            )
            effective_batch_size = reduced_batch_size
            torch.cuda.empty_cache()


def fetch_products(cursor, resume: bool = False) -> list[str]:
    if resume:
        cursor.execute(
            """
            SELECT DISTINCT trim(p.product_name) AS product_name
            FROM gem_product p
            WHERE p.product_name IS NOT NULL
              AND trim(p.product_name) <> ''
              AND NOT EXISTS (
                  SELECT 1 FROM product_embedding_staging pe
                  WHERE pe.product_name = trim(p.product_name)
              )
            ORDER BY trim(p.product_name)
            """
        )
    else:
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


def validate_resume_table(
    cursor,
    embedding_dimension: int,
    model_name: str,
) -> int:
    cursor.execute("SELECT to_regclass(%s)", (STAGING_TABLE,))
    if cursor.fetchone()[0] is None:
        raise RuntimeError("--resume requested, but product_embedding_staging does not exist")
    cursor.execute("SELECT vector_dims(embedding) FROM product_embedding_staging LIMIT 1")
    row = cursor.fetchone()
    if row and int(row[0]) != embedding_dimension:
        raise RuntimeError(
            "Cannot resume: existing vector dimension "
            f"{row[0]} != model dimension {embedding_dimension}"
        )
    cursor.execute("SELECT obj_description(%s::regclass)", (STAGING_TABLE,))
    comment = cursor.fetchone()[0]
    try:
        build_settings = json.loads(comment or "")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Cannot resume: staging build metadata is missing or invalid") from exc
    expected = embedding_build_settings(model_name, embedding_dimension)
    if build_settings != expected:
        raise RuntimeError(
            f"Cannot resume: staging settings {build_settings!r} != current settings {expected!r}"
        )
    cursor.execute("SELECT count(*) FROM product_embedding_staging")
    return int(cursor.fetchone()[0])


def recreate_embedding_table(
    cursor,
    embedding_dimension: int,
    model_name: str,
) -> None:
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    cursor.execute("DROP TABLE IF EXISTS product_embedding_staging")
    cursor.execute(
        sql.SQL(
            """
            CREATE TABLE product_embedding_staging (
                product_name TEXT PRIMARY KEY,
                semantic_text TEXT NOT NULL,
                lexical_text TEXT NOT NULL,
                embedding vector({dimension}) NOT NULL
            )
            """
        ).format(dimension=sql.Literal(embedding_dimension))
    )
    cursor.execute(
        "CREATE INDEX idx_product_embedding_staging_name "
        "ON product_embedding_staging(product_name)"
    )
    cursor.execute(
        "COMMENT ON TABLE product_embedding_staging IS %s",
        (
            json.dumps(embedding_build_settings(model_name, embedding_dimension)),
        ),
    )
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
        "CREATE INDEX IF NOT EXISTS idx_product_embedding_staging_lexical_trgm "
        "ON product_embedding_staging USING gin (lexical_text gin_trgm_ops)"
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_product_embedding_staging_vector_cosine
            ON product_embedding_staging
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = %s)
        """,
        (VECTOR_INDEX_LISTS,),
    )
    cursor.execute("ANALYZE product_embedding_staging")
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_gem_bill_duplicate_lookup
            ON gem_bill(fk_central_unit, supply_order_date, order_id)
            WHERE record_status = 'V' AND approved IS TRUE
        """
    )


def repair_prepared_text(cursor) -> int:
    """Repair text-only staging defects without recomputing valid embeddings."""
    cursor.execute(
        "SELECT product_name FROM product_embedding_staging "
        "WHERE semantic_text = '' OR lexical_text = ''"
    )
    product_names = [row[0] for row in cursor.fetchall()]
    if not product_names:
        return 0
    records = [
        (name, prepare_embedding_text(name), prepare_lexical_text(name))
        for name in product_names
    ]
    execute_values(
        cursor,
        """
        UPDATE product_embedding_staging AS target
        SET semantic_text = source.semantic_text,
            lexical_text = source.lexical_text
        FROM (VALUES %s) AS source(product_name, semantic_text, lexical_text)
        WHERE target.product_name = source.product_name
        """,
        records,
        page_size=1000,
    )
    return len(records)


def verify_staging_table(cursor, expected_count: int, embedding_dimension: int) -> None:
    cursor.execute(
        "SELECT count(*), min(vector_dims(embedding)), max(vector_dims(embedding)) "
        "FROM product_embedding_staging"
    )
    count, min_dimension, max_dimension = cursor.fetchone()
    if int(count) != expected_count:
        raise RuntimeError(f"Staging row count {count} != expected {expected_count}")
    if min_dimension != embedding_dimension or max_dimension != embedding_dimension:
        raise RuntimeError("Staging table contains incompatible vector dimensions")
    cursor.execute(
        "SELECT count(*) FROM product_embedding_staging "
        "WHERE semantic_text = '' OR lexical_text = ''"
    )
    if int(cursor.fetchone()[0]):
        raise RuntimeError("Staging table still contains empty prepared text after repair")
    cursor.execute(
        "SELECT 1 FROM pg_indexes WHERE tablename = %s "
        "AND indexdef ILIKE '%%vector_cosine_ops%%'",
        (STAGING_TABLE,),
    )
    if cursor.fetchone() is None:
        raise RuntimeError("Staging cosine vector index was not created")
    cursor.execute(
        "SELECT 1 FROM pg_indexes WHERE tablename = %s "
        "AND indexdef ILIKE '%%gin_trgm_ops%%'",
        (STAGING_TABLE,),
    )
    if cursor.fetchone() is None:
        raise RuntimeError("Staging lexical trigram index was not created")


def swap_staging_table(cursor) -> None:
    cursor.execute("DROP TABLE IF EXISTS product_embedding_old")
    cursor.connection.commit()
    cursor.execute("BEGIN")
    cursor.execute("SELECT to_regclass(%s)", (LIVE_TABLE,))
    if cursor.fetchone()[0] is not None:
        cursor.execute("ALTER TABLE product_embedding RENAME TO product_embedding_old")
    cursor.execute("ALTER TABLE product_embedding_staging RENAME TO product_embedding")
    cursor.connection.commit()
    cursor.execute("DROP TABLE IF EXISTS product_embedding_old")
    cursor.execute(
        "ALTER TABLE product_embedding RENAME CONSTRAINT "
        "product_embedding_staging_pkey TO product_embedding_pkey"
    )
    cursor.execute(
        "ALTER INDEX idx_product_embedding_staging_name "
        "RENAME TO idx_product_embedding_name"
    )
    cursor.execute(
        "ALTER INDEX idx_product_embedding_staging_vector_cosine "
        "RENAME TO idx_product_embedding_vector_cosine"
    )
    cursor.execute(
        "ALTER INDEX idx_product_embedding_staging_lexical_trgm "
        "RENAME TO idx_product_embedding_lexical_trgm"
    )
    cursor.connection.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pgvector product embeddings.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Products to embed per batch. Default: {BATCH_SIZE}.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a compatible interrupted build in product_embedding_staging.",
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
        **embedding_build_settings(model_name, embedding_dimension),
        "device": device,
        "product_count": product_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage": "product_embedding",
        "search": "pgvector_cosine_product_name_nlp",
    }
    (artifact_path / MODEL_METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
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
        print("STEP 1: Preparing database connection")
        print("=" * 60)
        print("\n" + "=" * 60)
        print(f"STEP 2: Loading embedding model: {model_name}")
        print(f"Embedding device: {device}")
        print("=" * 60)
        embedder, device = load_embedding_model(model_name)
        sample_embedding = embedder.encode(
            ["dimension probe"],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embedding_dimension = int(np.asarray(sample_embedding).shape[1])
        print(f"Embedding dimension: {embedding_dimension}")

        print("\n" + "=" * 60)
        print("STEP 3: Preparing product_embedding_staging (live table stays available)")
        print("=" * 60)
        if args.resume:
            existing_count = validate_resume_table(
                cursor,
                embedding_dimension,
                model_name,
            )
            products = fetch_products(cursor, resume=True)
            print(f"Resuming with {existing_count} existing and {len(products)} remaining products")
        else:
            products = fetch_products(cursor)
            print(f"Loaded {len(products)} products")
            recreate_embedding_table(
                cursor,
                embedding_dimension,
                model_name,
            )
            conn.commit()
            existing_count = 0

        total_batches = math.ceil(len(products) / args.batch_size)
        inserted = existing_count
        embedding_batch_size = args.batch_size

        print("\n" + "=" * 60)
        print("STEP 4: NLP-normalizing, embedding, and inserting products")
        print("=" * 60)
        for batch_index, start in enumerate(range(0, len(products), args.batch_size), start=1):
            batch = products[start : start + args.batch_size]
            semantic_texts = [prepare_embedding_text(product_name) for product_name in batch]
            lexical_texts = [prepare_lexical_text(product_name) for product_name in batch]
            embeddings, reduced_batch_size = encode_normalized_batch(
                embedder,
                semantic_texts,
                embedding_batch_size,
                device,
            )
            if reduced_batch_size < embedding_batch_size:
                embedding_batch_size = reduced_batch_size
                print(f"  Using embedding batch size {embedding_batch_size} for remaining batches.")
            records = [
                (product_name, semantic_text, lexical_text, to_pgvector(embedding))
                for product_name, semantic_text, lexical_text, embedding
                in zip(batch, semantic_texts, lexical_texts, embeddings)
            ]
            execute_values(
                cursor,
                """
                INSERT INTO product_embedding_staging
                    (product_name, semantic_text, lexical_text, embedding)
                VALUES %s
                """,
                records,
                page_size=1000,
            )
            conn.commit()
            inserted += len(records)

            if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
                print(
                    f"  Batch {batch_index}/{total_batches}: total rows {inserted}/"
                    f"{existing_count + len(products)}"
                )

        print("\n" + "=" * 60)
        print("STEP 5: Creating vector index")
        print("=" * 60)
        create_runtime_indexes(cursor)
        conn.commit()

        print("\n" + "=" * 60)
        print("STEP 6: Verifying staging table and swapping atomically")
        print("=" * 60)
        repaired_count = repair_prepared_text(cursor)
        if repaired_count:
            print(f"Repaired prepared text for {repaired_count} staging rows")
            conn.commit()
        verify_staging_table(cursor, inserted, embedding_dimension)
        swap_staging_table(cursor)

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
        print("Database table atomically replaced: product_embedding")
        print(f"Local model saved to: {artifact_path / 'embedder'}")
        print(f"Model metadata saved to: {artifact_path / MODEL_METADATA_FILE}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
