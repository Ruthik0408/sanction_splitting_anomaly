# sanction_splitting_anomaly

For a quick contributor map of important files and folders, see
[`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).

## Duplicate detection stack

The duplicate checker uses:

- `BAAI/bge-m3` SentenceTransformer embeddings by default.
- NLP normalization derived from `product_name` only, with no category column required.
- `pgvector` cosine search for semantic retrieval over AI embeddings.
- Optional cross-encoder reranking over the top embedding candidates.
- An OpenAI-compatible vLLM judge compares the final candidates and confirms whether
  each name is the same underlying product, rather than only matching a broad category.
- Normalized semantic text and embeddings stored in the separate `product_embedding` table.
- Similarity decisions based on embedding cosine similarity inside `scripts/duplicate_detector.py`.

After changing `EMBEDDING_MODEL` or after pulling this upgrade, rebuild the embedding
table because model dimensions and search indexes must match runtime:
Rebuild is also required after changing product-name NLP normalization rules.

```bash
cd /home/ruthikreddy/Desktop/sanction_spliting
source venv/bin/activate
python3 -m scripts.build_product_embeddings
```

Useful `.env` settings:

```env
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cuda
DUPLICATE_MATCH_THRESHOLD=0.88
DUPLICATE_REVIEW_THRESHOLD=0.78
DUPLICATE_RETRIEVAL_THRESHOLD=0.50
DUPLICATE_MAX_CANDIDATES=100
RERANK_ENABLED=true
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_DEVICE=cpu
RERANK_TOP_K=20
RERANK_MATCH_THRESHOLD=0.85
RERANK_REVIEW_THRESHOLD=0.70
SEMANTIC_CATEGORY_MATCH_ENABLED=true
SEMANTIC_TIMEOUT_SECONDS=5
```

## Run duplicate checker UI

Run frontend and backend together:

```bash
cd /home/ruthikreddy/Desktop/sanction_spliting
./run_app.sh
```

Open:

```text
http://localhost:5000/ui
```

Manual frontend build:

```bash
cd /home/ruthikreddy/Desktop/sanction_spliting/frontend
npm install
npm run build
```

Start the Python API:

```bash
cd /home/ruthikreddy/Desktop/sanction_spliting
source venv/bin/activate
uvicorn scripts.api_server:app --host 0.0.0.0 --port 5000
```

Open:

```text
http://localhost:5000/ui
```

If port 5000 is already in use, stop the old server with `Ctrl+C` and run the command again.

For frontend development with hot reload:

```bash
cd /home/ruthikreddy/Desktop/sanction_spliting/frontend
npm run dev
```

Open:

```text
http://localhost:5173/ui
```

## Audit existing purchases for similar same-unit buys

Use this when you want to scan historical exports and find products bought by
the same `fk_central_unit` within the configured date window:

```bash
cd /home/ruthikreddy/Desktop/sanction_spliting
source venv/bin/activate
python3 -m scripts.find_existing_similar_purchases \
  --bill-csv exports/gem_bill.csv \
  --product-csv exports/gem_product.csv \
  --output-csv exports/existing_similar_purchases.csv \
  --threshold 0.78 \
  --window-days 60 \
  --top-k-neighbors 50 \
  --max-results 5000
```

For one central unit:

```bash
python3 -m scripts.find_existing_similar_purchases \
  --fk-central-unit 1263 \
  --output-csv exports/existing_similar_purchases_1263.csv
```

The output includes `match_type`, `similarity_score`, transaction/order IDs,
dates, original product names, and normalized semantic text used for matching.
