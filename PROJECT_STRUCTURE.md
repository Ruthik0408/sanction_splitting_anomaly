# Project Structure

This file is a quick map for contributors who need to edit or debug the
duplicate purchase detection project.

## Main entry points

- `run_app.sh` - Builds the React frontend and starts the FastAPI backend on
  port `5000`. Use this for normal local runs.
- `README.md` - Main usage guide with setup, run commands, and common scripts.
- `.env` - Local runtime configuration for database, embedding, reranker, and
  semantic matching settings. Do not commit secrets.
- `.env.example` - Template for required environment variables.
- `requirements.txt` - Python dependencies for the backend and ML scripts.

## Backend and matching logic

- `scripts/api_server.py` - FastAPI app. Exposes health checks, purchase search,
  duplicate checking APIs, and serves the frontend UI.
- `scripts/duplicate_detector.py` - Core duplicate detection flow. Loads the
  embedding model, queries the database, applies similarity thresholds, optional
  reranking, and optional semantic same-product validation.
- `scripts/config.py` - Shared `.env` loading and typed runtime settings.
- `scripts/product_nlp.py` - Product name cleanup and normalization before
  embedding/search.
- `scripts/embedding_runtime.py` - Chooses the embedding runtime device, such as
  CPU or CUDA.
- `scripts/vllm_category_matcher.py` - Optional OpenAI-compatible vLLM semantic
  judge for confirming whether two product names refer to the same product.
- `scripts/build_product_embeddings.py` - Rebuilds product embeddings in the
  database. Run after changing embedding model or normalization logic.
- `scripts/find_existing_similar_purchases.py` - Offline CSV/database audit for
  finding similar purchases by the same central unit.
- `scripts/evaluate_product_matching.py` - Evaluation script for product-pair
  matching thresholds and retrieval quality.
- `scripts/llm_same_product_audit.py` - LLM-assisted audit flow for same-product
  candidate review.
- `scripts/step1_find_similar_product_transactions.py` - Earlier/batch script
  for finding similar product transactions.
- `scripts/integration_example.py` - Example code showing how to call the
  duplicate checker from another process.

## Frontend UI

- `frontend/package.json` - React/Vite dependencies and frontend commands.
- `frontend/vite.config.js` - Vite build/dev-server configuration.
- `frontend/index.html` - Browser HTML entry point.
- `frontend/src/main.jsx` - Main React application. Contains the UI state,
  forms, existing-purchase search, and duplicate-check result rendering.
- `frontend/src/api.js` - Small API client for backend calls.
- `frontend/src/styles.css` - UI styling.
- `frontend/dist/` - Generated frontend build output. Do not edit directly.

## Data, artifacts, and evaluation

- `exports/` - CSV inputs and generated audit outputs, including `gem_bill.csv`,
  `gem_product.csv`, and duplicate/similar-purchase result files.
- `evaluation/product_pair_eval.csv` - Labeled product-pair evaluation data.
- `evaluation/results/` - Evaluation outputs with reranking or the configured
  full matching stack.
- `evaluation/results_embedding_only/` - Evaluation outputs for embedding-only
  matching.
- `categorization_artifacts/` - Local embedding model metadata and saved model
  artifacts used by the duplicate detector.

## Generated or local-only folders

- `venv/` - Local Python virtual environment.
- `frontend/node_modules/` - Installed frontend packages.
- `scripts/__pycache__/` - Python bytecode cache.

These folders are environment/build artifacts, not source code. Recreate them
from `requirements.txt` or `frontend/package.json` instead of editing them.
