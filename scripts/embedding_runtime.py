"""Runtime helpers for embedding model loading."""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

from scripts.config import embedding_device


def resolved_embedding_device() -> str:
    requested_device = embedding_device()
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "EMBEDDING_DEVICE is set to cuda, but PyTorch cannot access a CUDA GPU; "
            "falling back to cpu."
        )
        return "cpu"
    return requested_device


def load_embedding_model(model_source: str) -> tuple[SentenceTransformer, str]:
    """Load an embedder and recover when CUDA is visible but cannot allocate."""
    device = resolved_embedding_device()
    try:
        return SentenceTransformer(model_source, device=device), device
    except Exception as exc:
        if not device.startswith("cuda"):
            raise
        print(f"CUDA embedding initialization failed; falling back to CPU: {exc}")
        return SentenceTransformer(model_source, device="cpu"), "cpu"
