"""Runtime helpers for embedding model loading."""

from __future__ import annotations

import torch

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
