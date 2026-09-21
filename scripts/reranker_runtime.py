"""Consistent reranker loading for the API and offline evaluation."""
from sentence_transformers import CrossEncoder
from torch import nn

JINA_RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"


def load_reranker(model_name: str, device: str, *, local_files_only: bool = False) -> CrossEncoder:
    is_jina = model_name == JINA_RERANKER_MODEL
    return CrossEncoder(
        model_name,
        device=device,
        trust_remote_code=is_jina,
        local_files_only=local_files_only,
        # Callers convert logits to scores once, after prediction.
        activation_fn=nn.Identity(),
        processor_kwargs={"fix_mistral_regex": True} if is_jina else {},
        config_kwargs={"use_flash_attn": False} if is_jina else {},
    )
