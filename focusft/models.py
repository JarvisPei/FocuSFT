"""Model and tokenizer loading utilities."""

from __future__ import annotations

import logging

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    padding_side: str = "right",
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side=padding_side,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name_or_path: str,
    torch_dtype: str | torch.dtype = "auto",
    attn_implementation: str | None = "sdpa",
    trust_remote_code: bool = True,
    gradient_checkpointing: bool = True,
    max_length: int | None = None,
) -> AutoModelForCausalLM:
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    if max_length is not None and hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = max(config.max_position_embeddings, max_length)

    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        torch_dtype = getattr(torch, torch_dtype)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
    except (ValueError, ImportError):
        logger.warning(
            "Failed with attn_implementation='%s'; falling back to the model default.",
            attn_implementation,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Loaded %s (%.1fB params)", model_name_or_path, num_params / 1e9)
    return model
