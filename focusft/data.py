"""
Multi-turn agentic SFT dataset.

Adapted from Open-AgentRL's MultiTurnSFTDataset. Loads parquet files with
``messages`` and optional ``tools`` / ``enable_thinking`` columns, applies
the tokenizer's chat template, and builds per-role loss masks (only
assistant content is trained on).

Designed to work with HuggingFace ``Trainer`` and its default data collator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _to_list(x):
    """Recursively convert numpy arrays / pandas Series to plain lists."""
    if isinstance(x, dict):
        return {k: _to_list(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_list(e) for e in x]
    if isinstance(x, np.ndarray):
        return _to_list(x.tolist())
    return x


@dataclass
class DataConfig:
    train_files: str | list[str] = ""
    val_files: str | list[str] = ""
    max_length: int = 4096
    truncation: str = "right"  # "left", "right", or "error"
    messages_key: str = "messages"
    tools_key: str = "tools"
    enable_thinking_key: str = "enable_thinking"
    apply_chat_template_kwargs: dict = field(default_factory=dict)


class AgenticSFTDataset(Dataset):
    """Multi-turn agentic conversation dataset with per-role loss masking."""

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DataConfig | dict | None = None,
    ):
        if config is None:
            config = DataConfig()
        if isinstance(config, dict):
            config = DataConfig(**{k: v for k, v in config.items() if k in DataConfig.__dataclass_fields__})

        self.max_length = config.max_length
        self.truncation = config.truncation
        self.messages_key = config.messages_key
        self.tools_key = config.tools_key
        self.enable_thinking_key = config.enable_thinking_key
        self.extra_template_kwargs = config.apply_chat_template_kwargs
        assert self.truncation in ("left", "right", "error")

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        self.tokenizer = tokenizer

        dfs = [pd.read_parquet(f) for f in parquet_files]
        self.df = pd.concat(dfs, ignore_index=True)

        self.messages = self.df[self.messages_key].apply(_to_list).tolist()
        self.tools = (
            self.df[self.tools_key].apply(_to_list).tolist()
            if self.tools_key in self.df.columns
            else None
        )
        self.enable_thinking = (
            self.df[self.enable_thinking_key].tolist()
            if self.enable_thinking_key in self.df.columns
            else None
        )

    def __len__(self) -> int:
        return len(self.messages)

    # -----------------------------------------------------------------
    # Tokenization helpers (adapted from Open-AgentRL)
    # -----------------------------------------------------------------
    def _template_kwargs(self, tools, enable_thinking) -> dict:
        kw: dict[str, Any] = dict(self.extra_template_kwargs)
        if tools is not None:
            kw["tools"] = tools
        if enable_thinking is not None:
            kw["enable_thinking"] = enable_thinking
        return kw

    def _process_turn(
        self,
        messages: list[dict],
        start: int,
        end: int,
        is_assistant: bool,
        **template_kw,
    ) -> tuple[list[int], list[int]]:
        """Tokenize one turn and return (tokens, loss_mask)."""
        prev_text = (
            self.tokenizer.apply_chat_template(
                messages[:start],
                tokenize=False,
                add_generation_prompt=False,
                **template_kw,
            )
            if start > 0
            else ""
        )

        cur_text = self.tokenizer.apply_chat_template(
            messages[:end],
            tokenize=False,
            add_generation_prompt=False,
            **template_kw,
        )

        if is_assistant and start > 0:
            prev_w_gen = self.tokenizer.apply_chat_template(
                messages[:start],
                tokenize=False,
                add_generation_prompt=True,
                **template_kw,
            )
            gen_prompt_text = prev_w_gen[len(prev_text):]
            gen_prompt_toks = self.tokenizer.encode(gen_prompt_text, add_special_tokens=False)
            body_toks = self.tokenizer.encode(
                cur_text[len(prev_w_gen):], add_special_tokens=False
            )
            tokens = gen_prompt_toks + body_toks
            mask = [0] * len(gen_prompt_toks) + [1] * len(body_toks)
        else:
            delta_text = cur_text[len(prev_text):]
            tokens = self.tokenizer.encode(delta_text, add_special_tokens=False)
            if is_assistant:
                mask = [1] * len(tokens)
            else:
                mask = [0] * len(tokens)

        return tokens, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.messages[idx]
        tools = self.tools[idx] if self.tools is not None else None
        enable_thinking = self.enable_thinking[idx] if self.enable_thinking is not None else None
        tkw = self._template_kwargs(tools, enable_thinking)

        full_tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            add_generation_prompt=False, **tkw,
        )

        all_tokens: list[int] = []
        all_mask: list[int] = []
        i = 0
        while i < len(messages):
            role = messages[i]["role"]
            if role == "assistant":
                toks, m = self._process_turn(messages, i, i + 1, is_assistant=True, **tkw)
                all_tokens.extend(toks)
                all_mask.extend(m)
                i += 1
            elif role == "tool":
                st = i
                while i < len(messages) and messages[i]["role"] == "tool":
                    i += 1
                toks, m = self._process_turn(messages, st, i, is_assistant=False, **tkw)
                all_tokens.extend(toks)
                all_mask.extend(m)
            else:
                toks, m = self._process_turn(messages, i, i + 1, is_assistant=False, **tkw)
                all_tokens.extend(toks)
                all_mask.extend(m)
                i += 1

        ft = full_tokens[0]
        full_list = ft.ids if hasattr(ft, "ids") else ft.tolist()
        if len(all_tokens) != len(full_list) or all_tokens != full_list:
            logger.warning(
                "Token mismatch (full=%d, concat=%d). Using concatenated version.",
                len(full_list), len(all_tokens),
            )

        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        loss_mask = torch.tensor(all_mask, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        seq_len = input_ids.size(0)
        if seq_len < self.max_length:
            pad_id = self.tokenizer.pad_token_id or 0
            pad_len = self.max_length - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=torch.long)])
        elif seq_len > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            else:
                raise ValueError(f"Sequence length {seq_len} > max_length {self.max_length}")

        position_ids = torch.arange(len(input_ids), dtype=torch.long) * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
