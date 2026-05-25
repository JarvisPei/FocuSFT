"""Training entry point for FocuSFT."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field

from transformers import HfArgumentParser

from focusft.data import AgenticSFTDataset, DataConfig
from focusft.models import load_model, load_tokenizer
from focusft.trainer import FocuSFTConfig, FocuSFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B",
        metadata={"help": "Hugging Face model ID or local model path."},
    )
    torch_dtype: str = field(default="bfloat16")
    attn_implementation: str = field(default="sdpa")
    trust_remote_code: bool = field(default=True)


@dataclass
class DataArguments:
    train_data: str = field(default="", metadata={"help": "Comma-separated parquet training files."})
    val_data: str = field(default="", metadata={"help": "Optional comma-separated parquet validation files."})
    max_length: int = field(default=4096)
    truncation: str = field(default="right")
    messages_key: str = field(default="messages")
    tools_key: str = field(default="tools")
    enable_thinking_key: str = field(default="enable_thinking")


def _split_files(value: str) -> list[str]:
    return [path.strip() for path in value.split(",") if path.strip()]


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, FocuSFTConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not data_args.train_data:
        raise ValueError("--train_data is required")

    logger.info("Loading tokenizer: %s", model_args.model_name_or_path)
    tokenizer = load_tokenizer(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    logger.info("Loading model: %s", model_args.model_name_or_path)
    model = load_model(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
        gradient_checkpointing=training_args.gradient_checkpointing,
        max_length=data_args.max_length,
    )

    data_config = DataConfig(
        max_length=data_args.max_length,
        truncation=data_args.truncation,
        messages_key=data_args.messages_key,
        tools_key=data_args.tools_key,
        enable_thinking_key=data_args.enable_thinking_key,
    )

    train_dataset = AgenticSFTDataset(_split_files(data_args.train_data), tokenizer, data_config)
    logger.info("Training samples: %d", len(train_dataset))

    eval_dataset = None
    if data_args.val_data:
        eval_dataset = AgenticSFTDataset(_split_files(data_args.val_data), tokenizer, data_config)
        logger.info("Validation samples: %d", len(eval_dataset))

    trainer = FocuSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    num_inner = sum(p.numel() for p in trainer.memory.get_all_inner_parameters())
    num_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "FocuSFT enabled: %d fast-weight parameters (%.2f%% of total), "
        "inner_steps=%d, inner_lr=%.4f, layer_fraction=%.2f, attention=glm_bidir",
        num_inner,
        100.0 * num_inner / num_total,
        training_args.num_inner_steps,
        training_args.inner_lr,
        training_args.inner_layer_fraction,
    )

    has_checkpoint = os.path.isdir(training_args.output_dir) and any(
        name.startswith("checkpoint-") for name in os.listdir(training_args.output_dir)
    )
    trainer.train(resume_from_checkpoint=has_checkpoint or None)
    trainer.save_model()
    logger.info("Training complete. Model saved to %s", training_args.output_dir)


if __name__ == "__main__":
    main()
