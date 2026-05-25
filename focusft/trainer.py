"""Hugging Face Trainer implementation for FocuSFT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import TrainingArguments

from focusft.attention import build_attention_4d
from focusft.memory import MemoryConfig, ParametricMemory


@dataclass
class FocuSFTConfig(TrainingArguments):
    """Training arguments for the FocuSFT main method."""

    remove_unused_columns: bool = field(default=False)

    focusft_enabled: bool = field(default=True, metadata={"help": "Enable FocuSFT bilevel training."})
    num_inner_steps: int = field(default=2, metadata={"help": "Number of inner-loop SGD steps."})
    inner_lr: float = field(default=1.0, metadata={"help": "Inner-loop learning rate."})
    inner_grad_clip: float = field(default=1.0, metadata={"help": "Inner-loop gradient clipping norm."})
    inner_rank: int = field(default=32, metadata={"help": "Fast LoRA rank."})
    inner_alpha: float = field(default=64.0, metadata={"help": "Fast LoRA alpha."})
    inner_dropout: float = field(default=0.0, metadata={"help": "Fast LoRA dropout."})
    inner_layer_fraction: float = field(
        default=0.35,
        metadata={"help": "Fraction of top transformer layers with fast LoRA adapters."},
    )
    inner_target_modules: str = field(
        default="gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated FFN module names for fast LoRA injection."},
    )
    inner_gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use activation checkpointing inside the inner-loop forward."},
    )
    sync_inner: bool = field(
        default=False,
        metadata={"help": "Use optimizer.step for inner updates. False avoids DDP all-reduce on inner grads."},
    )
    attention_mode: str = field(
        default="glm_bidir",
        metadata={"help": "Attention mask for inner and outer loops: 'glm_bidir' or 'causal'."},
    )


class FocuSFTTrainer(transformers.Trainer):
    """Bilevel SFT trainer with transient fast-weight LoRA adapters."""

    def __init__(self, args: FocuSFTConfig, **kwargs):
        super().__init__(args=args, **kwargs)
        self.memory_config = MemoryConfig(
            enabled=args.focusft_enabled,
            num_inner_steps=args.num_inner_steps,
            inner_lr=args.inner_lr,
            inner_grad_clip=args.inner_grad_clip,
            inner_rank=args.inner_rank,
            inner_alpha=args.inner_alpha,
            inner_dropout=args.inner_dropout,
            inner_layer_fraction=args.inner_layer_fraction,
            inner_target_modules=args.inner_target_modules,
            inner_gradient_checkpointing=args.inner_gradient_checkpointing,
            sync_inner=args.sync_inner,
            attention_mode=args.attention_mode,
        )

        self.memory: ParametricMemory | None = None
        if self.memory_config.enabled:
            self.memory = ParametricMemory(model=self.model, config=self.memory_config)

    def create_optimizer(self):
        """Exclude transient inner-loop parameters from the outer optimizer."""
        if self.optimizer is None and self.memory is not None:
            inner_params = self.memory.get_all_inner_parameters()
            for param in inner_params:
                param.requires_grad = False
            super().create_optimizer()
            for param in inner_params:
                param.requires_grad = True
            return self.optimizer
        return super().create_optimizer()

    def save_model(self, output_dir=None, _internal_call=False):
        """Save a vanilla HF checkpoint without transient inner LoRA weights."""
        import os

        if output_dir is None:
            output_dir = self.args.output_dir

        if self.args.should_save:
            os.makedirs(output_dir, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            clean_state = {}
            for key, value in unwrapped.state_dict().items():
                if "mem_inner_A" in key or "mem_inner_B" in key:
                    continue
                clean_state[key.replace(".base_layer.", ".")] = value

            unwrapped.save_pretrained(
                output_dir,
                state_dict=clean_state,
                safe_serialization=True,
            )
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)

        self.accelerator.wait_for_everyone()

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        loss_mask = inputs["loss_mask"]

        if self.memory is not None and self.memory_config.enabled:
            self.memory.reset_inner()
            inner_losses = self.memory.adapt(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "loss_mask": loss_mask,
                }
            )
            if inner_losses and self.state.global_step % max(1, self.args.logging_steps) == 0:
                valid = [loss for loss in inner_losses if loss == loss]
                if valid:
                    self.log({"inner/loss": sum(valid) / len(valid)})

        outer_attention_mask = attention_mask
        if self.args.attention_mode != "causal":
            param_dtype = next(
                (p.dtype for p in model.parameters() if p.is_floating_point()),
                torch.float32,
            )
            outer_attention_mask = build_attention_4d(
                self.args.attention_mode,
                attention_mask,
                loss_mask,
                target_dtype=param_dtype,
            )

        outputs = model(
            input_ids=input_ids,
            attention_mask=outer_attention_mask,
            use_cache=False,
        )
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = loss_mask[:, 1:].contiguous().float()

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)

        loss = (loss_per_token * shift_mask).sum() / shift_mask.sum().clamp_min(1)
        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        logits = outputs.logits.detach().contiguous()
        labels = inputs.get("input_ids")
        if labels is not None:
            labels = labels.detach().contiguous()
        return (loss.detach(), logits, labels)
