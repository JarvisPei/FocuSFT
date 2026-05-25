"""Fast-weight parametric memory used by FocuSFT.

FocuSFT inserts transient LoRA adapters into the top fraction of transformer
FFN layers. At every training step these adapters are reset, adapted for a
small number of inner-loop gradient steps on the current batch, and then used
only to shape the outer SFT update. They are stripped from saved checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from focusft.attention import build_attention_4d


@dataclass
class MemoryConfig:
    enabled: bool = True
    num_inner_steps: int = 2
    inner_lr: float = 1.0
    inner_grad_clip: float = 1.0
    inner_rank: int = 32
    inner_alpha: float = 64.0
    inner_dropout: float = 0.0
    inner_layer_fraction: float = 0.35
    inner_target_modules: str = "gate_proj,up_proj,down_proj"
    inner_gradient_checkpointing: bool = True
    sync_inner: bool = False


class InnerLoRALinear(nn.Module):
    """LoRA adapter used as per-step fast weights."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.mem_inner_A = nn.Parameter(torch.zeros(rank, in_features, device=device, dtype=dtype))
        self.mem_inner_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.reset_inner()

    def reset_inner(self) -> None:
        nn.init.kaiming_uniform_(self.mem_inner_A, a=5**0.5)
        nn.init.zeros_(self.mem_inner_B)

    def inner_parameters(self) -> list[nn.Parameter]:
        return [self.mem_inner_A, self.mem_inner_B]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        x_cast = x.to(self.mem_inner_A.dtype) if x.dtype != self.mem_inner_A.dtype else x
        inner_out = self.dropout(x_cast) @ self.mem_inner_A.T @ self.mem_inner_B.T
        return base_out + self.scaling * inner_out.to(base_out.dtype)


class InnerLoRAManager:
    """Inject fast LoRA adapters into selected top transformer layers."""

    def __init__(self, model: nn.Module, config: MemoryConfig):
        self.model = model
        self.config = config
        self.adapter_layers: list[InnerLoRALinear] = []
        self._inject()
        if not self.adapter_layers:
            raise ValueError(
                "No inner LoRA adapters were injected. Check inner_target_modules "
                "and the model architecture."
            )

    def _resolve_layers(self, model: nn.Module):
        for attr in ("module", "model", "base_model", "model"):
            model = getattr(model, attr, model)
        for path in ("model.layers", "layers", "model.transformer.blocks", "transformer.blocks"):
            current = model
            ok = True
            for part in path.split("."):
                if not hasattr(current, part):
                    ok = False
                    break
                current = getattr(current, part)
            if ok:
                return current
        raise AttributeError("Cannot locate transformer layers for inner LoRA injection.")

    def _inject(self) -> None:
        layers = self._resolve_layers(self.model)
        num_layers = len(layers)
        fraction = max(0.0, min(1.0, self.config.inner_layer_fraction))
        start = max(0, int(num_layers * (1.0 - fraction)))
        targets = [name.strip() for name in self.config.inner_target_modules.split(",") if name.strip()]

        for idx in range(start, num_layers):
            self._inject_layer(layers[idx], targets)

    def _inject_layer(self, layer: nn.Module, targets: list[str]) -> None:
        for submodule_name in ("mlp", "self_attn", "attention"):
            if hasattr(layer, submodule_name):
                self._inject_module(getattr(layer, submodule_name), targets)
        self._inject_module(layer, targets)

    def _inject_module(self, parent: nn.Module, targets: list[str]) -> None:
        for name in targets:
            if not hasattr(parent, name):
                continue
            base = getattr(parent, name)
            linear = base if isinstance(base, nn.Linear) else getattr(base, "base_layer", None)
            if not isinstance(linear, nn.Linear) or isinstance(base, InnerLoRALinear):
                continue

            adapter = InnerLoRALinear(
                linear,
                rank=self.config.inner_rank,
                alpha=self.config.inner_alpha,
                dropout=self.config.inner_dropout,
            )
            if linear is base:
                setattr(parent, name, adapter)
            else:
                base.base_layer = adapter
            self.adapter_layers.append(adapter)

    def reset_inner(self) -> None:
        for layer in self.adapter_layers:
            layer.reset_inner()

    def get_inner_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.adapter_layers:
            params.extend(layer.inner_parameters())
        return params


class ParametricMemory:
    """Runs the FocuSFT inner loop over transient LoRA fast weights."""

    def __init__(self, model: nn.Module, config: MemoryConfig):
        self.model = model
        self.config = config
        self.inner_manager = InnerLoRAManager(model, config)
        self.inner_params = self.inner_manager.get_inner_parameters()
        self.inner_optimizer = torch.optim.SGD(self.inner_params, lr=config.inner_lr)

    def reset_inner(self) -> None:
        self.inner_manager.reset_inner()

    def get_all_inner_parameters(self) -> list[nn.Parameter]:
        return self.inner_manager.get_inner_parameters()

    def _resolve_decoder(self) -> nn.Module | None:
        model = self.model
        for attr in ("module", "model", "base_model", "model"):
            model = getattr(model, attr, model)
        if hasattr(model, "layers") and (
            hasattr(model, "embed_tokens") or hasattr(model, "word_embeddings")
        ):
            return model
        return None

    @staticmethod
    def _embed(decoder: nn.Module):
        return getattr(decoder, "embed_tokens", getattr(decoder, "word_embeddings", None))

    def _resolve_lm_head(self) -> nn.Module | None:
        model = self.model
        for attr in ("base_model", "model"):
            if hasattr(model, "lm_head"):
                return model.lm_head
            model = getattr(model, attr, model)
        return getattr(model, "lm_head", None)

    def _run_layer(self, layer, hidden, attention_mask, position_ids, position_embeddings):
        try:
            out = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
            )
        except TypeError:
            out = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        return out if isinstance(out, torch.Tensor) else out[0]

    def _inner_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        decoder = self._resolve_decoder()
        lm_head = self._resolve_lm_head()
        if decoder is None or lm_head is None:
            inner_attention_mask = self._build_attention_mask(attention_mask, loss_mask)
            return self.model(input_ids=input_ids, attention_mask=inner_attention_mask, use_cache=False)

        layers = decoder.layers
        num_layers = len(layers)
        fraction = max(0.0, min(1.0, self.config.inner_layer_fraction))
        start_idx = max(0, int(num_layers * (1.0 - fraction)))

        embed_fn = self._embed(decoder)
        inputs_embeds = embed_fn(input_ids)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = decoder.rotary_emb(inputs_embeds, position_ids)
        inner_attention_mask = self._build_attention_mask(attention_mask, loss_mask)

        layer_types = getattr(decoder.config, "layer_types", None)

        def layer_mask(layer_idx: int):
            if isinstance(inner_attention_mask, dict) and layer_types is not None:
                return inner_attention_mask.get(
                    layer_types[layer_idx],
                    inner_attention_mask.get("full_attention"),
                )
            return inner_attention_mask

        hidden_states = inputs_embeds
        with torch.no_grad():
            for idx in range(start_idx):
                hidden_states = self._run_layer(
                    layers[idx],
                    hidden_states,
                    layer_mask(idx),
                    position_ids,
                    position_embeddings,
                )

        hidden_states = hidden_states.detach()
        for idx in range(start_idx, num_layers):
            if self.config.inner_gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self._run_layer,
                    layers[idx],
                    hidden_states,
                    layer_mask(idx),
                    position_ids,
                    position_embeddings,
                    use_reentrant=False,
                )
            else:
                hidden_states = self._run_layer(
                    layers[idx],
                    hidden_states,
                    layer_mask(idx),
                    position_ids,
                    position_embeddings,
                )

        if hasattr(decoder, "norm") and decoder.norm is not None:
            hidden_states = decoder.norm(hidden_states)

        return SimpleNamespace(logits=lm_head(hidden_states))

    def _build_attention_mask(
        self,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        param_dtype = next(
            (p.dtype for p in self.model.parameters() if p.is_floating_point()),
            torch.float32,
        )
        return build_attention_4d(
            "glm_bidir",
            attention_mask,
            loss_mask,
            target_dtype=param_dtype,
        )

    def _compute_inner_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self._inner_forward(input_ids, attention_mask, loss_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = loss_mask[:, 1:].contiguous().float()

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)

        return (loss_per_token * shift_mask).sum() / shift_mask.sum().clamp_min(1)

    def adapt(self, inputs: dict[str, torch.Tensor]) -> list[float]:
        if not self.config.enabled:
            return []

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        loss_mask = inputs["loss_mask"]
        if loss_mask.sum() == 0:
            return []

        losses: list[float] = []
        for _ in range(self.config.num_inner_steps):
            if self.config.sync_inner:
                self.inner_optimizer.zero_grad(set_to_none=True)

            with torch.enable_grad():
                loss = self._compute_inner_loss(input_ids, attention_mask, loss_mask)

            if torch.isnan(loss) or torch.isinf(loss):
                losses.append(float("nan"))
                continue
            losses.append(loss.item())

            if self.config.sync_inner:
                torch.autograd.backward(loss, inputs=self.inner_params)
                if self.config.inner_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.inner_params, self.config.inner_grad_clip)
                self.inner_optimizer.step()
            else:
                grads = torch.autograd.grad(
                    loss,
                    self.inner_params,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )
                with torch.no_grad():
                    for param, grad in zip(self.inner_params, grads):
                        if grad is None:
                            continue
                        if self.config.inner_grad_clip > 0:
                            grad_norm = grad.norm().item()
                            if grad_norm > self.config.inner_grad_clip:
                                grad = grad * (self.config.inner_grad_clip / (grad_norm + 1e-12))
                        param.data.sub_(self.config.inner_lr * grad)

        if self.config.sync_inner:
            self.inner_optimizer.zero_grad(set_to_none=True)

        return losses
