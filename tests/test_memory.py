"""Tests for transient fast LoRA adapters."""

import torch
import torch.nn as nn

from focusft.memory import InnerLoRALinear, InnerLoRAManager, MemoryConfig


def test_inner_lora_forward_and_reset():
    base = nn.Linear(16, 32)
    layer = InnerLoRALinear(base, rank=4, alpha=8.0, dropout=0.0)

    x = torch.randn(2, 5, 16)
    out = layer(x)

    assert out.shape == (2, 5, 32)
    assert [tuple(p.shape) for p in layer.inner_parameters()] == [(4, 16), (32, 4)]
    layer.reset_inner()
    assert layer.mem_inner_B.abs().sum().item() == 0.0


def test_inner_lora_manager_injects_top_layers_only():
    class FakeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(64, 128)
            self.up_proj = nn.Linear(64, 128)
            self.down_proj = nn.Linear(128, 64)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = FakeMLP()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer() for _ in range(4)])

    model = FakeModel()
    config = MemoryConfig(
        inner_layer_fraction=0.5,
        inner_target_modules="gate_proj,up_proj,down_proj",
        inner_rank=8,
        inner_alpha=16.0,
    )
    manager = InnerLoRAManager(model, config)

    assert len(manager.adapter_layers) == 6
    assert isinstance(model.layers[0].mlp.gate_proj, nn.Linear)
    assert isinstance(model.layers[1].mlp.gate_proj, nn.Linear)
    assert isinstance(model.layers[2].mlp.gate_proj, InnerLoRALinear)
    assert isinstance(model.layers[3].mlp.down_proj, InnerLoRALinear)
    assert len(manager.get_inner_parameters()) == 12
