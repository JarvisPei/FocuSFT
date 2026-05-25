"""Tests for FocuSFT bidirectional-context attention masks."""

import torch

from focusft.attention import (
    build_attention_4d,
    build_glm_4d_mask,
    build_glm_4d_mask_multiturn,
    context_lens_from_loss_mask,
)


def test_simple_glm_block_structure():
    attn = torch.ones(1, 6, dtype=torch.long)
    context_lens = torch.tensor([3])
    mask = build_glm_4d_mask(attn, context_lens, target_dtype=torch.float32)
    allowed = mask.squeeze(0).squeeze(0) == 0.0

    assert allowed[:3, :3].all()
    assert not allowed[:3, 3:].any()
    assert allowed[3:, :3].all()
    for i in range(3, 6):
        for j in range(3, 6):
            assert allowed[i, j].item() is (j <= i)


def test_multiturn_glm_mask():
    loss = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1]])
    attn = torch.ones_like(loss)
    mask = build_glm_4d_mask_multiturn(attn, loss, target_dtype=torch.float32)
    allowed = mask.squeeze(0).squeeze(0) == 0.0

    assert allowed[0].tolist() == [True, True, False, False, False, False, False, False]
    assert allowed[3].tolist() == [True, True, True, True, False, False, False, False]
    assert allowed[5].tolist() == [True, True, True, True, True, True, False, False]
    assert allowed[7].tolist() == [True, True, True, True, True, True, True, True]


def test_context_lens_from_loss_mask():
    loss = torch.tensor([
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
    ])
    assert context_lens_from_loss_mask(loss).tolist() == [2, 4, 0]


def test_dispatcher_uses_multiturn_mask():
    attn = torch.ones(1, 6, dtype=torch.long)
    loss = torch.tensor([[0, 0, 1, 0, 0, 1]])
    dispatched = build_attention_4d("glm_bidir", attn, loss, target_dtype=torch.float32)
    direct = build_glm_4d_mask_multiturn(attn, loss, target_dtype=torch.float32)
    assert torch.equal(dispatched, direct)
