"""GLM-style partial-bidirectional attention masks + unified builder.

References
----------
Du et al., "GLM: General Language Model Pretraining with Autoregressive Blank
Infilling" (https://arxiv.org/abs/2103.10360). See Figure 2.

Mask structure (rows = query tokens, columns = key tokens):

                Part A (context)    Part B (response)
    Part A   [  bidirectional  ][    blocked       ]
    Part B   [      full       ][   causal (lower) ]

- Part A tokens attend to each other bidirectionally.
- Part A tokens never attend to Part B tokens.
- Part B tokens attend to the full Part A plus their causal prefix in Part B.
- Padding (attention_mask_2d == 0) is blocked on both axes.

This matches GLM for the blank-infilling setup. In the language-model SFT
case we use, Part A = context / user prompt (up to the first loss-masked
token) and Part B = the assistant response(s).

Design note
-----------
    :func:`build_attention_4d` is the single entry point that both the outer
forward (``FocuSFTTrainer.compute_loss``) and the inner forward
(``ParametricMemory._inner_forward``) call.  It dispatches on
``attn_mode`` and returns a 4D additive mask. Keeping a single dispatch
point ensures the two loops cannot
silently disagree on their attention pattern.
"""

from __future__ import annotations

import torch


# ── Low-level mask builders ──────────────────────────────────────────


def build_glm_4d_mask(
    attention_mask_2d: torch.Tensor,
    context_lens: torch.Tensor,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct a GLM-style additive 4D attention mask.

    Parameters
    ----------
    attention_mask_2d : (B, S) long/bool
        1 on real tokens, 0 on padding.
    context_lens : (B,) long
        Per-sample length of Part A (context). Tokens in [0:context_lens[i]]
        are bidirectional; tokens in [context_lens[i]:] are causal.
        If context_lens[i] == S (or >= S), the whole sequence is Part A
        (fully bidirectional) — useful for inner-loop / prefill forwards.
        If context_lens[i] == 0, the whole sequence is causal (matches
        standard causal attention).
    target_dtype : torch.dtype
        Output dtype. Additive masks are typically float matching the
        attention computation dtype (bf16/fp16/fp32).

    Returns
    -------
    (B, 1, S, S) tensor
        Additive mask: 0.0 on allowed positions, large negative on
        blocked. Suitable for SDPA / eager attention.
    """
    if attention_mask_2d.ndim != 2:
        raise ValueError(f"attention_mask_2d must be (B, S), got {attention_mask_2d.shape}")
    B, S = attention_mask_2d.shape
    device = attention_mask_2d.device

    neg_inf = torch.finfo(target_dtype).min

    idx = torch.arange(S, device=device)
    i = idx[:, None]
    j = idx[None, :]

    ctx = context_lens.to(device=device, dtype=torch.long).clamp_(0, S).view(B, 1, 1)

    q_in_A = (i < ctx)
    k_in_A = (j < ctx)
    q_in_B = ~q_in_A
    k_in_B = ~k_in_A
    causal_order = (j <= i)

    allowed = (
        (q_in_A & k_in_A)
        | (q_in_B & k_in_A)
        | (q_in_B & k_in_B & causal_order)
    )

    pad = attention_mask_2d.to(torch.bool)
    pad_row = pad[:, :, None]
    pad_col = pad[:, None, :]
    allowed = allowed & pad_row & pad_col

    out = torch.zeros((B, 1, S, S), dtype=target_dtype, device=device)
    out = out.masked_fill(~allowed.unsqueeze(1), neg_inf)
    return out


def context_lens_from_loss_mask(loss_mask: torch.Tensor) -> torch.Tensor:
    """Infer per-sample context length from the loss mask.

    Context length = position of the first loss-masked (=1) token.
    If a sample has no loss-masked tokens, context_len = sequence length
    (treat the whole sequence as Part A; this degenerates to fully
    bidirectional attention).
    """
    if loss_mask.ndim != 2:
        raise ValueError(f"loss_mask must be (B, S), got {loss_mask.shape}")
    B, S = loss_mask.shape
    m = loss_mask.to(torch.bool)
    has_any = m.any(dim=1)
    first = m.to(torch.long).argmax(dim=1)
    first = torch.where(has_any, first, torch.full_like(first, S))
    return first


def build_glm_4d_mask_multiturn(
    attention_mask_2d: torch.Tensor,
    loss_mask: torch.Tensor,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Turn-aware GLM 4D mask for multi-turn chat data.

    The loss mask defines turn structure implicitly:
      - Contiguous runs of 0 (+ attention_mask=1) are **user/system turns**
        (Part-A-like, bidirectional within the turn).
      - Contiguous runs of 1 are **assistant turns** (Part-B-like, causal
        within the turn).

    Cross-block rules: a later token always sees earlier tokens (user
    and assistant alike); future tokens are never visible.
    Padding positions (attention_mask=0) are blocked on both axes.

    Degenerate cases:
      - If ``loss_mask`` is all-1 (e.g. eval-time inner where the whole
        prompt is treated as a single assistant turn), this reduces to a
        pure causal mask.
      - If ``loss_mask`` is all-0 (eval-time inner where the whole prompt
        is treated as context), this reduces to fully bidirectional.
    """
    if attention_mask_2d.shape != loss_mask.shape:
        raise ValueError("attention_mask_2d and loss_mask must have the same shape")
    B, S = attention_mask_2d.shape
    device = attention_mask_2d.device

    neg_inf = torch.finfo(target_dtype).min

    lm = loss_mask.to(torch.long)
    changes = torch.zeros_like(lm)
    changes[:, 1:] = (lm[:, 1:] != lm[:, :-1]).to(lm.dtype)
    turn_id = torch.cumsum(changes, dim=1)
    is_asst = lm.to(torch.bool)

    ti = turn_id.unsqueeze(2)
    tj = turn_id.unsqueeze(1)
    ai = is_asst.unsqueeze(2)
    aj = is_asst.unsqueeze(1)

    idx = torch.arange(S, device=device)
    i = idx.view(1, S, 1)
    j = idx.view(1, 1, S)

    same_turn = (ti == tj)
    earlier = (tj < ti)

    same_user = same_turn & (~ai) & (~aj)
    same_asst = same_turn & ai & aj & (j <= i)

    allowed = same_user | same_asst | earlier

    pad = attention_mask_2d.to(torch.bool)
    allowed = allowed & pad.unsqueeze(2) & pad.unsqueeze(1)

    out = torch.zeros((B, 1, S, S), dtype=target_dtype, device=device)
    out = out.masked_fill(~allowed.unsqueeze(1), neg_inf)
    return out


# ── Unified dispatch ─────────────────────────────────────────────────


VALID_ATTN_MODES = ("glm_bidir",)


def build_attention_4d(
    attn_mode: str,
    attention_mask_2d: torch.Tensor,
    loss_mask: torch.Tensor | None,
    target_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Construct the 4D attention mask for a given mode.

    Both the outer forward (trainer.compute_loss) and the inner forward
    (memory._inner_forward) must route through this function so they
    cannot silently disagree on attention pattern.

    Parameters
    ----------
    attn_mode : str
        One of ``VALID_ATTN_MODES``.
    attention_mask_2d : (B, S)
        1 = real token, 0 = padding.
    loss_mask : (B, S) or None
        Needed for ``glm_bidir`` (multi-turn, derives user/assistant
        Needed for ``glm_bidir`` (multi-turn, derives user/assistant blocks).
    target_dtype : torch.dtype
        Additive-mask dtype.

    Returns
    -------
    (B, 1, S, S) additive tensor.
    """
    if attn_mode not in VALID_ATTN_MODES:
        raise ValueError(
            f"unknown attn_mode={attn_mode!r}; expected one of {VALID_ATTN_MODES}"
        )
    if loss_mask is None:
        raise ValueError(f"attn_mode={attn_mode!r} requires a loss_mask")

    if attn_mode == "glm_bidir":
        return build_glm_4d_mask_multiturn(
            attention_mask_2d, loss_mask, target_dtype=target_dtype,
        )
    raise RuntimeError(f"unreachable: attn_mode={attn_mode!r}")
