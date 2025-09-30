"""
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def mps_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
    """
    Simple attention implementation suitable for MPS. Expects shapes:
    - q: (B, H, Tq, D)
    - k: (B, H, Tk, D)
    - v: (B, H, Tk, D)
    Returns: (B, H, Tq, D)
    """
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.einsum("bhtd,bhkd->bhtk", q, k) * scale
    if causal:
        Tq, Tk = q.shape[-2], k.shape[-2]
        mask = torch.ones(Tq, Tk, device=q.device, dtype=torch.bool).triu(diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = F.softmax(scores, dim=-1)
    out = torch.einsum("bhtk,bhkd->bhtd", probs, v)
    return out


def streaming_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_block_size: int = 1024,
    k_block_size: int = 1024,
    causal: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Streaming attention using per-block accumulators (runs on MPS/CPU).
    Shapes:
      q: (B, H, Tq, D), k: (B, H, Tk, D), v: (B, H, Tk, D)
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    B, H, Tq, D = q.shape
    _, _, Tk, Dk = k.shape
    assert D == Dk

    device, dtype = q.device, q.dtype
    out = torch.zeros((B, H, Tq, D), device=device, dtype=dtype)
    scale = 1.0 / (D ** 0.5)

    # Iterate over q blocks
    for qs in range(0, Tq, q_block_size):
        qe = min(Tq, qs + q_block_size)
        Bq = qe - qs

        q_block = q[:, :, qs:qe, :].to(torch.float32)
        # Accumulators per (B, H, Bq)
        m = torch.full((B, H, Bq), float("-inf"), device=device, dtype=torch.float32)
        s = torch.zeros((B, H, Bq), device=device, dtype=torch.float32)
        wv = torch.zeros((B, H, Bq, D), device=device, dtype=torch.float32)

        # Iterate over k blocks
        for ks in range(0, Tk, k_block_size):
            ke = min(Tk, ks + k_block_size)
            Bk = ke - ks
            k_block = k[:, :, ks:ke, :].to(torch.float32)
            v_block = v[:, :, ks:ke, :].to(torch.float32)

            scores = torch.einsum("bhqd,bhkd->bhqk", q_block, k_block) * scale

            if causal:
                # Build a causal mask relative to absolute positions
                q_pos = torch.arange(qs, qe, device=device)
                k_pos = torch.arange(ks, ke, device=device)
                mask = k_pos.unsqueeze(0).unsqueeze(0).unsqueeze(0) > q_pos.unsqueeze(0).unsqueeze(0).unsqueeze(2)
                scores = scores.masked_fill(mask, float("-inf"))

            local_max = scores.amax(dim=-1)
            exp_scores = torch.exp(scores - local_max.unsqueeze(-1))
            sum_exp = exp_scores.sum(dim=-1)
            wv_chunk = torch.einsum("bhqk,bhkd->bhqd", exp_scores, v_block)

            first = torch.isinf(m)
            if first.any():
                idx = first
                m[idx] = local_max[idx]
                s[idx] = sum_exp[idx]
                wv[idx] = wv_chunk[idx]

            merge_idx = ~first
            if merge_idx.any():
                m_old = m[merge_idx]
                s_old = s[merge_idx]
                wv_old = wv[merge_idx]
                lm_new = local_max[merge_idx]
                se_new = sum_exp[merge_idx]
                wv_new = wv_chunk[merge_idx]

                new_m = torch.maximum(m_old, lm_new)
                alpha = torch.exp(m_old - new_m)
                beta = torch.exp(lm_new - new_m)
                s[merge_idx] = s_old * alpha + se_new * beta
                wv[merge_idx] = wv_old * alpha.unsqueeze(-1) + wv_new * beta.unsqueeze(-1)
                m[merge_idx] = new_m

            del scores, local_max, exp_scores, sum_exp, wv_chunk, k_block, v_block

        out_block = (wv / (s.unsqueeze(-1) + eps)).to(dtype)
        out[:, :, qs:qe, :] = out_block
        del q_block, m, s, wv, out_block

    return out



def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    use_streaming: bool = False,
    q_block_size: int = 1024,
    k_block_size: int = 1024,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Unified MPS attention entrypoint.
    - When use_streaming=True, uses streaming_softmax_attention to reduce memory.
    - Otherwise, falls back to a simple full-softmax attention.
    """
    if use_streaming:
        return streaming_softmax_attention(
            q, k, v,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            causal=causal,
            eps=eps,
        )
    return mps_attention(q, k, v, causal=causal)


if __name__ == "__main__":
    # Basic sanity test comparing streaming vs baseline on small sizes
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    B, H, Tq, Tk, D = 1, 4, 256, 256, 64
    q = torch.randn(B, H, Tq, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, Tk, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, Tk, D, device=device, dtype=torch.float16)

    out_ref = attention(q, k, v, causal=True, use_streaming=False)
    out_stream = attention(q, k, v, causal=True, use_streaming=True, q_block_size=64, k_block_size=64)
    err = (out_ref.float() - out_stream.float()).abs().max().item()
    print("max_abs_err:", err)