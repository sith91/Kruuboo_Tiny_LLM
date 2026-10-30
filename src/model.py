"""
TinyLLM — Upgraded Model Architecture
======================================

Changes from the original:
  1. Dimensions: d_model 64→256, n_heads 2→4, n_layers 2→4, d_ff 128→1024
     → ~4 M parameters  (was ~50 K — 80x more capacity)

  2. RoPE (Rotary Position Embeddings) replaces the learned position table.
     - No position table that overflows if seq_len > max_seq_len at export time
     - Better length generalisation

  3. Flash Attention via torch.nn.functional.scaled_dot_product_attention
     - Uses PyTorch 2.0's fused kernel when available (faster, less memory)
     - Falls back to manual matmul on older PyTorch

  4. SwiGLU feed-forward (used by LLaMA / PaLM) instead of plain GELU MLP
     - Empirically better perplexity at the same parameter count

  5. GPT-2 style weight initialisation
     - Residual projections scaled by 1/sqrt(n_layers) to keep
       variance stable at initialisation

  6. Embedding scale removed (√d_model scaling on embeddings can hurt
     small models — RoPE handles position separately)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Rotary Position Embeddings (RoPE)
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10_000.0) -> torch.Tensor:
    """
    Returns a (max_seq_len, head_dim/2, 2) tensor of (cos, sin) values.
    head_dim must be even.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    # θ_i = base^{-2i/head_dim}
    theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    # positions  [max_seq_len]
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # outer product → [max_seq_len, half]
    freqs = torch.outer(t, theta)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # [T, half, 2]


def _apply_rope(x: torch.Tensor, rope_freqs: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
    """
    x              : [B, n_heads, T, head_dim]
    rope_freqs     : [max_seq_len, head_dim/2, 2]  (cos, sin pair per frequency)
    position_offset: token position offset for KV cache decoding
    """
    B, H, T, D = x.shape
    half = D // 2
    # split into pairs
    x1 = x[..., :half]   # [B, H, T, half]
    x2 = x[..., half:]   # [B, H, T, half]
    # broadcast rope_freqs with offset over batch and head dims
    cos = rope_freqs[position_offset : position_offset + T, :, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, T, half]
    sin = rope_freqs[position_offset : position_offset + T, :, 1].unsqueeze(0).unsqueeze(0)
    # rotate
    rotated = torch.cat([x1 * cos - x2 * sin,
                         x1 * sin + x2 * cos], dim=-1)
    return rotated


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Attention with Flash Attention & KV Cache
# ─────────────────────────────────────────────────────────────────────────────

_FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")


class TinyAttention(nn.Module):
    """
    Causal multi-head / grouped-query self-attention (MHA / GQA / MQA) with RoPE,
    Flash Attention, and KV Caching.

    Parameters
    ----------
    d_model    : model dimension (must be divisible by n_heads)
    n_heads    : number of query attention heads
    dropout    : attention dropout probability
    n_kv_heads : number of key/value heads (defaults to n_heads; must divide n_heads)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, n_kv_heads: Optional[int] = None):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        if n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")

        self.head_dim = d_model // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = dropout

        # Single fused projection for Q, K, V (supports GQA when n_kv_heads < n_heads)
        total_qkv_heads = self.n_heads + 2 * self.n_kv_heads
        self.qkv = nn.Linear(d_model, total_qkv_heads * self.head_dim, bias=False)
        # Output projection
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        offset = past_key.shape[-2] if past_key is not None else 0

        # Project and split into Q [B, n_heads, T, D], K [B, n_kv_heads, T, D], V [B, n_kv_heads, T, D]
        total_qkv_heads = self.n_heads + 2 * self.n_kv_heads
        qkv = self.qkv(x).reshape(B, T, total_qkv_heads, self.head_dim).transpose(1, 2)
        q, k, v = torch.split(qkv, [self.n_heads, self.n_kv_heads, self.n_kv_heads], dim=1)

        # Apply RoPE to Q and K with exact position offset
        q = _apply_rope(q, rope_freqs, position_offset=offset)
        k = _apply_rope(k, rope_freqs, position_offset=offset)

        # Concatenate past KV cache
        if past_key is not None:
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)

        present_k = k
        present_v = v
        total_seq = k.shape[-2]

        # Expand K and V heads for Grouped-Query Attention (GQA) when n_kv_heads < n_heads
        if self.n_rep > 1:
            k_attn = k.repeat_interleave(self.n_rep, dim=1)
            v_attn = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_attn = k
            v_attn = v

        if _FLASH_AVAILABLE and past_key is None and not torch.jit.is_tracing():
            # PyTorch 2.0+ fused scaled dot-product attention (for full sequence / training)
            out = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k_attn.transpose(-2, -1)) * scale
            q_pos = offset + torch.arange(T, device=x.device).unsqueeze(1)
            k_pos = torch.arange(total_seq, device=x.device).unsqueeze(0)
            mask = q_pos >= k_pos
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v_attn)

        # Re-assemble heads
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(out), present_k, present_v


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class TinyFeedForward(nn.Module):
    """
    SwiGLU FFN  (used by LLaMA, PaLM, Mistral …)

    FFN(x) = (W1·x ⊙ σ(W3·x)) · W2

    Uses 2/3 of the standard d_ff to keep parameter count comparable.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Standard recommendation: use ≈ 2/3 of d_ff for SwiGLU gate
        hidden = int(d_ff * 2 / 3)
        # round to nearest multiple of 64 for hardware alignment
        hidden = ((hidden + 63) // 64) * 64

        self.w1 = nn.Linear(d_model, hidden, bias=False)   # gate
        self.w3 = nn.Linear(d_model, hidden, bias=False)   # value
        self.w2 = nn.Linear(hidden, d_model, bias=False)   # down-project
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU (Swish) gating
        return self.w2(self.drop(F.silu(self.w1(x)) * self.w3(x)))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Transformer Block  (Pre-Norm)
# ─────────────────────────────────────────────────────────────────────────────

class TinyTransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attn → residual, LN → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, n_kv_heads: Optional[int] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = TinyAttention(d_model, n_heads, dropout, n_kv_heads=n_kv_heads)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff   = TinyFeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, present_k, present_v = self.attn(
            self.ln1(x), rope_freqs, past_key=past_key, past_value=past_value
        )
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, present_k, present_v


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Full Model
# ─────────────────────────────────────────────────────────────────────────────

class TinyLLM(nn.Module):
    """
    Upgraded Tiny Language Model.

    Default configuration (~4 M parameters, trainable on CPU in minutes):
    ┌─────────────┬──────────┬───────────────────────────────────┐
    │ Parameter   │  Old     │  New                              │
    ├─────────────┼──────────┼───────────────────────────────────┤
    │ d_model     │   64     │  256                              │
    │ n_heads     │    2     │    4  (head_dim = 64)             │
    │ n_layers    │    2     │    4                              │
    │ d_ff        │  128     │ 1024  (SwiGLU hidden ≈ 682)       │
    │ max_seq_len │   32     │  128  (fixed by BPE upgrade)      │
    │ Total params│ ~50 K    │ ~4 M  (80× more capacity)         │
    │ Position    │ learned  │ RoPE  (no overflow risk)          │
    │ Attention   │ manual   │ Flash (PyTorch 2.0+) / KV Cache   │
    │ FFN         │ GELU MLP │ SwiGLU                            │
    └─────────────┴──────────┴───────────────────────────────────┘
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:     int   = 256,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        d_ff:        int   = 1024,
        max_seq_len: int   = 128,
        dropout:     float = 0.1,
        n_kv_heads:  Optional[int] = None,
    ):
        super().__init__()

        self.d_model     = d_model
        self.n_heads     = n_heads
        self.n_kv_heads  = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_layers    = n_layers
        self.d_ff        = d_ff
        self.max_seq_len = max_seq_len
        self.vocab_size  = vocab_size

        # Token embedding only — position is handled by RoPE
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TinyTransformerBlock(d_model, n_heads, d_ff, dropout, n_kv_heads=self.n_kv_heads)
            for _ in range(n_layers)
        ])

        # Final layer norm + language model head
        self.ln_f    = nn.LayerNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embedding ↔ lm_head (saves params, improves quality)
        self.token_embedding.weight = self.lm_head.weight

        # Precompute RoPE frequencies and register as a buffer (moves with .to(device))
        rope = _precompute_rope_freqs(d_model // n_heads, max_seq_len)
        self.register_buffer("rope_freqs", rope, persistent=False)

        # Initialise weights
        self.apply(self._init_weights)

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        """
        GPT-2 style init:
          - Linear / Embedding weights ~ N(0, 0.02)
          - Residual projection weights scaled by 1/√(2·n_layers)
            so the residual stream variance stays ≈ 1 at initialisation.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale down residual projections (output proj of attn + FFN down-proj)
        for name, p in module.named_parameters(recurse=False):
            if name == "weight" and isinstance(module, nn.Linear):
                # Detect residual projections by their parent class
                pass  # handled below

        if isinstance(module, (TinyAttention, TinyFeedForward)):
            # attn.proj  or  ff.w2  are the residual contributions
            proj = getattr(module, "proj", None) or getattr(module, "w2", None)
            if proj is not None:
                nn.init.normal_(
                    proj.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * self.n_layers),
                )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        past_keys: Optional[torch.Tensor] = None,
        past_values: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        x : LongTensor [batch, seq_len]  (token IDs)
        past_keys   : Optional FloatTensor [n_layers, batch, n_heads, past_seq_len, head_dim]
        past_values : Optional FloatTensor [n_layers, batch, n_heads, past_seq_len, head_dim]
        use_cache   : bool, whether to return updated KV cache
        return_hidden_states : bool, whether to also return post-LN hidden states [batch, seq_len, d_model]

        Returns:
            If use_cache is False and past_keys is None and not return_hidden_states:
                logits [batch, seq_len, vocab_size]
            Else:
                logits, present_keys, present_values (, hidden_states)
        """
        _B, T = x.shape
        # Token embeddings (no positional embedding — RoPE handles position)
        h = self.dropout(self.token_embedding(x))   # [B, T, d_model]

        rope = self.rope_freqs

        present_k_list: List[torch.Tensor] = []
        present_v_list: List[torch.Tensor] = []

        return_cache = use_cache or (past_keys is not None)

        for i, block in enumerate(self.blocks):
            pk = past_keys[i] if past_keys is not None else None
            pv = past_values[i] if past_values is not None else None
            h, pres_k, pres_v = block(h, rope, past_key=pk, past_value=pv)
            if return_cache:
                present_k_list.append(pres_k)
                present_v_list.append(pres_v)

        # Final norm + vocab projection
        h = self.ln_f(h)
        logits = self.lm_head(h)                       # [B, T, vocab_size]

        if return_cache:
            present_keys = torch.stack(present_k_list, dim=0)
            present_values = torch.stack(present_v_list, dim=0)
            if return_hidden_states:
                return logits, present_keys, present_values, h
            return logits, present_keys, present_values

        if return_hidden_states:
            return logits, h
        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the post-LayerNorm hidden states before the LM head projection.
        x : LongTensor [batch, seq_len]
        returns hidden states [batch, seq_len, d_model]
        """
        _B, T = x.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}."
        )
        h = self.token_embedding(x)
        rope = self.rope_freqs
        for block in self.blocks:
            h, _, _ = block(h, rope)
        return self.ln_f(h)

    def get_embeddings(self, x: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        """
        Compute mean-pooled, L2-normalized sequence embeddings for native RAG.
        x : LongTensor [batch, seq_len]
        returns normalized embeddings [batch, d_model]
        """
        features = self.forward_features(x)  # [B, T, d_model]
        mask = (x != pad_id).unsqueeze(-1).float()  # [B, T, 1]
        mask_sum = mask.sum(dim=1).clamp(min=1e-9)
        pooled = (features * mask).sum(dim=1) / mask_sum  # [B, d_model]
        return F.normalize(pooled, p=2, dim=-1)

    # ── Convenience ───────────────────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return total (trainable) parameter count."""
        params = (p for p in self.parameters() if (not trainable_only or p.requires_grad))
        return sum(p.numel() for p in params)