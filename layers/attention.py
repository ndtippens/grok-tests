import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from torch import Tensor
from layers.positionals import RotaryPositionalEmbedding



def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = RotaryPositionalEmbedding(head_dim, max_seq_len)
        self.c_proj = nn.Linear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor, ve: Tensor | None = None):
        B, T = x.size(0), x.size(1) # batch size, sequence length

        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v

        # Transpose to [batch, num_heads, seq_len, head_dim] for scaled_dot_product_attention
        q = q.transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.transpose(1, 2)  # [B, num_heads, T, head_dim]
        v = v.transpose(1, 2)  # [B, num_heads, T, head_dim]

        # Use torch's scaled_dot_product_attention with causal masking
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            scale=0.12  # Custom scale factor
        )

        # Transpose back and reshape
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism using torch's scaled_dot_product_attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for multi-head attention."""
        batch_size, seq_len, d_model = x.size()

        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Use torch's scaled_dot_product_attention
        attention_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)

        return self.w_o(attention_output)


class CausalMultiHeadAttention(nn.Module):
    """Causal (masked) multi-head attention for GPT-2 using torch's scaled_dot_product_attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 1024):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

        # Combined QKV projection like HF GPT-2 (c_attn)
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=True)
        # Output projection (c_proj)
        self.c_proj = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Forward pass with causal masking."""
        batch_size, seq_len, d_model = x.size()

        # Combined QKV projection
        qkv = self.c_attn(x)  # [batch_size, seq_len, 3 * d_model]

        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch_size, seq_len, d_model]

        # Reshape for multi-head attention
        Q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Use torch's scaled_dot_product_attention with causal masking
        attention_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True  # Enable causal masking
        )

        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)

        return self.c_proj(attention_output)


class AdaptiveMask(nn.Module):
    """Soft masking function for adaptive size.
    It masks out the last K values of an input. The masking value
    goes from 1 to 0 gradually, so K can be learned with
    back-propagation.

    Args:
        max_size: maximum size (i.e. input dimension)
        ramp_size: size of the ramp going from 0 to 1
        init_val: initial size proportion not to be masked out
        shape: learn multiple sizes independent of each other
    """

    def __init__(self, max_size, ramp_size, init_val=0, shape=(1,)):
        nn.Module.__init__(self)
        self._max_size = max_size
        self._ramp_size = ramp_size
        self.current_val = nn.Parameter(torch.zeros(*shape) + init_val)
        mask_template = torch.linspace(1 - max_size, 0, steps=max_size)
        self.register_buffer('mask_template', mask_template)

    def forward(self, x):
        mask = self.mask_template + self.current_val * self._max_size
        mask = mask / self._ramp_size + 1
        mask = mask.clamp(0, 1)
        if x.size(-1) < self._max_size:
            # the input could have been trimmed beforehand to save computation
            mask = mask[..., -x.size(-1):]
        x = x * mask
        return x

    def get_current_max_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.max().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def get_current_avg_size(self, include_ramp=True):
        current_size = math.ceil(self.current_val.mean().item() * self._max_size)
        if include_ramp:
            current_size += self._ramp_size
        current_size = max(0, min(self._max_size, current_size))
        return current_size

    def clamp_param(self):
        """this need to be called after each update"""
        self.current_val.data.clamp_(0, 1)


class AdaptiveSpan(nn.Module):
    """Adaptive attention span for Transformers.
    This module learns an attention span length from data for each
    self-attention head.

    Args:
        attn_span: maximum attention span
        adapt_span_loss: loss coefficient for the span length
        adapt_span_ramp: length of the masking ramp
        adapt_span_init: initial size ratio
        adapt_span_cache: adapt cache size to reduce memory usage
        nb_heads: number of attention heads
    """
    def __init__(self, attn_span, adapt_span_loss, adapt_span_ramp,
                 adapt_span_init, adapt_span_cache, nb_heads, **kargs):
        nn.Module.__init__(self)
        self._adapt_cache = adapt_span_cache
        self._max_span = attn_span
        self._loss_coeff = adapt_span_loss
        self._nb_heads = nb_heads
        self._mask = AdaptiveMask(max_size=self._max_span,
                                  ramp_size=adapt_span_ramp,
                                  init_val=adapt_span_init,
                                  shape=(nb_heads, 1, 1))

    def forward(self, attn, normalize=True):
        """mask attention with the right span"""
        # batch and head dimensions are merged together, so separate them first
        B, M, _ = attn.shape # batch size, sequence length
        attn = attn.reshape(B // self._nb_heads, self._nb_heads, M, -1)

        attn = self._mask(attn)
        if normalize:
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8) # normalize so sum is 1

        attn = attn.view(B, M, -1)
        return attn

    def get_trim_len(self):
        """how much of memory can be trimmed to reduce computation"""
        L = self._max_span
        trim_len = min(L - 1, L - self._mask.get_current_max_size())
        # too fine granularity might be bad for the memory management
        trim_len = math.floor(trim_len / 64) * 64
        return trim_len

    def trim_memory(self, query, key, value, key_pe=None):
        """trim out unnecessary memory beforehand to reduce computation"""
        trim_len = self.get_trim_len()
        if trim_len == 0:
            return key, value, key_pe

        cache_size = key.size(1) - query.size(1)
        trim_len_cache = trim_len - (self._max_span - cache_size)
        if trim_len_cache > 0:
            key = key[:, trim_len_cache:, :]
            value = value[:, trim_len_cache:, :]
        elif trim_len_cache < 0:
            # cache is too short! this happens when validation resumes
            # after a lot of updates.
            key = F.pad(key, [0, 0, -trim_len_cache, 0])
            value = F.pad(value, [0, 0, -trim_len_cache, 0])
        if trim_len > 0 and key_pe is not None:
                key_pe = key_pe[:, :, trim_len:]
        return key, value, key_pe

    def get_cache_size(self):
        """determine how long the cache should be"""
        if self._adapt_cache:
            trim_len = self.get_trim_len()
            # give a buffer of 64 steps since a span might increase
            # in future updates
            return min(self._max_span, self._max_span - trim_len + 64)
        else:
            return self._max_span

    def get_loss(self):
        """a loss term for regularizing the span length"""
        return self._loss_coeff * self._max_span * self._mask.current_val.mean()

    def get_current_max_span(self):
        return self._mask.get_current_max_size()

    def get_current_avg_span(self):
        return self._mask.get_current_avg_size()

    def clamp_param(self):
        self._mask.clamp_param()