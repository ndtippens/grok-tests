import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.positionals import RotaryPositionalEmbedding


def norm(x: nn.Linear):
    return F.rms_norm(x, (x.size(-1),))

class ResVAttention(nn.Module):
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
        self.lambdas = nn.Parameter(torch.nn.Linear([0.5, 0.5]))
        self.rotary = RotaryPositionalEmbedding(head_dim, max_seq_len)
        self.c_proj = BitLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self,
                x: nn.Linear,
                ve: nn.Linear | None = None,
                shared_values: nn.Linear | None = None,
                use_value_residual: bool = True):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        if shared_values is not None:
            v = shared_values
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        resv = v
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
        if use_value_residual:
            # Project residual values to match output dimension
            resv_proj = resv.contiguous().view(B, T, self.num_heads * self.head_dim)
            resv_proj = self.c_proj(resv_proj)
            y = y + resv_proj
        return y

