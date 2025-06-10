import torch
import torch.nn as nn
import math
from torch import Tensor

class AbsolutePositionalEmbedding(nn.Module):
    """
    Implements Absolute Positional Embeddings (APE) as described in the original Transformer paper.
    Each position in the sequence is assigned a unique, learnable embedding vector.

    Args:
        max_seq_len (int): The maximum sequence length.
        d_model (int): The dimensionality of the model's embeddings.
    """
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional embeddings to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(0, seq_len, device=x.device).expand(batch_size, -1)
        return x + self.embedding(positions)

class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embeddings (RoPE) which applies rotations to the query and key vectors.
    This method has been shown to be effective in capturing relative positional information.

    Args:
        d_model (int): The dimensionality of the model's embeddings.
        base (int, optional): The base for the sinusoidal functions. Defaults to 10000.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class AliBi(nn.Module):
    """
    Implements Attention with Linear Biases (ALiBi).
    Instead of adding positional embeddings, ALiBi adds a bias to the attention scores
    that is proportional to the distance between keys and queries.

    Args:
        num_heads (int): The number of attention heads.
    """
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.alibi_bias = None

    def _get_alibi_slopes(self, n_heads: int) -> torch.Tensor:
        n = 2 ** math.floor(math.log2(n_heads))
        m_0 = 2.0 ** (-8.0 / n)
        m = torch.pow(m_0, torch.arange(1, 1 + n))

        if n < n_heads:
            m_hat_0 = 2.0 ** (-4.0 / n)
            m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            m = torch.cat([m, m_hat])
        return m

    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Adds ALiBi biases to the attention scores.

        Args:
            attention_scores (torch.Tensor): The attention scores of shape (batch_size, num_heads, seq_len, seq_len).

        Returns:
            torch.Tensor: The attention scores with ALiBi biases added.
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape

        if self.alibi_bias is None or self.alibi_bias.shape[1] < seq_len:
            slopes = self._get_alibi_slopes(num_heads).to(attention_scores.device)
            self.alibi_bias = -torch.arange(seq_len, device=attention_scores.device).unsqueeze(0).unsqueeze(0) * slopes.unsqueeze(1).unsqueeze(1)
            self.alibi_bias = self.alibi_bias.expand(batch_size, -1, -1, -1)

        return attention_scores + self.alibi_bias[..., :seq_len, :seq_len]
