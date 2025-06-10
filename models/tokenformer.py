import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_circuit import Circuit, SaveInput, GetInput, StartBlock, EndBlock
from utils.BitLinear import BitLinear


class ParametricAttention(nn.Module):
    """Parametric attention mechanism for tokenformer."""

    def __init__(self, input_dim, output_dim, param_token_num=16, norm_type='gelu'):
        super().__init__()
        self.param_token_num = param_token_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm_type = norm_type

        # Use Linear layers instead of direct parameters
        self.key_projection = BitLinear(input_dim, param_token_num, bias=False)
        self.value_projection = BitLinear(param_token_num, output_dim, bias=False)
        
        # Initialize weights
        nn.init.xavier_normal_(self.key_projection.weight)
        nn.init.xavier_normal_(self.value_projection.weight)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim] or [batch_size, input_dim]
        """
        squeeze_output = False

        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension: [batch, 1, input_dim]
            squeeze_output = True

        # Compute attention weights using key projection
        attn_weights = self.key_projection(x)  # [batch, seq_len, param_token_num]

        # Apply normalization (simplified softmax)
        if self.norm_type == 'softmax':
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif self.norm_type == 'gelu':
            attn_weights = F.gelu(attn_weights)
            attn_weights = F.normalize(attn_weights, p=2, dim=-1) * math.sqrt(attn_weights.shape[-1])

        # Apply attention to values using value projection
        output = self.value_projection(attn_weights)  # [batch, seq_len, output_dim]

        # Maintain original sequence length
        if squeeze_output and output.shape[1] == 1:
            output = output.squeeze(1)  # Remove sequence dimension if it was added

        return output


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention block using ParametricAttention for Q, K, V generation."""

    def __init__(self, d_model, num_param_tokens=16, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        # Pattention for Q, K, V generation
        self.query_pattention = ParametricAttention(d_model, d_model, num_param_tokens)
        self.key_pattention = ParametricAttention(d_model, d_model, num_param_tokens)
        self.value_pattention = ParametricAttention(d_model, d_model, num_param_tokens)
                        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Generate Q, K, V using Pattention
        q = self.query_pattention(x)
        k = self.key_pattention(x)
        v = self.value_pattention(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use torch's scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        # Reshape back
        output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return output


class Tokenformer(nn.Module):
    """Tokenformer model following GPT-2 architecture with ParametricAttention."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 1024,
        num_param_tokens: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_param_tokens = num_param_tokens

        # Token and position embeddings (same as GPT-2)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Build the tokenformer using Circuit with repeatable transformer blocks
        self.transformer = Circuit(
            # StartRepeat for transformer blocks
            StartBlock("tokenformer_block", num_repeats=num_layers),

            # Save input for first residual connection (attention)
            SaveInput("attn_input"),

            # Pre-norm and tokenformer attention
            nn.LayerNorm(d_model),
            MultiHeadAttentionBlock(d_model, num_param_tokens, num_heads, dropout),
            ParametricAttention(d_model, d_model, num_param_tokens),
            nn.Dropout(dropout),

            # Add first residual connection
            GetInput("attn_input"),

            # Save for second residual connection (feed-forward)
            SaveInput("ff_input"),

            # Pre-norm and parametric feed-forward
            nn.LayerNorm(d_model),
            ParametricAttention(d_model, d_ff, num_param_tokens),
            nn.GELU(),  # GPT-2 uses GELU activation
            ParametricAttention(d_ff, d_model, num_param_tokens),
            nn.Dropout(dropout),

            # Add second residual connection
            GetInput("ff_input"),

            EndBlock("tokenformer_block"),

            # Final layer norm
            nn.LayerNorm(d_model)
        )

        # Output projection (same as GPT-2)
        self.lm_head = BitLinear(d_model, vocab_size, bias=False)

        # Tie weights between token embedding and lm_head (like GPT-2)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights (same as GPT-2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, BitLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through Tokenformer (same structure as GPT-2)."""
        batch_size, seq_len = input_ids.size()

        # Create position indices
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)

        # Combine embeddings
        x = self.dropout(token_embeds + position_embeds)

        # Pass through transformer blocks
        x = self.transformer(x)

        # Output projection
        logits = self.lm_head(x)

        return logits
