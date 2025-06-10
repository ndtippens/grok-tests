import torch
import torch.nn as nn

# Import Circuit and attention layers
from torch_circuit import Circuit, SaveInput, GetInput, StartBlock, EndBlock
from layers.attention import CausalSelfAttention
from layers.positionals import RotaryPositionalEmbedding

class GPT2Model(nn.Module):
    """Simple GPT-2 model implementation using Circuit with repeatable blocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        #self.position_embedding = RotaryPositionalEmbedding(d_model)
        # RoPE will be applied in attention layers instead
        self.dropout = nn.Dropout(dropout)

        # Build the entire GPT-2 model using Circuit with repeatable transformer blocks
        self.transformer = Circuit(
            # StartRepeat for transformer blocks
            StartBlock("transformer_block", num_repeats=num_layers),

            # Save input for first residual connection (attention)
            SaveInput("attn_input"),

            # Pre-norm and causal attention
            nn.LayerNorm(d_model),
            CausalSelfAttention(d_model, num_heads, max_seq_len, head_dim=max_seq_len//num_heads),
            nn.Dropout(dropout),

            # Add first residual connection
            GetInput("attn_input"),

            # Save for second residual connection (feed-forward)
            SaveInput("ff_input"),

            # Pre-norm and feed-forward
            nn.LayerNorm(d_model),

            # MLP layers directly in the Circuit
            nn.Linear(d_model, d_ff, bias=True),  # c_fc
            nn.GELU(),  # GPT-2 uses GELU activation
            nn.Linear(d_ff, d_model, bias=True),  # c_proj
            nn.Dropout(dropout),

            # Add second residual connection
            GetInput("ff_input"),

            EndBlock("transformer_block"),

            # Final layer norm
            nn.LayerNorm(d_model)
        )

        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights between token embedding and lm_head (like GPT-2)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through GPT-2."""
        batch_size, seq_len = input_ids.size()

        x = self.token_embedding(input_ids)

        # Pass through transformer blocks
        x = self.transformer(x)

        # Output projection
        logits = self.lm_head(x)

        return logits
