import torch
import torch.nn as nn

# Import Circuit and attention layers
from torch_circuit import Circuit, SaveInput, GetInput, StartBlock, EndBlock
from layers.resv_attention import ResVAttention


class GPT2ResVModel(nn.Module):
    """
    ResFormer/SVFormer implementation of GPT-2 using Circuit with repeatable blocks.
    
    Enhances information flow by incorporating value residual connections
    in addition to hidden state residuals. SVFormer variant shares the first
    layer's value embedding across all layers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        share_values: bool = False  # SVFormer when True, ResFormer when False
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.share_values = share_values
        self.num_heads = num_heads
        self.dropout_rate = dropout
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # For SVFormer: create a shared value projection
        if share_values:
            self.shared_value_proj = nn.Linear(d_model, d_model, bias=False)
            nn.init.normal_(self.shared_value_proj.weight, mean=0.0, std=0.02)
        
        # Build the transformer circuit
        self.transformer = Circuit(
            # Process input embeddings
            self.token_embedding,

            # Transformer blocks
            StartBlock("transformer_block", num_repeats=num_layers),

            # Attention block with residual
            SaveInput("attn_residual"),
            nn.LayerNorm(d_model),
            ResVAttention(d_model, self.num_heads, self.max_seq_len),
            GetInput("attn_residual", op=torch.add),
            nn.Dropout(dropout),

            # Feed-forward block with residual
            SaveInput("ff_residual"),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            GetInput("ff_residual", op=torch.add),
            nn.Dropout(dropout),

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
        """Forward pass through ResFormer/SVFormer."""
        # Pass through transformer circuit
        x = self.transformer(input_ids)

        # Apply language model head
        return self.lm_head(x)