import torch
import torch.nn as nn
import torch.nn.functional as F

# Import existing components
from torch_circuit import Circuit, SaveInput, GetInput, StartBlock, EndBlock
from layers.attention import CausalMultiHeadAttention
from layers.positionals import RotaryPositionalEmbedding


class CrossAttention(nn.Module):
    """Cross-attention layer for meta-learning between frames."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 1024):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

        # Query from x, Key and Value from w
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Causal mask for cross-attention
        self.register_buffer('causal_mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Query tensor [batch_size, seq_len, d_model]
            w: Key/Value tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()

        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(w).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(w).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Use torch's scaled_dot_product_attention with causal masking
        attention_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=self.causal_mask[:seq_len, :seq_len],
            dropout_p=self.dropout.p if self.training else 0.0
        )

        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)

        return self.w_o(attention_output)

class MetaTransformerBlock(nn.Module):
    """Meta-learning transformer block with cross-attention between frames."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_frames: int, dropout: float = 0.1, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.num_frames = num_frames

        # Main sequence self-attention and feed-forward
        self.main_attn = CausalMultiHeadAttention(d_model, num_heads, dropout, max_seq_len)
        self.main_ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout)
        )

        # Frame-specific layers (if we have frames)
        if num_frames > 0:
            # Self-attention for each frame
            self.frame_attn = nn.ModuleList([
                CausalMultiHeadAttention(d_model, num_heads, dropout, max_seq_len)
                for _ in range(num_frames)
            ])

            # Feed-forward for each frame
            self.frame_ff = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_ff, bias=True),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model, bias=True),
                    nn.Dropout(dropout)
                ) for _ in range(num_frames)
            ])

            # Cross-attention: main sequence attends to frames
            self.cross_attn_main = nn.ModuleList([
                CrossAttention(d_model, num_heads, dropout, max_seq_len)
                for _ in range(num_frames)
            ])

            # Cross-attention: frames attend to main sequence
            self.cross_attn_frame = nn.ModuleList([
                CrossAttention(d_model, num_heads, dropout, max_seq_len)
                for _ in range(num_frames)
            ])

        # Layer norms
        self.ln_main_1 = nn.LayerNorm(d_model)
        self.ln_main_2 = nn.LayerNorm(d_model)

        if num_frames > 0:
            self.ln_frame_1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_frames)])
            self.ln_frame_2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_frames)])
            self.ln_cross_1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_frames)])
            self.ln_cross_2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_frames)])

    def forward(self, x: torch.Tensor, frames: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: Main sequence [batch_size, seq_len, d_model]
            frames: Frame sequences [num_frames, batch_size, seq_len, d_model] or None
        """
        # Main sequence self-attention and feed-forward
        x = x + self.main_attn(self.ln_main_1(x))
        x = x + self.main_ff(self.ln_main_2(x))

        if frames is not None and self.num_frames > 0:
            # Process each frame with self-attention and feed-forward
            processed_frames = []
            for i in range(self.num_frames):
                frame = frames[i]
                frame = frame + self.frame_attn[i](self.ln_frame_1[i](frame))
                frame = frame + self.frame_ff[i](self.ln_frame_2[i](frame))
                processed_frames.append(frame)
            frames = torch.stack(processed_frames, dim=0)

            # Cross-attention between main sequence and frames
            cross_outputs = []
            for i in range(self.num_frames):
                # Main sequence attends to frame
                x_cross = self.cross_attn_main[i](self.ln_cross_1[i](x), frames[i])
                x = x + x_cross

                # Frame attends to main sequence
                frame_cross = self.cross_attn_frame[i](self.ln_cross_2[i](frames[i]), x)
                cross_outputs.append(frames[i] + frame_cross)

            frames = torch.stack(cross_outputs, dim=0)

        return x, frames

class GPT2MetaModel(nn.Module):
    """GPT-2 Meta-learning model compatible with train.py."""

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

        # For now, we'll create a simplified version without frames for train.py compatibility
        # Main sequence embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        #self.position_embedding = RotaryPositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

        # Simplified transformer using Circuit (no frames for now)
        self.transformer = Circuit(
            StartBlock("meta_transformer_block", num_repeats=num_layers),

            # Save input for first residual connection (attention)
            SaveInput("attn_input"),

            # Pre-norm and causal attention
            nn.LayerNorm(d_model),
            CausalMultiHeadAttention(d_model, num_heads, dropout, max_seq_len),
            nn.Dropout(dropout),

            # Add first residual connection
            GetInput("attn_input"),

            # Save for second residual connection (feed-forward)
            SaveInput("ff_input"),

            # Pre-norm and feed-forward
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=True),
            nn.Dropout(dropout),

            # Add second residual connection
            GetInput("ff_input"),

            EndBlock("meta_transformer_block"),

            # Final layer norm
            nn.LayerNorm(d_model)
        )

        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, (nn.Linear, nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass compatible with train.py."""
        batch_size, seq_len = input_ids.size()

        # Create position indices
        #position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        #position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        #position_embeds = self.position_embedding(position_ids)

        # Combine embeddings
        x = self.dropout(token_embeds)

        # Pass through transformer blocks
        x = self.transformer(x)

        # Output projection
        logits = self.lm_head(x)

        return logits