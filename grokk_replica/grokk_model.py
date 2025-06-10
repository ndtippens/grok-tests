import torch
import torch.nn as nn
import torch.nn.functional as F
from grokk_replica.transformer import Transformer
from grokk_replica.utils import causal_attn_mask, parameter_norm

class GrokkModel(nn.Module):
    def __init__(self, transformer_config, vocab_size, output_size, device, l1_weight=0.0):
        super(GrokkModel, self).__init__()
        # Remove 'l1_weight' if present in transformer_config
        transformer_config = dict(transformer_config)
        transformer_config.pop('l1_weight', None)
        self.transformer = Transformer(**transformer_config, vocab_size=vocab_size, output_size=output_size)
        self.device = device
        self.l1_weight = l1_weight
    
    def forward(self, x):
        attn_mask = causal_attn_mask(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1, 1).to(self.device)
        predictions, _ = self.transformer(x, attn_mask)
        return predictions
