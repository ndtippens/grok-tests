import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import Transformer
from utils.utils import causal_attn_mask, parameter_norm

class GrokkModel(nn.Module):
    def __init__(self, transformer_config, vocab_size, output_size, device):
        super(GrokkModel, self).__init__()
        # Remove 'l1_weight' if present in transformer_config
        transformer_config = dict(transformer_config)
        self.transformer = Transformer(**transformer_config, vocab_size=vocab_size, output_size=output_size)
        self.device = device
    
    def forward(self, x):
        #attn_mask = causal_attn_mask(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1, 1).to(self.device)
        predictions, _ = self.transformer(x)
        return predictions
