from torch import nn, Tensor
import torch
import torch.nn.functional as F


class SimpleRMSNorm(nn.Module):
    """Simple RMS normalization layer"""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize
        x_norm = x / rms
        # Scale
        return x_norm * self.scale


def activation_quant(x: Tensor):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): Input tensor to quantize

    Returns:
        Tensor: Quantized tensor
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: Tensor):
    """Quantize weights to binary values

    Args:
        w (Tensor): Weight tensor to quantize

    Returns:
        Tensor: Quantized weight tensor
    """
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u


class BitLinear(nn.Linear):
    """
    Custom linear layer with bit quantization.

    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        bias (bool, optional): If set to False, the layer will not learn an additive bias. 
                              Default: True
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    def __init__(self, in_features, out_features, bias=True, *args, **kwargs):
        super().__init__(in_features, out_features, bias, *args, **kwargs)
        self.norm = SimpleRMSNorm(in_features)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        w = self.weight
        x_norm = self.norm(x)

        # STE (Straight-Through Estimator) using detach
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        
        # Apply quantized linear transformation
        y = F.linear(x_quant, w_quant, self.bias)
        return y
