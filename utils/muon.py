import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.BitLinear import BitLinear
from torch.types import Tensor
import warnings
#import jax.numpy as jnp

@torch.compile
def orthogonalize(M):
   # by @YouJiacheng (with stability loss idea from @leloykun)

   abc_list = [
      (3955/1024, -8306/1024, 5008/1024),
      (3735/1024, -6681/1024, 3463/1024),
      (3799/1024, -6499/1024, 3211/1024),
      (4019/1024, -6385/1024, 2906/1024),
      (2677/1024, -3029/1024, 1162/1024),
      (2172/1024, -1833/1024,  682/1024)
   ]

   transpose = M.shape[1] > M.shape[0]
   if transpose:
      M = M.T
   M /= M.norm()
   for a, b, c in abc_list:
      A = M.T @ M
      I = torch.eye(A.shape[0])
      M = M @ (a * I + b * A + c * A @ A)
   if transpose:
      M = M.T
   return M


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=3)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="mps")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = orthogonalize(g).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()


class CayleyLinear(nn.Module):
    """
    This implementation caches the orthogonal weight matrix during evaluation (`.eval()` mode)
    to avoid redundant and expensive re-computation. During training (`.train()` mode),
    it computes the weight dynamically on each forward pass to ensure correct
    gradient flow to the underlying parameters.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if in_features != out_features:
            raise ValueError(f"CayleyLinear requires in_features == out_features. "
                             f"Got {in_features} and {out_features}.")

        # Use a linear layer to hold the parameters for the skew-symmetric matrix 'A'
        self.A_layer = BitLinear(in_features, out_features, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Register a buffer for the cached orthogonal weight matrix.
        # Buffers are part of the model's state_dict but are not considered parameters.
        self.register_buffer('cached_Q', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_layer.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.A_layer.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # Invalidate the cache after reset
        self.cached_Q = None

    def _get_orthogonal_weight(self) -> torch.Tensor:
        A = self.A_layer.weight
        A_skew = 0.5 * (A - A.transpose(0, 1))
        I = torch.eye(self.out_features, device=A.device, dtype=A.dtype)
        
        try:
            Q = (I - A_skew) @ torch.inverse(I + A_skew)
        except torch.linalg.LinAlgError:
            warnings.warn("Singular matrix in Cayley transform. Using pseudo-inverse.")
            Q = (I - A_skew) @ torch.linalg.pinv(I + A_skew)
        return Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If in training mode, always recompute Q to ensure gradients are tracked.
        if self.training:
            Q = self._get_orthogonal_weight()
        # If in evaluation mode, use the cached Q. If the cache is empty, compute and store it.
        else:
            if self.cached_Q is None:
                # Use torch.no_grad() to avoid tracking history during this one-time computation
                with torch.no_grad():
                    self.cached_Q = self._get_orthogonal_weight()
            Q = self.cached_Q
            
        return F.linear(x, Q, self.bias)

    # Override the train() and eval() methods to manage the cache
    def train(self, mode: bool = True):
        """
        Sets the module in training mode.
        This will invalidate the cache, forcing re-computation of the orthogonal matrix.
        """
        super().train(mode)
        if mode:
            # Invalidate cache when switching to training mode
            self.cached_Q = None
        return self

    def eval(self):
        """
        Sets the module in evaluation mode.
        The orthogonal matrix will be computed and cached on the first forward pass.
        """
        return self.train(False)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
