import torch
import torch.nn as nn
import torch.nn.functional as F
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