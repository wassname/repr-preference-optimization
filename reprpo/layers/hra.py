import torch
from torch import nn
import math

class HRATransform(nn.Module):
    """
    see
    - https://github.com/huggingface/peft/blob/54be5a3db61748d698ca2e6b55bcfef229a9b475/src/peft/tuners/hra/layer.py#L197
    """

    def __init__(self, in_features, out_features, 
                 r=8, apply_GS=False):
        super().__init__()
        

        self.hra_r = r
        self.apply_GS = apply_GS
        self.hra_u = nn.Parameter(torch.randn(in_features, r))

        self.reset_hra_parameters()

    def __repr__(self):
        return f"HRATransform(in_features={self.hra_u.shape[0]}, out_features={self.hra_u.shape[1]}, r={self.hra_r}, apply_GS={self.apply_GS})"

    def reset_hra_parameters(self):
        if self.hra_r % 2 != 0:
            warnings.warn("The symmetric initialization can NOT be performed when r is odd!")
            nn.init.kaiming_uniform_(self.hra_u, a=math.sqrt(5))
        else:
            shape = self.hra_u.shape
            half_u = torch.zeros(shape[0], shape[1] // 2)
            nn.init.kaiming_uniform_(half_u, a=math.sqrt(5))
            self.hra_u = nn.Parameter(torch.repeat_interleave(half_u, 2, dim=1))

    def get_delta_weight(self, reverse: bool = False) -> torch.Tensor:
        rank = self.hra_r
        apply_GS = self.apply_GS
        opt_u = self.hra_u
        shape = opt_u.shape

        # def norm_w_eps(a, eps=1e-5):
        #     """if we want to contrain it by lambda like in the paper
        #     https://github.com/DaShenZi721/HRA/blob/master/llama/peft/oft/layer.py#L249C51-L249C115
        #     """
        #     return torch.sqrt(torch.sum(a ** 2) + eps)

        if apply_GS:
            weight = [(opt_u[:, 0] / opt_u[:, 0].norm()).view(-1, 1)]
            for i in range(1, rank):
                ui = opt_u[:, i].view(-1, 1)
                for j in range(i):
                    ui = ui - (weight[j].t() @ ui) * weight[j]
                weight.append((ui / ui.norm()).view(-1, 1))
            weight = torch.cat(weight, dim=1)
            weight = torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype) - 2 * weight @ weight.t()

        else:
            opt_u = opt_u / opt_u.norm(dim=0)
            weight = torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype)
            if reverse:
                indices = range(rank - 1, -1, -1)
            else:
                indices = range(rank)

            for i in indices:
                ui = opt_u[:, i].view(-1, 1)
                weight = weight @ (torch.eye(shape[0], device=opt_u.device, dtype=opt_u.dtype) - 2 * ui @ ui.t())

        return weight
    
    def forward(self, input):
        delta_weight = self.get_delta_weight()
        return torch.matmul(input, delta_weight)

