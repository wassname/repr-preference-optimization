"""

Here we define a transform to isolate supressed activations, where we hypothesis that style/concepts/scratchpads and other internal only representations must be stored.

See the following references for more information:
- https://arxiv.org/html/2406.19384v1
    - > Previous work suggests that networks contain ensembles of “prediction" neurons, which act as probability promoters [66, 24, 32] and work in tandem with suppression neurons (Section 5.4). 

- https://arxiv.org/pdf/2401.12181
    > We find a striking pattern which is remarkably consistent across the different seeds: after about the halfway point in the model, prediction neurons become increasingly prevalent until the very end of the network where there is a sudden shift towards a much larger number of suppression neurons.
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from einops import rearrange
from torch import nn
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict
from .helpers import TransformByLayer
from ..types import HS


def get_supressed_activations(
    hs: Float[Tensor, "l b t h"], w_out, w_inv
) -> Float[Tensor, "l b t h"]:
    """
    Novel experiment: Here we define a transform to isolate supressed activations, where we hypothesis that style/concepts/scratchpads and other internal only representations must be stored.

    See the following references for more information:

    - https://arxiv.org/pdf/2401.12181
        - > Suppression neurons that are similar, except decrease the probability of a group of related tokens

    - https://arxiv.org/html/2406.19384
        - > Previous work suggests that networks contain ensembles of “prediction" neurons, which act as probability promoters [66, 24, 32] and work in tandem with suppression neurons (Section 5.4).

    - https://arxiv.org/pdf/2401.12181
        > We find a striking pattern which is remarkably consistent across the different seeds: after about the halfway point in the model, prediction neurons become increasingly prevalent until the very end of the network where there is a sudden shift towards a much larger number of suppression neurons.
    """ 
    with torch.no_grad():
        # here we pass the hs through the last layer, take a diff, and then project it back to find which activation changes lead to supressed
        hs2 = rearrange(hs[:, :, -1:], "l b t h -> (l b t) h")
        hs_out2 = torch.nn.functional.linear(hs2, w_out)
        hs_out = rearrange(
            hs_out2, "(l b t) h -> l b t h", l=hs.shape[0], b=hs.shape[1], t=1
        )
        diffs = hs_out[:, :, :].diff(dim=0)
        diffs_flat = rearrange(diffs, "l b t h -> (l b t) h")
        # W_inv = get_cache_inv(w_out)

        diffs_inv_flat = torch.nn.functional.linear(diffs_flat.to(dtype=w_inv.dtype), w_inv)
        diffs_inv = rearrange(
            diffs_inv_flat, "(l b t) h -> l b t h", l=hs.shape[0] - 1, b=hs.shape[1], t=1
        ).to(w_out.dtype)
        # TODO just return this?
        eps = 1.0e-1
        supressed_mask = (diffs_inv > eps).to(hs.dtype)
        # supressed_mask = repeat(supressed_mask, 'l b 1 h -> l b t h', t=hs.shape[2])
    supressed_act = hs[1:] * supressed_mask
    return supressed_act



class SupressedHSTransform(nn.Module):
    def __init__(self, dim_sizes: Dict[str, int], model: nn.Module, **kwargs):
        super().__init__()
        self.Wo = model.get_output_embeddings().weight.detach().clone().cuda()
        self.Wo_inv = torch.pinverse(self.Wo.clone().float()).cuda()

    def forward(self, x: Dict[str, HS]) -> Dict[str, HS]:
        keys = sorted([k for k in x.keys()], key=lambda x: int(x))   
        hs = torch.stack([x[k] for k in keys], dim=0) #l b t h
        hs = get_supressed_activations(
            hs=hs,
            w_out=self.Wo,
            w_inv=self.Wo_inv,
        )
        return {k: hs[i] for i, k in enumerate(keys[1:])}

# class EtherTransforms(TransformByLayer):
#     Transform = SupressedHSTransform


@dataclass
class SupressedHSConfig:
    """
    Get only the activations that are supressed between layers. These are the ones values that are not used for prediction but may be used internally.
    """

    def c(
        self,
        *args,
        **kwargs,
    ):
        return SupressedHSTransform(*args, **kwargs, **asdict(self))
