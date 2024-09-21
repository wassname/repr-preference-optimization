
from jaxtyping import Float, Int
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch import Tensor
from torch.nn import functional as F
import torch
from dataclasses import dataclass, asdict
from .helpers import cross_entropy_loss
from ..types import HS, Mask, ReprPOModelOutput
from ..reprpo.helpers import mean_tokens_w_attention, detach_hsd


def log_dist_ratio(a, b, a_ref, b_ref, eps=1e-12) -> Float[Tensor, "b l"]:
    """distance between a and b, as a log ratio to the distance between a_ref and b_ref"""

    dist = a-b
    dist = torch.norm(dist, dim=-1) # over h
    dist = dist + eps
    log_dist_ratio = torch.log(dist)

    # if provided with reference points, return the distance as a ratio to the reference distance
    if (a_ref is not None) and (b_ref is not None):
        dist_ref = (a_ref-b_ref).detach()
        dist_ref = torch.norm(dist_ref, dim=-1)
        dist_ref = dist_ref + eps

    # get the ratio in log space to avoid div by zero
    log_dist_ratio = torch.log(dist) - torch.log(dist_ref).detach()

    return log_dist_ratio


def mse_loss(pi_cho: ReprPOModelOutput,
            pi_rej: ReprPOModelOutput, 
            ref_cho: ReprPOModelOutput, 
            ref_rej: ReprPOModelOutput, 
            batch: Dict[str, Any],
            transform: Optional[Callable] = None,
            # custom loss_args
            alpha: Float = 1,
            eps=1e-12,
            ):
    """
    movement of hs along the hs pref vector.
    """


    def preproc_hs(o):
        if transform is not None:
            hs = transform(o.hs)
        hs = hs.log_softmax(-1)
        hs = mean_tokens_w_attention(hs, o.mask)
        return hs

    hs_pi_cho = preproc_hs(pi_cho)
    hs_pi_rej = preproc_hs(pi_rej)
    hs_ref_cho = preproc_hs(ref_cho)#.detach()
    hs_ref_rej = preproc_hs(ref_rej)#.detach()

    # loss_retain: the representation of policy chosen responses should be closer to the reference chosen responses
    # and again we scale it using the reference model as a stable target
    loss_retain = log_dist_ratio(
        detach_hsd(hs_ref_cho),
        hs_pi_cho,
        hs_ref_cho,
        hs_ref_rej,
    )

    # loss_reroute: the representation of policy rejected responses should be closer to the reference chosen responses
    # we measure it as a ratio to the distance between the chosen responses and the rejected responses in the reference model as this is a stable target
    loss_reroute = log_dist_ratio(
        detach_hsd(hs_ref_cho),
        hs_pi_rej,
        hs_ref_cho,
        hs_ref_rej,
    )
    loss = (loss_reroute + loss_retain * alpha).nanmean()

    info = dict(
        loss_reroute=loss_reroute,
        loss_retain=loss_retain,
    )
    info = {k:v.mean().detach() for k,v in info.items()}

    return loss, info


@dataclass(frozen=True)
class MSELossConfig:
    alpha: Float = 1.
    eps: Float = 1e-12

    @property
    def c(self):
        return log_dist_ratio(**asdict(self))
