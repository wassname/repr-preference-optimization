from jaxtyping import Float
from typing import Any, Callable, Dict, Optional
from torch import Tensor
import torch
from dataclasses import dataclass, asdict
from ..types import HS2, ReprPOModelOutput
from ..reprpo.helpers import reduce_tokens_w_attention


def log_dist_ratio(
    a: HS2, b: HS2, a_ref: HS2, b_ref: HS2, eps=1e-12
) -> Float[Tensor, "b"]:
    """distance between a and b, as a log ratio to the distance between a_ref and b_ref"""

    dist = a - b
    dist = torch.norm(dist, dim=-1)  # over h
    dist = dist + eps
    log_dist_ratio = torch.log(dist)

    # if provided with reference points, return the distance as a ratio to the reference distance
    if (a_ref is not None) and (b_ref is not None):
        dist_ref = a_ref - b_ref
        dist_ref = torch.norm(dist_ref, dim=-1)
        dist_ref = dist_ref + eps

    # get the ratio in log space to avoid div by zero
    log_dist_ratio = torch.log(dist) - torch.log(dist_ref).detach()

    return log_dist_ratio


def mse_loss(
    pi_cho: ReprPOModelOutput,
    pi_rej: ReprPOModelOutput,
    ref_cho: ReprPOModelOutput,
    ref_rej: ReprPOModelOutput,
    batch: Dict[str, Any],
    transforms: Optional[Callable] = None,
    # custom loss_args
    α: float = 1,
    eps=1e-12,
):
    """
    movement of hs along the hs pref vector.
    """

    if transforms is not None:
        pi_cho.hs = transforms(pi_cho.hs)
        pi_rej.hs = transforms(pi_rej.hs)
        ref_cho.hs = transforms(ref_cho.hs)
        ref_rej.hs = transforms(ref_rej.hs)

    def preproc_hs(o, k):
        hs = reduce_tokens_w_attention(o.hs[k], o.mask)
        return hs

    def per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) -> Dict[str, Float[Tensor, "b"]]:
        hs_pi_cho = preproc_hs(pi_cho, k)
        hs_pi_rej = preproc_hs(pi_rej, k)
        hs_ref_cho = preproc_hs(ref_cho, k)  # .detach()
        hs_ref_rej = preproc_hs(ref_rej, k)  # .detach()

        # loss_reroute: the repr of policy rejected responses should be closer to the reference chosen responses
        # we measure it as a ratio to the distance between the chosen responses and the rejected responses in the reference model as this is a stable target
        loss_reroute = log_dist_ratio(
            hs_ref_cho,  # .detach(),
            hs_pi_rej,
            hs_ref_cho,
            hs_ref_rej,
            eps=eps,
        )

        # loss_retain: the repr of policy model chosen responses should be closer to the ref model chosen responses
        # we scale it using the reference model as a stable target
        loss_retain = log_dist_ratio(
            hs_ref_cho,  # .detach(),
            hs_pi_cho,
            hs_ref_cho,
            hs_ref_rej,
            eps=eps,
        )

        return dict(
            loss_retain=loss_retain,
            loss_reroute=loss_reroute,
        )

    # compute losses per layer
    ll = {k: per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) for k in pi_cho.hs.keys()}
    # combine layer losses
    ll_keys = next(iter(ll.values())).keys()
    ll = {k: torch.stack([v[k] for v in ll.values()], -1).mean(-1) for k in ll_keys}
    loss_reroute, loss_retain = ll["loss_reroute"], ll["loss_retain"]

    loss = (loss_reroute + loss_retain * α).nanmean() * 100 # multiply by 10 to make ideal lr around 1e-4

    # log info
    info = dict(
        loss_reroute=loss_reroute,
        loss_retain=loss_retain,
    )
    info = {k: v.mean().detach() for k, v in info.items()}

    return loss, info


@dataclass
class MSELossConfig:
    α: float = 0.6
    """weight between retain and reroute loss."""    
    eps: float = 1e-12

    def c(self, *args, **kwargs):
        return mse_loss(*args, **kwargs, **asdict(self))
