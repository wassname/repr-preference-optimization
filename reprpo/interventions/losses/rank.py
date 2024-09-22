from jaxtyping import Float, Int
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F
from dataclasses import dataclass, asdict
from .helpers import cross_entropy_loss, compute_ptheta
from ..types import HS, Mask, ReprPOModelOutput
from ..reprpo.helpers import mean_tokens_w_attention


def rank_loss(
    pi_cho: ReprPOModelOutput,
    pi_rej: ReprPOModelOutput,
    ref_cho: ReprPOModelOutput,
    ref_rej: ReprPOModelOutput,
    batch: Dict[str, Any],
    transform: Optional[Callable] = None,
    # custom loss_args
    alpha: float = 1,
):
    """
    This loss treats the hidden states like probabilities by taking the softmax. Despite the fact that they are not used as probabilities, this lets us modify the relative ranking as if they are.

    Then we make a loss of the log ratios (ptheta), making the chosen hs more likely, and the rejected less

    To avoid this taking us into a degenerate solution, we also try various retain losses
    - DPO to ensure that the chosen is more likely than rejected
    - nll to make sure the chosen is at least as likely
    """

    def preproc_hs(o, k):
        hs = o.hs[k]
        if transform is not None:
            hs = transform(hs)
        hs = hs.log_softmax(-1)
        hs = mean_tokens_w_attention(hs, o.mask)
        return hs

    def per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) -> Dict[str, Float[Tensor, 'b']]:
        β = 100
        ptheta_left = preproc_hs(pi_rej, k) - preproc_hs(ref_rej, k)
        ptheta_right = preproc_hs(pi_cho, k) - preproc_hs(ref_cho, k)

        ptheta = ptheta_right - ptheta_left
        # OR?
        # ptheta = - ptheta_left

        loss_reroute = (β * ptheta - 1) ** 2  # as in IPO

        # loss_retain = (β * ptheta_right)**2 # make sure chosen ratio stays the same... but this woould limit us
        return dict(loss_reroute=loss_reroute.mean(1))

    # compute losses per layer
    ll = {k: per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) for k in pi_cho.hs.keys()}
    # combine layer losses
    ll_keys = next(iter(ll.values())).keys()
    ll = {k: torch.stack([v[k] for v in ll.values()], -1).mean(-1) for k in ll_keys}
    loss_reroute = ll["loss_reroute"]

    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"], batch["chosen_mask"])
    ref_nll_loss = cross_entropy_loss(
        ref_cho.logits, batch["chosen"], batch["chosen_mask"]
    )
    nll_loss_ratio = nll_loss - ref_nll_loss
    dpo_ptheta = compute_ptheta(
        pi_cho.label_logprobs,
        pi_rej.label_logprobs,
        ref_cho.label_logprobs,
        ref_rej.label_logprobs,
    )
    loss_dpo_retain = F.relu(-dpo_ptheta)
    loss_nll_retain = F.relu(nll_loss_ratio)
    loss_retain = loss_nll_retain.mean(1) + loss_dpo_retain

    loss = loss_reroute.mean() + alpha * loss_retain.mean()

    info = dict(
        loss_reroute=loss_reroute,
        loss_retain=loss_retain,
        nll_loss=nll_loss,
        ref_nll_loss=ref_nll_loss,
        nll_loss_ratio=nll_loss_ratio,
        dpo_ptheta=dpo_ptheta,
    )
    info = {k: v.mean().detach() for k, v in info.items()}
    return loss, info


@dataclass(frozen=True)
class RankLossConfig:
    alpha: float = 1.0
    # eps: float = 1e-12

    def c(self, *args, **kwargs):
        return rank_loss(*args, **kwargs, **asdict(self))
