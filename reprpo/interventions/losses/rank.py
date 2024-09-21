
from jaxtyping import Float, Int
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch import Tensor
from torch.nn import functional as F
from dataclasses import dataclass, asdict
from .helpers import cross_entropy_loss, compute_ptheta
from ..types import HS, Mask, ReprPOModelOutput
from ..reprpo.helpers import mean_tokens_w_attention



def rank_loss(pi_cho: ReprPOModelOutput,
            pi_rej: ReprPOModelOutput, 
            ref_cho: ReprPOModelOutput, 
            ref_rej: ReprPOModelOutput, 
            batch: Dict[str, Any],
            transform: Optional[Callable] = None,
            # custom loss_args
            alpha: Float = 1,
            ):
    """
    This loss treats the hidden states like probabilities by taking the softmax. Despite the fact that they are not used as probabilities, this lets us modify the relative ranking as if they are.

    Then we make a loss of the log ratios (ptheta), making the chosen hs more likely, and the rejected less

    To avoid this taking us into a degenerate solution, we also try various retain losses
    - DPO to ensure that the chosen is more likely than rejected
    - nll to make sure the chosen is at least as likely
    """

    def preproc_hs(o):
        if transform is not None:
            hs = transform(o.hs)
        hs = hs.log_softmax(-1)
        hs = mean_tokens_w_attention(hs, o.mask)
        return hs
    
    β = 100
    ptheta_left = preproc_hs(pi_rej)  - preproc_hs(ref_rej)
    ptheta_right = preproc_hs(pi_cho) - preproc_hs(ref_cho)


    ptheta = ptheta_right - ptheta_left
    # OR?
    # ptheta = - ptheta_left

    loss_reroute = (β*ptheta - 1) ** 2 # as in IPO

    # loss_retain = (β * ptheta_right)**2 # make sure chosen ratio stays the same... but this woould limit us

    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"], batch['chosen_mask'])
    ref_nll_loss = cross_entropy_loss(ref_cho.logits, batch["chosen"], batch['chosen_mask'])
    nll_loss_ratio = nll_loss - ref_nll_loss
    dpo_ptheta = compute_ptheta(
        pi_cho.logprobs,
        pi_rej.logprobs,
        ref_cho.logprobs,
        ref_rej.logprobs,
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
    alpha: Float = 1
    eps: Float = 1e-12

    @property
    def c(self):
        return rank_loss(**asdict(self))
