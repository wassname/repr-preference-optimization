from jaxtyping import Float
from typing import Any, Callable, Dict, Optional
import torch
from torch import Tensor
from torch.nn import functional as F
from dataclasses import dataclass, asdict
from ..dpo_helpers import cross_entropy_loss, compute_ptheta
from ..types import ReprPOModelOutput
from ..reprpo.helpers import reduce_tokens_w_attention


def rank_loss(
    pi_cho: ReprPOModelOutput,
    pi_rej: ReprPOModelOutput,
    ref_cho: ReprPOModelOutput,
    ref_rej: ReprPOModelOutput,
    batch: Dict[str, Any],
    transforms: Optional[Callable] = None,
    # custom loss_args
    α: float = 1,
    β: float = 100,
    use_dpo_loss=True,
    use_nll_loss=False,
    use_rank_retain=False,
):
    """
    This loss treats the hidden states like probabilities by taking the softmax. Despite the fact that they are not used as probabilities, this lets us modify the relative ranking as if they are.

    Then we make a loss of the log ratios (ptheta), making the chosen hs more likely, and the rejected less

    To avoid this taking us into a degenerate solution, we also try various retain losses
    - DPO to ensure that the chosen is more likely than rejected
    - nll to make sure the chosen is at least as likely
    """

    if transforms is not None:
        pi_cho.hs = transforms(pi_cho.hs)
        pi_rej.hs = transforms(pi_rej.hs)
        ref_cho.hs = transforms(ref_cho.hs)
        ref_rej.hs = transforms(ref_rej.hs)

    def preproc_hs(o, k: str):
        hs = o.hs[k].log_softmax(-1)
        hs = reduce_tokens_w_attention(hs, o.mask)
        return hs

    def per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) -> Dict[str, Float[Tensor, "b"]]:
        
        # measure has the psudo-prob distribution improved over the reference model
        ptheta_rej = preproc_hs(pi_rej, k) - preproc_hs(ref_rej, k)
        ptheta_cho = preproc_hs(pi_cho, k) - preproc_hs(ref_cho, k)

        # has the chosen response become higher magnitude (rel to other activations, rel to base model) than the rejected response
        ptheta = ptheta_cho - ptheta_rej
        # OR?
        # ptheta = - ptheta_rej

        loss_reroute = (β * ptheta - 1) ** 2  # as in IPO, values above and below beta are punished

        loss_retain = (β * ptheta_cho)**2 # make sure chosen ratio stays the same... but this would limit us

        loss_reroute = - torch.log(torch.sigmoid(-β * ptheta)) # as in DPO
        return dict(
            loss_reroute=loss_reroute.mean(1), 
            loss_rank_retain=loss_retain.mean(1),
            _ptheta=ptheta.mean(1), _ptheta_cho=ptheta_cho.mean(1), _ptheta_rej=ptheta_rej.mean(1))

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
    nll_loss_ratio = (nll_loss - ref_nll_loss).mean(1)
    dpo_ptheta = compute_ptheta(
        pi_cho.label_logprobs,
        pi_rej.label_logprobs,
        ref_cho.label_logprobs,
        ref_rej.label_logprobs,
    )

    # this loss is saying: make the hs assocated with the chosen response higher in value than the rejected response, while keeping the dpo (prefer chosen) and nll loss (coherent) the same or better
    # only punish when the dpo/preference of the nll/coherency is worse than the base model
    loss_retain_dpo = F.relu(-dpo_ptheta)
    loss_retain_nll = F.relu(nll_loss_ratio)
    loss_retain = torch.zeros_like(loss_reroute)
    if use_dpo_loss:
        loss_retain += loss_retain_dpo
    if use_nll_loss:
        loss_retain += loss_retain_nll * 100 # HACKY balance
    if use_rank_retain:
        loss_retain += ll["loss_rank_retain"]

    loss = loss_reroute.mean() + α * loss_retain.mean()
    # TODO rebalance

    info = dict(
        # loss_reroute=loss_reroute,
        loss_retain=loss_retain,
        _nll_loss=nll_loss,
        _ref_nll_loss=ref_nll_loss,
        _nll_loss_ratio=nll_loss_ratio,
        loss_nll_retain=loss_retain_nll,
        loss_dpo_retain=loss_retain_dpo,
        _dpo_ptheta=dpo_ptheta,
        **ll,
    )
    info = {k: v.mean().detach() for k, v in info.items()}
    return loss, info


@dataclass
class RankLossConfig:
    α: float = 0.25
    """weight between retain and reroute loss."""    

    β: float = 1.
    """like dpo regularization coefficient beta."""

    use_dpo_loss: bool = True
    """punish model if rejected completion is more likely than the chosen"""

    use_nll_loss: bool = False
    """punish model if output is less coherent than reference model"""

    use_rank_retain: bool = False
    """keep the activations of the chosen response the same while making the rejected lower."""

    def c(self, *args, **kwargs):
        return rank_loss(*args, **kwargs, **asdict(self))
