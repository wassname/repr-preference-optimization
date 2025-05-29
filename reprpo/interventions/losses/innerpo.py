from jaxtyping import Float
from typing import Any, Callable, Dict, Optional
from torch import Tensor
from torch.nn import functional as F
import torch
from dataclasses import dataclass, asdict

from ..dpo_helpers import cross_entropy_loss, compute_ptheta
from ..types import ReprPOModelOutput
from ..reprpo.helpers import reduce_tokens_w_attention

def safe_signed_log(x: Tensor, eps: float = 1e-12):
    # preserve the sign, only clamp the magnitude
    sign = x.sign()
    mag  = x.abs().clamp(min=eps)
    return sign * torch.log(mag)

def innerpo_loss(
    pi_cho: ReprPOModelOutput,
    pi_rej: ReprPOModelOutput,
    ref_cho: ReprPOModelOutput,
    ref_rej: ReprPOModelOutput,
    batch: Dict[str, Any],
    transforms: Optional[Callable] = None,
    # custom loss_args
    α: float = 1.0,
    eps=1e-12,
    β=1,
    use_orth_loss=True,
    use_dpo_loss=True,
    use_proj_loss=False,
    use_proj_abs_loss=True,
    use_logsigmoid=True,
    use_policy_weights: bool = False,
):
    """
    movement of hs along the hs pref vector.
    """

    if transforms is not None:
        pi_cho.hs = transforms(pi_cho.hs)
        pi_rej.hs = transforms(pi_rej.hs)
        ref_cho.hs = transforms(ref_cho.hs)
        ref_rej.hs = transforms(ref_rej.hs)

    def preproc_hs(o, k: str):
        hs = o.hs[k]#.softmax(-1)
        hs = reduce_tokens_w_attention(hs, o.mask)
        hs = F.normalize(hs, p=2, dim=-1)    # <<< unit length in hidden-state space
        return hs

    def per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) -> Dict[str, Float[Tensor, "b"]]:
        hs_pi_cho = preproc_hs(pi_cho, k)
        hs_pi_rej = preproc_hs(pi_rej, k)
        hs_ref_cho = preproc_hs(ref_cho, k)  # .detach()
        hs_ref_rej = preproc_hs(ref_rej, k)  # .detach()

        # we define the reference vector as the direction between the reference chosen and rejected hidden states. It's a high dim vector in the space of hidden states
        pref_dir_ref = hs_ref_cho - hs_ref_rej  # preference vector

        # model preference direction
        pref_dir_pi = hs_pi_cho - hs_pi_rej
        # shift relative to reference
        delta = pref_dir_pi - pref_dir_ref

        # magnitude shift
        # mag_shift = delta.norm(dim=-1, keepdim=True)  # how far we moved
        # normalize by reference magnitude (to get a unitless ratio)
        ref_mag = pref_dir_ref.norm(dim=-1, keepdim=True).clamp(min=eps)

        unit = pref_dir_ref / ref_mag
        proj_vec = (delta * unit).sum(dim=-1, keepdim=True) * unit
        orth_vec = delta - proj_vec

        rel_proj = proj_vec.norm(dim=-1) / (ref_mag + eps)    # ∈ [0,∞)
        rel_orth = orth_vec.norm(dim=-1) / (ref_mag + eps)

        # normalize into bounded ratios in [0,1]
        proj_ratio = rel_proj / (1 + rel_proj)
        orth_ratio = rel_orth / (1 + rel_orth)

        # rel_pref_signed = signed_proj (can be negative)
        log_rel_pref  = safe_signed_log(rel_proj,  eps)
        log_rel_orth  = safe_signed_log(rel_orth,  eps)

        # Try to put those in the same domain as DPO
        loss_logsigmoid_proj = -F.logsigmoid(β * torch.abs(log_rel_pref)).mean()
        loss_logsigmoid_orth = -F.logsigmoid(β * (-log_rel_orth)).mean()

        return dict(
            loss_logsigmoid_orth=loss_logsigmoid_orth,
            loss_logsigmoid_proj=loss_logsigmoid_proj,
            rel_proj=rel_proj,
            rel_orth=rel_orth,
            proj_ratio=proj_ratio,
            orth_ratio=orth_ratio,
        )

    # compute losses per layer
    ll = {k: per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) for k in pi_cho.hs.keys()}
    # combine layer losses
    ll_keys = next(iter(ll.values())).keys()
    ll = {k: torch.stack([v[k] for v in ll.values()], -1).mean(-1) for k in ll_keys}

    # compute final loss as average of normalized ratio components
    # projection loss term: want more movement along ref-dir → 0 when perfect
    loss_proj_term = (1.0 - ll["proj_ratio"])# if use_proj_loss else None
    # orthogonal loss term: want no orthogonal drift → 0 when perfect
    loss_orth_term = ll["orth_ratio"]# if use_orth_loss else None
    # dpo preference term: probability of preferring chosen over rejected
    dpo_ptheta = compute_ptheta(
        pi_cho.label_logprobs,
        pi_rej.label_logprobs,
        ref_cho.label_logprobs,
        ref_rej.label_logprobs,
    )
    dpo_prob  = torch.sigmoid(β * dpo_ptheta)
    loss_dpo_prob_term  = (1.0 - dpo_prob)
    loss_logsigmoid_dpo = -F.logsigmoid(β * (1.0 - dpo_prob)).mean() 

    # collect and average whatever terms are enabled
    terms = []
    if use_proj_loss:
        if use_proj_abs_loss:
            loss_proj_term = loss_proj_term.abs()
        terms.append(loss_proj_term.mean(1))
    if use_orth_loss:
        terms.append(loss_orth_term.mean(1))
    if use_dpo_loss:
        terms.append(loss_dpo_prob_term)
    loss = torch.stack(terms, dim=0).mean()


    # Or logsigmoid style
    if use_logsigmoid:
        loss = loss_logsigmoid_dpo.mean() + ll['loss_logsigmoid_proj'].mean() + ll['loss_logsigmoid_orth'].mean()
    
    if use_policy_weights:
        policy_weights = torch.clamp(
            pi_cho['policy_weights'] + pi_rej['policy_weights'],
            max=1
        )
        loss = loss * policy_weights

    ll = {k:v.mean() for k, v in ll.items()}
    info = dict(
        loss_orth_term=loss_orth_term.mean(),
        loss_proj_term=loss_proj_term.mean(),
        loss_dpo_term=loss_dpo_prob_term.mean(),
        dpo_prob=dpo_prob.mean(),
        dpo_ptheta=dpo_ptheta.mean(),
        loss_logsigmoid_dpo=loss_logsigmoid_dpo.mean(),
        **ll,
    )

    return loss, info


@dataclass
class InnerPOLossConfig:
    """
    moves the hidden states of the chosen and rejected hidden states apart along the preference vector, with some constraints, while also doing DPO on outpts
    - keep text at least as coherent (relu(mode/base), (nll_loss)
    - keep the chosen answer at least prefered (relu(rej-cho) dpo_loss
    - punish movement orthogonal to the preference vector: by distance * β
    - punish movement orthogonal to the preference vector: by angle * β
    """

    # α: float = 1.0
    # """balance between reroute and retain loss."""

    eps: float = 1.0e-12

    β: float = 1.
    """factor to punish orthogonal movement"""

    use_dpo_loss: bool = True
    """punish model if rejected completion is more likely than the chosen"""

    use_orth_loss: bool = True
    """punish movement orthogonal to the preference vector: by distance"""

    use_proj_loss: bool = False
    """encourage chosen to be more in the pref dir than rejected"""

    use_proj_abs_loss: bool = True
    """use absolute value of the projection loss, otherwise use relative"""

    use_logsigmoid: bool = False

    use_policy_weights: bool = False
    """# Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827"""
   

    def c(self, *args, **kwargs):
        return innerpo_loss(*args, **kwargs, **asdict(self))
