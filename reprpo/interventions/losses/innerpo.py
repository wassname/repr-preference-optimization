from jaxtyping import Float
from typing import Any, Callable, Dict, Optional
from torch import Tensor
from torch.nn import functional as F
import torch
from dataclasses import dataclass, asdict

from .helpers import cross_entropy_loss, compute_ptheta
from ..types import ReprPOModelOutput
from ..reprpo.helpers import reduce_tokens_w_attention


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
    β=0.1,
    use_orth_loss=True,
    use_sep_loss=False,
    use_angle_loss=True,
    use_dpo_loss=True,
    use_proj_loss=False,
    use_proj_abs_loss=True,
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
        return hs

    def per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) -> Dict[str, Float[Tensor, "b"]]:
        hs_pi_cho = preproc_hs(pi_cho, k)
        hs_pi_rej = preproc_hs(pi_rej, k)
        hs_ref_cho = preproc_hs(ref_cho, k)  # .detach()
        hs_ref_rej = preproc_hs(ref_rej, k)  # .detach()

        # we define the reference vector as the direction between the reference chosen and rejected hidden states. It's a high dim vector in the space of hidden states
        pref_dir_ref = hs_ref_cho - hs_ref_rej  # preference vector

        def signed_proj_magnitude(a, pref_dir):
            pref_dir_unit = pref_dir / pref_dir.norm(dim=-1, keepdim=True).clamp(eps)

            # get projection of `a` along ref_dir
            a_proj = (pref_dir_unit * a).sum(dim=-1, keepdim=True) * pref_dir_unit

            a_orth = a - a_proj
            cosine_sim = F.cosine_similarity(a, pref_dir, dim=-1)
            return a_proj.mean(dim=-1), a_orth.mean(dim=-1), cosine_sim

        rel_pref, rel_orth, rel_cossim = signed_proj_magnitude(hs_pi_cho-hs_pi_rej, pref_dir_ref)

        loss_proj = -rel_pref

        loss_sep_dist = -torch.norm(hs_pi_cho - hs_pi_rej, dim=-1)

        loss_orth = torch.abs(rel_orth)

        # we could also optimize angle, we want it to be close to 1, so we make it negative
        loss_angle = -torch.abs(rel_cossim) # aligned
        loss_angle_orth = 1-torch.abs(rel_cossim) # orth

        return dict(
            loss_orth=loss_orth,
            loss_angle=loss_angle,
            loss_proj=loss_proj,
            loss_sep_dist=loss_sep_dist,
            loss_angle_orth=loss_angle_orth,
            _cosine_similarity=rel_cossim,
        )

    # compute losses per layer
    ll = {k: per_layer(pi_cho, pi_rej, ref_cho, ref_rej, k) for k in pi_cho.hs.keys()}
    # combine layer losses
    ll_keys = next(iter(ll.values())).keys()
    ll = {k: torch.stack([v[k] for v in ll.values()], -1).mean(-1) for k in ll_keys}

    loss_inner = torch.zeros_like(ll["loss_proj"])
    if use_proj_loss:
        loss_inner = ll["loss_proj"] * 1e5 # a very small number so we make it bigger
    if use_proj_abs_loss:
        loss_inner = torch.abs(loss_inner)

    # this beeds balance
    if use_orth_loss:
        loss_inner += β * ll["loss_orth"] * 1e6
    
    if use_angle_loss:
        loss_inner += β * ll["loss_angle"] * 1e5
    
    # nll loss, to ensure it's punished for less coherent outputs
    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"], batch["chosen_mask"])
    ref_nll_loss = cross_entropy_loss(
        ref_cho.logits, batch["chosen"], batch["chosen_mask"]
    )
    nll_loss_ratio = (nll_loss - ref_nll_loss)
    loss_nll = F.relu(nll_loss_ratio).mean(1)

    # dpo loss, punished model if rejected completion is more likely than the chosen
    # normally dpo has logsigmoid, https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L82
    dpo_ptheta = compute_ptheta(
        pi_cho.label_logprobs,
        pi_rej.label_logprobs,
        ref_cho.label_logprobs,
        ref_rej.label_logprobs,
    )

    loss_dpo = torch.zeros_like(loss_inner)
    if use_dpo_loss:
        loss_dpo = F.logsigmoid(-dpo_ptheta)

    loss = loss_inner.mean() + α * loss_dpo.mean()

    info = dict(
        loss_reroute=loss_inner,
        loss_dpo=loss_dpo,
        loss_nll=loss_nll,
        nll_loss_ratio=nll_loss_ratio,
        ptheta=dpo_ptheta,
        **ll,
    )
    info = {k: v.mean().detach() for k, v in info.items()}

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

    β: float = 1
    """factor to punish orthogonal movement"""

    use_dpo_loss: bool = True
    """punish model if rejected completion is more likely than the chosen"""

    use_orth_loss: bool = False
    """punish movement orthogonal to the preference vector: by distance"""

    use_sep_loss: bool = False
    """punish closeness between chosen and rejected hidden states"""

    use_proj_loss: bool = True
    """encourage chosen to be more in the pref dir than rejected"""

    use_proj_abs_loss: bool = True
    """encourage chosen to be more in the pref dir than rejected, but absolute value"""

    use_angle_loss: bool = False
    """punish movement orthogonal to the preference vector: by angle"""

    def c(self, *args, **kwargs):
        return innerpo_loss(*args, **kwargs, **asdict(self))
