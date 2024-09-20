import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from einops import rearrange, repeat, reduce
import math
import warnings

from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from reprpo.train.pl_base import PL_MODEL, TrainingArgumentswCollection, cross_entropy_loss
from reprpo.train.dpo import compute_logprobs, compute_dpo_loss, compute_ptheta
from types import SimpleNamespace
from baukit.nethook import TraceDict
from dataclasses import dataclass
import itertools
from ..layers.ether import ETHERLinear, ETHERLinearSmall, _ETHERConfig
from ..layers.hra import HRATransform
from .reprpo_hra import mean_with_attention, reprpo_forward, dist_ratio

from torch.linalg import vecdot

def loss_dist_along_pref_vect(pi_cho, pi_rej, attn_cho, ref_cho, ref_rej, attn_rej) -> Float[Tensor, "b"]:
    """Reward for moving hs along preference vector
    """

    def proj(a, b):
        """Project a onto b"""
        dot = torch.linalg.vecdot
        d = dot(a, b)[..., None]  / dot(b, b)[..., None] * b
        return d

    def dist(a, ref_dir):
        """Distance from origin with sign"""
        dot = torch.linalg.vecdot
        return torch.norm(a, dim=-1) * torch.sign(dot(a, ref_dir))
    
    pi_cho = mean_with_attention(pi_cho, attn_cho)
    pi_rej = mean_with_attention(pi_rej, attn_rej)
    ref_cho = mean_with_attention(ref_cho, attn_cho)#.detach()
    ref_rej = mean_with_attention(ref_rej, attn_rej)#.detach()

    ref_dir = (ref_cho - ref_rej) # preference vector
    ref_dir_norm = torch.norm(ref_dir, dim=-1, keepdim=True)
    cho = pi_cho-ref_cho # vector describing movement of chosen hidden state compared to base model
    rej = pi_rej-ref_rej

    def proj(a, b):
        """Project a onto b"""
        return torch.sum(cho * ref_dir, dim=-1, keepdim=True) * ref_dir / ref_dir_norm
    proj_cho = proj(cho, ref_dir)
    ortho_cho = cho - proj_cho
    proj_rej = proj(rej, ref_dir)
    ortho_rej = rej - proj_rej

    # here we have a loss where we reward the model for moving the hidden states along the preference (chosen-rejected) vector
    # and punished for any movement orthogonal to this vector
    β = 2 # factor to punish orthogonal movement
    loss_reroute = -dist(proj_cho, ref_dir) -dist(proj_rej, ref_dir) + abs(dist(ortho_cho, ref_dir)) * β  + abs(dist(ortho_rej, ref_dir)) * β 


    return loss_reroute


def compute_reprpo_hs_dist_loss_batch(batch, model, alpha, collection_layers, transform):

    model.eval()
    with torch.no_grad():
        with model.disable_adapter():
            ref_cho = reprpo_forward(
                model=model,
                input_ids=batch["chosen"],
                attn_mask=batch["chosen_mask"],
                collection_layers=collection_layers,
            )
            ref_rej = reprpo_forward(
                model=model,
                input_ids=batch["rejected"],
                attn_mask=batch["rejected_mask"],
                collection_layers=collection_layers,
            )

    model.train()
    pi_cho = reprpo_forward(
        model=model,
        input_ids=batch["chosen"],
        attn_mask=batch["chosen_mask"],
        collection_layers=collection_layers,
    )
    pi_rej = reprpo_forward(
        model=model,
        input_ids=batch["rejected"],
        attn_mask=batch["rejected_mask"],
        collection_layers=collection_layers,
    )
    assert torch.isfinite(pi_rej.hs).all()
    cho_attn_mask = batch["chosen_mask"]
    rej_attn_mask = batch["rejected_mask"]
    # comb_attn_mask = cho_attn_mask * rej_attn_mask


    def p(hs):
        """to log probs."""
        # hs = model.lm_head(hs)
        hs = torch.log_softmax(hs, -1)
        return hs

    def t(hs):
        """use learnable transformation to get the residual part of the hs"""
        # return hs
        return transform(hs)

    # loss_retain: more of chosen, less of rejected, on the plane defined by the learnable orthogonal transformation
    loss_t_reroute = loss_dist_along_pref_vect(
        t(pi_cho.hs),
        t(pi_rej.hs),
        cho_attn_mask,
        t(ref_cho.hs),
        t(ref_rej.hs),
        rej_attn_mask,
    )

    # # loss_reroute: keep chosen the same
    loss_reroute = loss_dist_along_pref_vect(
        pi_cho.hs,
        pi_rej.hs,
        cho_attn_mask,
        ref_cho.hs,
        ref_rej.hs,
        rej_attn_mask,
    )

    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"], batch['chosen_mask'])
    ref_nll_loss = cross_entropy_loss(ref_cho.logits, batch["chosen"], batch['chosen_mask'])
    nll_loss_ratio = nll_loss - ref_nll_loss

    # we want to punish it for coherency loss, which nll is a proxy for
    # but not reward it for coherency increase over the reference model
    # as this is not what we are optimising
    loss_nll_retain = F.relu(nll_loss_ratio) #  - math.log(0.9))
    # β = 2
    # loss_nll_retain = (β*nll_loss_ratio - 1) ** 2
    # loss_nll_retain = F.logsigmoid(nll_loss - ref_nll_loss)

    # or use nll on cho?

    # get the dpo metrics for comparison
    loss_dpo, info = compute_dpo_loss(
        pi_cho.logprobs,
        pi_rej.logprobs,
        ref_cho.logprobs,
        ref_rej.logprobs,
    )
    ptheta = compute_ptheta(
        pi_cho.logprobs,
        pi_rej.logprobs,
        ref_cho.logprobs,
        ref_rej.logprobs,
    )
    
    
    # this loss says: find a transform, where you can make the good hidden states more common and the bad hidden states less common, while not making the outputs incoherent (nll) or favouring the bad response (dpo)
    # TODO if this works tidy it up, and find a way to naturally balance the losses, for example make all in log domain
    loss_dpo_retain = F.relu(-ptheta)
    # loss_reroute = loss_t_reroute
    # loss_retain = loss_dpo
    loss_retain = loss_nll_retain.mean(1) + loss_dpo_retain
    loss = loss_reroute.mean() + alpha * loss_retain.mean()

    def cosine_on_keys(hs1, hs2):
        return F.cosine_similarity(hs1, hs2, dim=-1).nanmean()
    
    with torch.no_grad():
        info['retain_cosine'] = cosine_on_keys(pi_cho.hs, ref_cho.hs)
        info['rr_cosine'] = cosine_on_keys(pi_rej.hs, ref_cho.hs)

        # Lets monitor the comparitive norms of the decomposed parts
        hs = torch.norm(ref_cho.hs)
        hs_t = torch.norm(t(ref_cho.hs))
        hs_resid = torch.norm(ref_cho.hs - t(ref_cho.hs))
        info['hs_t/hs'] = (hs_t / hs).mean()
        info['hs_resid/hs'] = (hs_resid / hs).mean()
        info['hs_t/hs_resid'] = (hs_t / hs_resid).mean()

        # also the norm of the weights of the transformation
        info['transform_norm'] = sum([torch.norm(p).mean() for p in transform.parameters()])


        info = dict(
            loss_reroute=loss_reroute.mean(),
            loss_retain=loss_retain.mean() * alpha,
            nll_loss=nll_loss.mean(),
            ref_nll_loss=ref_nll_loss.mean(),
            nll_loss_ratio=nll_loss_ratio.mean(),
            loss_nll_retain=loss_nll_retain.mean(),
            loss_dpo_retain=loss_dpo_retain.mean(),
            **info,
        )
    assert torch.isfinite(loss)
    return loss, info

class PL_REPRPO_HS_DIST_MODEL(PL_MODEL):
    def __init__(self, *args, alpha, collection_layers, 
                #  r, apply_GS,  
                nb, Htype, ether_dropout, flip_side,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        self.hparams.collection_layers = collection_layers

        dim_hs = self._model.config.hidden_size
        # dim_hs = self._model.config.vocab_size
        # self.transform = HRATransform(dim_hs, dim_hs, r=r, apply_GS=apply_GS)
        self.transform = ETHERLinear(dim_hs, dim_hs, nb=nb,
                                                      Htype=Htype,
                                                      ether_dropout=ether_dropout,
                                                      flip_side=flip_side,)

    def _loss_fn(self, batch, model):
        return compute_reprpo_hs_dist_loss_batch(
            batch,
            model,
            self.hparams.alpha,
            self.hparams.collection_layers,
            self.transform,
        )


@dataclass
class HSDist(_ETHERConfig, TrainingArgumentswCollection):
    """
    Loss: movement of hs along the hs pref vector.
    
    Transform: ETHER  
     
    """

    alpha: int = 1
    """balancing retrain and reroute losses"""

    collection_layers: tuple=(10, 20, 30) 
    """The layers to collect the hidden states from. HRA operates on the residual stream so only needs a couple of points of collection"""

    lr: float = 1e-3

    Htype: str = 'etherplus'

    nb: int = 32

    rel_loss: bool = True

    _reprpo_class = PL_REPRPO_HS_DIST_MODEL
    _model_keys = ['alpha', 'collection_layers',  'nb', 'Htype', 'ether_dropout', 'flip_side', ]
