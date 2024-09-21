import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from torch import Tensor
from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from reprpo.interventions.pl_base import PL_MODEL, TrainingArgumentswCollection, cross_entropy_loss
from reprpo.interventions.dpo import compute_logprobs, compute_dpo_loss
from types import SimpleNamespace
from baukit.nethook import TraceDict
from dataclasses import dataclass
import itertools
from reprpo.interventions.reprpo_hra import reprpo_forward, dist_ratio

def compute_reprpo_hs_loss_batch(batch, model, alpha, collection_layers_hs):

    model.eval()
    with torch.no_grad():
        with model.disable_adapter():
            ref_cho = reprpo_forward(
                model=model,
                input_ids=batch["chosen"],
                attn_mask=batch["chosen_mask"],
                collection_layers_hs=collection_layers_hs,
            )
            ref_rej = reprpo_forward(
                model=model,
                input_ids=batch["rejected"],
                attn_mask=batch["rejected_mask"],
                collection_layers_hs=collection_layers_hs,
            )

    model.train()
    pi_cho = reprpo_forward(
        model=model,
        input_ids=batch["chosen"],
        attn_mask=batch["chosen_mask"],
        collection_layers_hs=collection_layers_hs,
    )
    pi_rej = reprpo_forward(
        model=model,
        input_ids=batch["rejected"],
        attn_mask=batch["rejected_mask"],
        collection_layers_hs=collection_layers_hs,
    )
    assert torch.isfinite(pi_rej.hs).all()
    cho_attn_mask = batch["chosen_mask"]
    rej_attn_mask = batch["rejected_mask"]

    comb_attn_mask = cho_attn_mask * rej_attn_mask
    # loss_retain: the representation of policy chosen responses should be closer to the reference chosen responses
    # and again we scale it using the reference model as a stable target
    # so should start at a -ve and go to 0 (as we optimize rr, not this)
    loss_retain = dist_ratio(
        ref_cho.hs.detach(),
        pi_cho.hs,
        comb_attn_mask,
        ref_cho.hs,
        ref_rej.hs,
        comb_attn_mask,
    ) 


    # loss_reroute: the representation of policy rejected responses should be closer to the reference chosen responses
    # we measure it as a ratio to the distance between the chosen responses and the rejected responses in the reference model as this is a stable target
    # start at 0 and go to -ve inf
    # start at log(1)=0 go to log(0)=-inf
    loss_reroute = dist_ratio(
        (ref_cho.hs).detach(),
        (pi_rej.hs),
        comb_attn_mask,
        (ref_cho.hs),
        (ref_rej.hs),
        comb_attn_mask,
    )

    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"], batch['chosen_mask'])
    ref_nll_loss = cross_entropy_loss(ref_cho.logits, batch["chosen"], batch['chosen_mask'])
    nll_loss_ratio = nll_loss / ref_nll_loss
    
    loss = (loss_reroute.mean() + loss_retain * alpha).nanmean()

    # get the dpo metrics for comparison
    _, info = compute_dpo_loss(
        pi_cho.logprobs,
        pi_rej.logprobs,
        ref_cho.logprobs,
        ref_rej.logprobs,
    )

    def cosine_on_keys(hs1, hs2):
        return F.cosine_similarity(hs1, hs2, dim=-1).nanmean()
    
    with torch.no_grad():
        info['retain_cosine'] = cosine_on_keys(pi_cho.hs, ref_cho.hs)
        info['rr_cosine'] = cosine_on_keys(pi_rej.hs, ref_cho.hs)

        # # Lets monitor the comparitive norms of the decomposed parts
        # hs = norm(ref_cho.hs)
        # hs_r = norm((ref_cho.hs))
        # hs_io = norm(decomposer(ref_cho.hs))
        # info['hs_r/hs'] = (hs_r / hs).mean()
        # info['hs_io/hs'] = (hs_io / hs).mean()
        # info['hs_r/hs_io'] = (hs_r / hs_io).mean()


        info = dict(
            loss_reroute=loss_reroute.mean(),
            loss_retain=loss_retain.mean() * alpha,
            nll_loss=nll_loss.mean(),
            ref_nll_loss=ref_nll_loss.mean(),
            nll_loss_ratio=nll_loss_ratio.mean(),
            **info,
        )
    assert torch.isfinite(loss)
    return loss, info

def validate_args(model, args):
    # first check collection layers exist'

    # HACK: llama specific
    assert max(args.collection_layers_hs)<(model.config.num_hidden_layers+1), 'collection layers should be less than the number of layers'

class PL_REPRPO_HS_MODEL(PL_MODEL):
    def __init__(self, *args, alpha, collection_layers_hs, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        self.hparams.collection_layers_hs = collection_layers_hs
        validate_args(self._model, self.hparams)



    def _loss_fn(self, batch, model):
        return compute_reprpo_hs_loss_batch(
            batch,
            model,
            self.hparams.alpha,
            self.hparams.collection_layers_hs,
        )

@dataclass
class HS(TrainingArgumentswCollection):
    """Transform: None. Target: hidden_states"""

    alpha: float = 0.3
    """weights retrain and reroute losses"""

    lr: float = 3e-5

    _reprpo_class = PL_REPRPO_HS_MODEL
    _model_keys = ['alpha', 'collection_layers_hs']


