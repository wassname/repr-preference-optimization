import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from torch import Tensor
from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from reprpo.train.pl_base import PL_MODEL, TrainingArgumentswCollection, cross_entropy_loss
from reprpo.train.dpo import compute_logprobs, compute_dpo_loss
from types import SimpleNamespace
from baukit.nethook import TraceDict
from dataclasses import dataclass
import itertools
from reprpo.helpers.svd_decomposer import SoftSVDDecomposer, DualSVDDecomposer, SVDDecomposer
from reprpo.train.reprpo_hra import reprpo_forward, norm_mean, dist_ratio

def compute_reprpo_svd_loss_batch(batch, model, alpha, collection_layers, decomposer):

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

    def res_det(hs):
        """use SVD to decompose hs into the inputoutput and residual components"""
        hs_io = decomposer(hs)
        return hs - hs_io#.detach() # FIXME, should I not detatch this?

    # loss_reroute: the representation of policy rejected responses should be closer to the reference chosen responses
    # we measure it as a ratio to the distance between the chosen responses and the rejected responses in the reference model as this is a stable target
    # start at 0 and go to -ve inf
    # start at log(1)=0 go to log(0)=-inf
    loss_reroute = dist_ratio(
        res_det(ref_cho.hs).detach(),
        res_det(pi_rej.hs),
        comb_attn_mask,
        res_det(ref_cho.hs),
        res_det(ref_rej.hs),
        comb_attn_mask,
    )

    nll_loss = cross_entropy_loss(pi_cho.logits, batch["chosen"])
    ref_nll_loss = cross_entropy_loss(ref_cho.logits, batch["chosen"])
    nll_loss_ratio = nll_loss / ref_nll_loss
    
    loss = (loss_reroute + loss_retain * alpha).nanmean()

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

        # Lets monitor the comparitive norms of the decomposed parts
        hs = norm_mean(ref_cho.hs)
        hs_r = norm_mean(res_det(ref_cho.hs))
        hs_io = norm_mean(decomposer(ref_cho.hs))
        info['hs_r/hs'] = (hs_r / hs).mean()
        info['hs_io/hs'] = (hs_io / hs).mean()
        info['hs_r/hs_io'] = (hs_r / hs_io).mean()


        info = dict(
            loss_reroute=loss_reroute.mean(),
            loss_retain=loss_retain.mean() * alpha,
            nll_loss=nll_loss,
            ref_nll_loss=ref_nll_loss,
            nll_loss_ratio=nll_loss_ratio,
            **info,
        )
    assert torch.isfinite(loss)
    return loss, info

class PL_REPRPO_SVD_MODEL(PL_MODEL):
    def __init__(self, *args, alpha=1, collection_layers=[10, 20], quantile=0.75, dual_svd=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        self.hparams.collection_layers = collection_layers

        # convert
        if dual_svd:
            self.decomposer = DualSVDDecomposer(
                self._model.get_input_embeddings().weight.clone().float(),
                self._model.lm_head.weight.clone(),
                quantile=quantile,
            )
        else:
            if quantile < 1:
                self.decomposer = SoftSVDDecomposer(
                    self._model.lm_head.weight.clone().float(), quantile=quantile
                )
            else:
                self.decomposer = SVDDecomposer(
                    self._model.lm_head.weight.clone().float()
                )

    def _loss_fn(self, batch, model):
        return compute_reprpo_svd_loss_batch(
            batch,
            model,
            self.hparams.alpha,
            self.hparams.collection_layers,
            self.decomposer,
        )



@dataclass
class SVD(TrainingArgumentswCollection):
    """
    Target: hs. Transform: SVD
    """
    """
    This intervention does not seem to work of converge. It attempt to remove the parts of the hs that are used by lm_head, but because hs is low rank, and lm_head is highrank, it is all used. Hence we try to train on a tiny noisy residual, and cannot.

    It's left in here to show a negative finding, and the question: where do transformer store the working internal memory?
    """

    alpha: int = 0.3
    """weights retrain and reroute losses"""

    quantile: float=0.5
    """What quantile of top singular values to from from hs

    Note if you set this to 1, we switch to normal SVD
    
    we decompose the embedded and de-embedding layers using SVD then remove the top <quantile> of singular values from the hidden states"""


    dual_svd: bool = False
    """if true, will use the embedding and lm_head, if false only lm_head"""

    lr: float = 3e-5

    _reprpo_class = PL_REPRPO_SVD_MODEL
    _model_keys = ['alpha', 'quantile', 'dual_svd', 'collection_layers']
