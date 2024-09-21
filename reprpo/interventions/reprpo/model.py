
import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from torch import Tensor
from jaxtyping import Float, Int
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from types import SimpleNamespace
from baukit.nethook import TraceDict, get_module
from dataclasses import dataclass

from reprpo.interventions.types import ReprPOModelOutput, HS, Mask
from reprpo.interventions.pl_base import PL_MODEL, ModelConfigBase
from reprpo.interventions.helpers import compute_logprobs

from .helpers import get_layer_paths, validate_layer_paths
from ..losses import Losses
from ..transforms import Transforms


def reprpo_forward_baukit(model, input_ids, attn_mask, layer_paths, collect_input=True):

    reprs = {}
    with TraceDict(
        model,
        layer_paths,
        retain_input=collect_input,
        retain_output=(not collect_input),
        retain_grad=True,
    ) as ret:
        outs = model(
            input_ids,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        for p in layer_paths:
            if collect_input:
                reprs[p] = ret[p].input
            else:
                reprs[p] = ret[p].output
            assert torch.isfinite(reprs[p]).all()

    logprobs = compute_logprobs(
        logits=outs.logits, labels=input_ids, selection_mask=attn_mask
    )

    return ReprPOModelOutput(hs=reprs, logits=outs.logits, label_logprobs=logprobs, mask=attn_mask)



class PL_REPRPO_MODEL(PL_MODEL):
    def __init__(self, *args, collection_layers_side, collect_input, collection_keys_in: tuple=None, collection_keys_out: tuple=None,  
                 loss_fn: Losses,
                 transform: Transforms,
                 **kwargs):
        super().__init__(*args, **kwargs)
        collection_keys = collection_keys_in if collect_input else collection_keys_out
        self.hparams.layer_paths = get_layer_paths(collection_keys, collection_layers_side)
        validate_layer_paths(self._model, self.hparams.layer_paths)
        self.hparams.collect_input = collect_input

        # we do one learnable householder roation per layer
        if collect_input:
            hra_sizes = {p:get_module(self._model, p).in_features for p in self.hparams.layer_paths}
        else:
            hra_sizes = {p:get_module(self._model, p).out_features for p in self.hparams.layer_paths}
        
        self.transforms = torch.nn.ParameterDict({
            k: transform.c(dim_hs, dim_hs) for k,dim_hs in hra_sizes.items()})
        self.transforms = self.transforms.to(self._model.dtype).to(self._model.device)

    def _loss_fn(self, batch, model):
        h = self.hparams
        model.eval()
        with torch.no_grad():
            with model.disable_adapter():
                ref_cho = reprpo_forward_baukit(
                    model=model,
                    input_ids=batch["chosen"],
                    attn_mask=batch["chosen_mask"],
                    layer_paths=h.layer_paths,
                    collect_input=h.collect_input,
                )
                ref_rej = reprpo_forward_baukit(
                    model=model,
                    input_ids=batch["rejected"],
                    attn_mask=batch["rejected_mask"],
                    layer_paths=h.layer_paths,
                    collect_input=h.collect_input,
                )

        model.train()
        pi_cho = reprpo_forward_baukit(
            model=model,
            input_ids=batch["chosen"],
            attn_mask=batch["chosen_mask"],
            layer_paths=h.layer_paths,
            collect_input=h.collect_input,
        )
        pi_rej = reprpo_forward_baukit(
            model=model,
            input_ids=batch["rejected"],
            attn_mask=batch["rejected_mask"],
            layer_paths=h.layer_paths,
            collect_input=h.collect_input,
        )
        
        return h.loss_fn.c(
            pi_cho=pi_cho,
            pi_rej=pi_rej,
            ref_cho=ref_cho,
            ref_rej=ref_rej,
            batch=batch,
        )


