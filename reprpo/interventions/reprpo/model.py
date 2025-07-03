import torch
import torch.nn.functional as F
from typing import Union, List
import os
from baukit.nethook import TraceDict, get_module
import numpy as np
from loguru import logger
from reprpo.interventions.types import ReprPOModelOutput
from reprpo.interventions.pl_base import PL_MODEL
from reprpo.interventions.dpo_helpers import compute_logprobs
import warnings
from typing import List
import re
from typing import Optional
from .helpers import get_layer_paths, validate_layer_paths
from ..losses import LossesType
from ..transforms import TransformType
from ..dpo import calc_dpo_loss_w_metrics


def get_regexp_layers(collection_keys: List[str], model):
    """
    Select layers that match a regex pattern e.g.
    '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'
    or just suffix like ['q', 'v'] or ['q_proj', 'v_proj']

    see how peft does it https://github.com/huggingface/peft/blob/8af29c646860e617b641225caf7ef47f7c3dcd26/src/peft/tuners/tuners_utils.py#L458
    """
    lyrs = dict(model.named_modules()).keys()
    out = []
    for k in collection_keys:
        if k.startswith(".*") or k.endswith("$"):
            # regex
            out += [l for l in lyrs if re.search(k, l)]
        else:
            # suffix
            out += [l for l in lyrs if l.endswith(k)]
    out = list(set(out))
    if len(out) == 0:
        raise ValueError(
            f"Collection keys {collection_keys} do not match any layers in the model. Layers found: {lyrs}"
        )
    return out


def reprpo_forward_baukit(
    model, input_ids, attn_mask, layer_paths, collect_input=True, collect_hs=False, prompt_mask=None, special_tokens_mask=None, logp_agg_type='ipo', calc_wpo=False, calc_mallows=False
):
    # if the layer paths are just str(ints) then just collect the hidden states
    if collect_hs:
        layer_paths = [int(p) for p in layer_paths]
        outs = model(
            input_ids,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        reprs = {str(k): outs.hidden_states[k] for k in layer_paths}
    else:
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

    for p in reprs:
        if os.environ.get("DEBUG", False):
            if not torch.isfinite(reprs[p]).all():
                warnings.warn(
                    f"gathered activations for layer [{p}] are not finite {reprs[p]}. This could be due to an high lr or unstable loss function."
                )
        else:
            assert torch.isfinite(
                reprs[p]
            ).all(), f"gathered activations for layer [{p}] are not finite {reprs[p]}. This could be due to an high lr or unstable loss function."

    # we should filter out the prompt which is the same for both chosen and rejected
    if prompt_mask is not None:
        attn_mask = attn_mask * (1-prompt_mask)
    # we should filter out the special tokens which might contain attention sinks
    if special_tokens_mask is not None:
        attn_mask = attn_mask * (1-special_tokens_mask)

    out_lp = compute_logprobs(
        logits=outs.logits, input_ids=input_ids, selection_mask=attn_mask, logp_agg_type=logp_agg_type, calc_wpo=calc_wpo, calc_mallows=calc_mallows,
    )
    return ReprPOModelOutput(
        hs=reprs, logits=outs.logits, label_logprobs=out_lp['label_logp'], mask=attn_mask, log_policy_weights=out_lp['log_policy_weights'],
    )



def parse_collection_layers(
    collection_layers: str, num_hidden_layers: int
) -> List[int]:
    """
    Parse the collection layers. Supports various formats:
    - A comma-separated string like "-2,-1" to collect the last two layers
    - A string representing a range, e.g., "range(3,10,2)"
    - A shorthand range with percentages, e.g., "0.5, 0.9, 2" which converts 0.5 to the 50% layer and 0.9 to the 90% layer
    - A list of integers or floats
    """
    # Convert string input to a list 
    collection_layers = collection_layers.strip()
    if collection_layers.startswith("range(") and collection_layers.endswith(")"):
        method = range
        collection_layers = collection_layers.split("(", 1)[1].rstrip(")")
    else:
        method = list
    
    collection_layers = [float(item.strip()) for item in collection_layers.split(",") if item.strip() != ""]

    # now convert negative indices to positive ones
    collection_layers = [
        float(layer) if float(layer) >= 0 else num_hidden_layers + int(layer)
        for layer in collection_layers
    ]
    # now float to int
    collection_layers = [
        int(layer) if int(layer)== layer else int(layer*num_hidden_layers)
        for layer in collection_layers
    ]
    
    # Check if the collection layers are within the valid range
    for layer in collection_layers:
        if layer < 0 or layer >= num_hidden_layers:
            raise ValueError(
                f"Invalid collection layer {layer}. Must be between 0 and {num_hidden_layers - 1}."
            )
        
    if method is range:
        # Convert to a list of integers
        collection_layers = list(
            range(
                collection_layers[0],
                collection_layers[1],
                collection_layers[2] if len(collection_layers) > 2 else 1,
            )
        )
    
    # remove duplicates while keeping order
    collection_layers = list(dict.fromkeys(collection_layers))

    # TODO unit tests range(0.3, -2, 2), "0.5,-2,-1" "1,2,-1"
    return collection_layers

class PL_REPRPO_MODEL(PL_MODEL):
    def __init__(
        self,
        *args,
        collection_layers,
        collect_input,
        collect_hs,
        collection_keys_in: tuple = None,
        collection_keys_out: tuple = None,
        logp_agg_type: str = "ipo",
        loss: LossesType,
        transform: TransformType,
        calc_wpo: bool = False,
        calc_mallows: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hparams.loss = loss
        self.hparams.transform = transform
        self.hparams.collection_layers = collection_layers
        self.hparams.collect_input = collect_input
        self.hparams.collect_hs = collect_hs
        self.hparams.logp_agg_type = logp_agg_type
        self.hparams.calc_wpo = calc_wpo
        self.hparams.calc_mallows = calc_mallows

        collection_keys = collection_keys_in if collect_input else collection_keys_out
        collection_keys = get_regexp_layers(
            collection_keys, self._model
        )

        N = self._model.config.num_hidden_layers
        if collection_layers is None:
            collection_layers = "range(0.3, -2)"
        collection_layers = parse_collection_layers(
            collection_layers, num_hidden_layers=N
        )
        logger.info(
            f"Using collection layers: {collection_layers} for {type(transform).__name__}"
        )

        # set layer_paths
        if not collect_hs:
            self.hparams.layer_paths = get_layer_paths(
                collection_keys, collection_layers
            )
            validate_layer_paths(self._model, self.hparams.layer_paths)
            # we do one learnable householder roation per layer
            if collect_input:
                hra_sizes = {
                    p: get_module(self._model, p).in_features
                    for p in self.hparams.layer_paths
                }
            else:
                hra_sizes = {
                    p: get_module(self._model, p).out_features
                    for p in self.hparams.layer_paths
                }
        else:            
            self.hparams.layer_paths = [str(s) for s in collection_layers]
            hra_sizes = {
                k: self._model.config.hidden_size for k in self.hparams.layer_paths
            }

        self.transforms = transform.c(hra_sizes, model=self._model)

        self.transforms = self.transforms.to(self._model.dtype).to(self._model.device)

    def _loss_fn(self, batch, model):
        h = self.hparams

        # collect the representations
        model.eval()
        with torch.no_grad():
            with model.disable_adapter():
                ref_cho = reprpo_forward_baukit(
                    model=model,
                    input_ids=batch["chosen_ids"],
                    attn_mask=batch["chosen_mask"],
                    layer_paths=h.layer_paths,
                    collect_input=h.collect_input,
                    collect_hs=h.collect_hs,
                    prompt_mask=batch["prompt_mask"],
                    special_tokens_mask=batch["chosen_special_tokens_mask"],
                    logp_agg_type=h.logp_agg_type,
                    calc_wpo=h.calc_wpo,
                    calc_mallows=h.calc_mallows,
                )
                ref_rej = reprpo_forward_baukit(
                    model=model,
                    input_ids=batch["rejected_ids"],
                    attn_mask=batch["rejected_mask"],
                    layer_paths=h.layer_paths,
                    collect_input=h.collect_input,
                    collect_hs=h.collect_hs,
                    prompt_mask=batch["prompt_mask"],
                    special_tokens_mask=batch["rejected_special_tokens_mask"],
                    logp_agg_type=h.logp_agg_type,
                    calc_wpo=h.calc_wpo,
                    calc_mallows=h.calc_mallows,
                )

        model.train()
        pi_cho = reprpo_forward_baukit(
            model=model,
            input_ids=batch["chosen_ids"],
            attn_mask=batch["chosen_mask"],
            layer_paths=h.layer_paths,
            collect_input=h.collect_input,
            collect_hs=h.collect_hs,
            prompt_mask=batch["prompt_mask"],
            special_tokens_mask=batch["chosen_special_tokens_mask"],
            logp_agg_type=h.logp_agg_type,
            calc_wpo=h.calc_wpo,
            calc_mallows=h.calc_mallows,
        )
        pi_rej = reprpo_forward_baukit(
            model=model,
            input_ids=batch["rejected_ids"],
            attn_mask=batch["rejected_mask"],
            layer_paths=h.layer_paths,
            collect_input=h.collect_input,
            collect_hs=h.collect_hs,
            prompt_mask=batch["prompt_mask"],
            special_tokens_mask=batch["rejected_special_tokens_mask"],
            logp_agg_type=h.logp_agg_type,
            calc_wpo=h.calc_wpo,
            calc_mallows=h.calc_mallows,
        )

        # run loss function
        loss, info = h.loss.c(
            pi_cho=pi_cho,
            pi_rej=pi_rej,
            ref_cho=ref_cho,
            ref_rej=ref_rej,
            batch=batch,
            transforms=self.transforms,
        )

        with torch.no_grad():
            # get the dpo metrics for comparison, this also has nll, and cosine
            _, info_dpo = calc_dpo_loss_w_metrics(
                batch,
                pi_cho,
                pi_rej,
                ref_cho,
                ref_rej,
            )

        return loss, {**info, **info_dpo}
