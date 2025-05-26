import torch
import torch.nn.functional as F

import os
from baukit.nethook import TraceDict, get_module
import numpy as np
from loguru import logger
from reprpo.interventions.types import ReprPOModelOutput
from reprpo.interventions.pl_base import PL_MODEL
from reprpo.interventions.helpers import compute_logprobs
import warnings
from typing import List
import re
from typing import Optional
from .helpers import get_layer_paths, validate_layer_paths
from ..losses import LossesType
from ..transforms import TransformType
from ..dpo import compute_dpo_loss


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
            f"Collection keys {collection_keys} do not match any layers in the model."
        )
    return out



# def get_default_layers(N, side_channels=False, stride: Optional[int] = None, include_last_two: bool = True):
#     """
#     Which layers to use? If we use all of them it will take lots of memory. For hidden space ones we don't need many. But for side channel ones we typically need many.

#     Simialr to [Circuit Breakers](https://arxiv.org/html/2406.04313v4), we use the layers at 33% and 66%. 
    
#     I also want the last two layers to enable supression neurons. 
    
#     """
#     N3 = int(N*0.33)
#     layers = list(np.arange(N3, N, 1 if stride is None else stride).astype(int)[1:-1]) # 33% and 66% as in Circuit Breakers
#     if include_last_two:
#         layers += [N-2, N-1] # make sure to include last two for supression neurons
#     if side_channels:
#         layers += list(range(int(N*0.33),int(N*0.66), 2 if stride is None else stride)) # for side channels we every 2nd use the middle 33% similar to the RepE setup in Circuit Breakers
#     return sorted(set(layers))

def reprpo_forward_baukit(
    model, input_ids, attn_mask, layer_paths, collect_input=True, collect_hs=False
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

    logprobs = compute_logprobs(
        logits=outs.logits, labels=input_ids, selection_mask=attn_mask
    )
    return ReprPOModelOutput(
        hs=reprs, logits=outs.logits, label_logprobs=logprobs, mask=attn_mask
    )

from typing import Union, List

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
        # FIXME no we need to turn float to ints first?
        # Safely evaluate the range string
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

    # FIXME unit tests range(0.3, -2, 2), "0.5,-2,-1" "1,2,-1"
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
        loss: LossesType,
        transform: TransformType,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hparams.loss = loss
        self.hparams.transform = transform
        self.hparams.collection_layers = collection_layers
        self.hparams.collect_input = collect_input
        self.hparams.collect_hs = collect_hs

        collection_keys = collection_keys_in if collect_input else collection_keys_out
        collection_keys = get_regexp_layers(
            collection_keys, self._model
        )

        # if collection_layers is None we collect the last 50% of layers


        N = self._model.config.num_hidden_layers
        if collection_layers is None:
            collection_layers = "range(0.3, 1)"
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
                    input_ids=batch["chosen"],
                    attn_mask=batch["chosen_mask"],
                    layer_paths=h.layer_paths,
                    collect_input=h.collect_input,
                    collect_hs=h.collect_hs,
                )
                ref_rej = reprpo_forward_baukit(
                    model=model,
                    input_ids=batch["rejected"],
                    attn_mask=batch["rejected_mask"],
                    layer_paths=h.layer_paths,
                    collect_input=h.collect_input,
                    collect_hs=h.collect_hs,
                )

        model.train()
        pi_cho = reprpo_forward_baukit(
            model=model,
            input_ids=batch["chosen"],
            attn_mask=batch["chosen_mask"],
            layer_paths=h.layer_paths,
            collect_input=h.collect_input,
            collect_hs=h.collect_hs,
        )
        pi_rej = reprpo_forward_baukit(
            model=model,
            input_ids=batch["rejected"],
            attn_mask=batch["rejected_mask"],
            layer_paths=h.layer_paths,
            collect_input=h.collect_input,
            collect_hs=h.collect_hs,
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

        # collect extra metrics
        with torch.no_grad():
            # get the dpo metrics for comparison
            _, info_dpo = compute_dpo_loss(
                pi_cho.label_logprobs,
                pi_rej.label_logprobs,
                ref_cho.label_logprobs,
                ref_rej.label_logprobs,
            )

            # measures preference for cho>ref compared to base model. Should increase
            info["logits"] = info_dpo["logits"]

            # measures if coherence has increased over ref model. Should be increase
            info["chosen_rewards"] = info_dpo["chosen_rewards"]

            def cosine_on_keys(hs1, hs2):
                return torch.stack(
                    [
                        F.cosine_similarity(hs1[k], hs2[k], dim=-1).nanmean()
                        for k in hs1.keys()
                    ]
                ).nanmean()

            # measure if chosen models are still close to the ref model, should stay at 1
            info["retain_cosine"] = cosine_on_keys(pi_cho.hs, ref_cho.hs)
            # measures if refject hs are getting close to cho, should rise towards 1
            info["rr_cosine"] = cosine_on_keys(pi_rej.hs, ref_cho.hs)

        return loss, {**info, **info_dpo}
