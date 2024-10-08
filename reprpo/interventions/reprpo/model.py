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
import warnings
from .helpers import get_layer_paths, validate_layer_paths
from ..losses import Losses, LossesType
from ..transforms import Transforms, TransformType
from ..losses.helpers import cross_entropy_loss
from ..dpo import compute_dpo_loss


def reprpo_forward_baukit(model, input_ids, attn_mask, layer_paths, collect_input=True):
    # if the layer paths are just str(ints) then just collect the hidden states
    try:
        layer_paths = [int(p) for p in layer_paths]
        outs = model(
            input_ids,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        reprs = {str(k): outs.hidden_states[k] for k in layer_paths}
    except ValueError:
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
        if not torch.isfinite(reprs[p]).all():
            warnings.warn(
                f"gathered activations for layer [{p}] are not finite {reprs[p]}. This could be due to an high lr or unstable loss function."
            )
        # assert torch.isfinite(
        #     reprs[p]
        # ).all(), f"gathered activations for layer [{p}] are not finite {reprs[p]}. This could be due to an high lr or unstable loss function."

    logprobs = compute_logprobs(
        logits=outs.logits, labels=input_ids, selection_mask=attn_mask
    )
    return ReprPOModelOutput(
        hs=reprs, logits=outs.logits, label_logprobs=logprobs, mask=attn_mask
    )


class PL_REPRPO_MODEL(PL_MODEL):
    def __init__(
        self,
        *args,
        collection_layers_side,
        collect_input,
        collection_keys_in: tuple = None,
        collection_keys_out: tuple = None,
        loss_fn: LossesType,
        transform: TransformType,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hparams.loss_fn = loss_fn
        self.hparams.transform = transform
        self.hparams.collection_layers_side = collection_layers_side
        self.hparams.collect_input = collect_input

        collection_keys = collection_keys_in if collect_input else collection_keys_out

        # set layer_paths
        if len(collection_keys) > 0:
            self.hparams.layer_paths = get_layer_paths(
                collection_keys, collection_layers_side
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
            # if no collection keys, we collect hidden states instead
            # we generally need only a few so lets just take the first and last
            self.hparams.layer_paths = tuple(
                set([collection_layers_side[0], collection_layers_side[-1]])
            )
            self.hparams.layer_paths = [str(s) for s in self.hparams.layer_paths]
            hra_sizes = {
                k: self._model.config.hidden_size for k in self.hparams.layer_paths
            }

        self.transforms = torch.nn.ParameterDict(
            {k: transform.c(dim_hs, dim_hs, model=self._model) for k, dim_hs in hra_sizes.items()}
        )
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

        # run loss function
        loss, info = h.loss_fn.c(
            pi_cho=pi_cho,
            pi_rej=pi_rej,
            ref_cho=ref_cho,
            ref_rej=ref_rej,
            batch=batch,
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
            info['logits'] = info_dpo['logits']

            # measures if coherence has increased over ref model. Should be increase
            info['chosen_rewards'] = info_dpo['chosen_rewards']

            def cosine_on_keys(hs1, hs2):
                return torch.stack(
                    [
                        F.cosine_similarity(hs1[k], hs2[k], dim=-1).nanmean()
                        for k in hs1.keys()
                    ]
                ).nanmean()
            # measure if chosen models are still close to the ref model, should stay at 1
            info['retain_cosine'] = cosine_on_keys(pi_cho.hs, ref_cho.hs)
            # measures if refject hs are getting close to cho, should rise towards 1
            info['rr_cosine'] = cosine_on_keys(pi_rej.hs, ref_cho.hs)



        return loss, {**info, **info_dpo}
