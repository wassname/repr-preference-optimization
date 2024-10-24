import torch
import torch.nn.functional as F

import os
from baukit.nethook import TraceDict, get_module

from reprpo.interventions.types import ReprPOModelOutput
from reprpo.interventions.pl_base import PL_MODEL
from reprpo.interventions.helpers import compute_logprobs
import warnings
from .helpers import get_layer_paths, validate_layer_paths
from ..losses import LossesType
from ..transforms import TransformType
from ..dpo import compute_dpo_loss


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


class PL_REPRPO_MODEL(PL_MODEL):
    def __init__(
        self,
        *args,
        collection_layers_side,
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
        self.hparams.collection_layers_side = collection_layers_side
        self.hparams.collect_input = collect_input
        self.hparams.collect_hs = collect_hs

        collection_keys = collection_keys_in if collect_input else collection_keys_out

        # if collection_layers_side is None we collect the last 50% of layers
        if collection_layers_side is None:
            N = self._model.config.num_hidden_layers
            collection_layers_side = list(range(N//2, N))

        # turn negative numbers into offsets from the end
        collection_layers_side = [i if i >= 0 else self._model.config.num_hidden_layers + i for i in collection_layers_side]

        # set layer_paths
        if not collect_hs:
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
            self.hparams.layer_paths = [str(s) for s in collection_layers_side]
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
