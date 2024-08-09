from baukit.nethook import TraceDict
import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from torch import Tensor
from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union


from trl import DPOConfig, DPOTrainer
from reprpo.train.trainer import (
    ReprPOTrainer,
    ReprPOConfig,
    mean_with_attention,
    normalize_output,
)


def mean_with_attention(
    x: Float[Tensor, "b t h"], attn_mask: Float[Tensor, "b t"], dim: int = 1
) -> Float[Tensor, "b h"]:
    """mean of x, weighted by the attention mask, over dim (token or batch)"""
    layer_attn_mask = repeat(attn_mask, "b t -> b t h", h=1).detach()
    return (x * layer_attn_mask).sum(dim) / layer_attn_mask.sum(dim)


def detach_hsd(hs):
    return {k: v.detach() for k, v in hs.items()}


def get_layer_paths(args):
    layer_paths = [
        [p.format(layer=layer) for p in args.collection_keys]
        for layer in args.collection_layers
    ]
    layer_paths = list(itertools.chain(*layer_paths))
    return layer_paths


class ReprPOTrainerSideChannel(ReprPOTrainer):
    def __init__(self, args: Optional[ReprPOConfig] = None, **kwargs):
        DPOTrainer.__init__(self, args=args, **kwargs)
        self.collection_layers = args.collection_layers
        self.alpha = args.alpha
        self.loss_type = "ipo"

        self.num_training_steps = self.args.max_steps
        if self.num_training_steps == -1:
            self.num_training_steps = (
                self.args.num_train_epochs
                * len(self.get_train_dataloader())
                // self.args.gradient_accumulation_steps
            )

        self.layer_paths = get_layer_paths(args)
        print(self.layer_paths)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        model.eval()
        with torch.no_grad():
            with self.null_ref_context():
                (
                    ref_chosen_logps,
                    ref_rejected_logps,
                    _,
                    _,
                    _,
                    ref_chosen_hs,
                    ref_rejected_hs,
                    _,
                    _,
                ) = self.concatenated_forward(self.model, batch)
        ref_chosen_hs = detach_hsd(ref_chosen_hs)
        ref_rejected_hs = detach_hsd(ref_rejected_hs)
        ref_chosen_logps = ref_chosen_logps.detach()
        ref_rejected_logps = ref_rejected_logps.detach()

        model.train()
        (
            pi_chosen_logps,
            pi_rejected_logps,
            _,
            _,
            pi_chosen_logps_avg,
            pi_chosen_hs,
            pi_rejected_hs,
            chosen_attn_mask,
            rejected_attn_mask,
        ) = self.concatenated_forward(model, batch)

        loss, loss_info = self.reprpo_loss(
            pi_chosen_logps,
            pi_rejected_logps,
            pi_chosen_hs,
            pi_rejected_hs,
            ref_chosen_logps,
            ref_rejected_logps,
            ref_chosen_hs,
            ref_rejected_hs,
            chosen_attn_mask,
            rejected_attn_mask,
        )
        # losses, chosen_rewards, rejected_rewards, loss_retain, loss_rr = loss_info
        chosen_rewards, rejected_rewards = (
            loss_info["chosen_rewards"],
            loss_info["rejected_rewards"],
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            loss = loss * self.args.rpo_alpha - pi_chosen_logps_avg

        prefix = "eval_" if train_eval == "eval" else ""

        # how often the policy model is better at choosing the right response
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.nanmean().cpu()
        # how much the policy model is better
        metrics[f"{prefix}rewards/margins"] = (
            (chosen_rewards - rejected_rewards).nanmean().cpu()
        )

        # the log probability that the model would generate the tokens of the rejected string
        metrics[f"{prefix}logps/rejected"] = pi_rejected_logps.detach().nanmean().cpu()
        metrics[f"{prefix}logps/chosen"] = pi_chosen_logps.detach().nanmean().cpu()

        for k in loss_info.keys():
            if "_" in k:
                a, b = k.split("_", 1)
                k2 = f"{b}/{a}"
            else:
                k2 = k
            v = loss_info[k]
            if isinstance(v, torch.Tensor):
                v = v.nanmean().detach().cpu().item()
            metrics[f"{prefix}{k2}"] = float(v)

        if self.state.global_step % self.args.print_every == 0:

            def cosine_on_keys(hs1, hs2):
                return torch.stack(
                    [
                        F.cosine_similarity(hs1[k], hs2[k], dim=-1).nanmean()
                        for k in hs1.keys()
                    ]
                ).nanmean()

            retain_cosine = cosine_on_keys(pi_chosen_hs, ref_chosen_hs)
            rr_cosine = cosine_on_keys(pi_rejected_hs, ref_chosen_hs)

            metrics[f"{prefix}retain_cosine"] = retain_cosine
            metrics[f"{prefix}rr_cosine"] = rr_cosine

            print({k: f"{v:.4g}" for k, v in metrics.items()})

        return loss.nanmean(), metrics

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
            max_length=self.max_length,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop(
                    "concatenated_decoder_input_ids", None
                ),
            }
            if self.is_encoder_decoder
            else {}
        )

        reprs = {}
        with TraceDict(
            model,
            self.layer_paths,
            retain_input=self.args.collect_input,
            retain_output=(not self.args.collect_input),
            retain_grad=False,
        ) as ret:
            outs = model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
                **model_kwargs,
            )
            for p in self.layer_paths:
                if self.args.collect_input:
                    reprs[p] = ret[p].input
                else:
                    reprs[p] = ret[p].output
                assert torch.isfinite(reprs[p]).all()
            # print(reprs[p].shape, reprs[p].dtype)
        all_logits = outs.logits

        # # this includes prompt and padding
        # hs = collect_hs(outs.hidden_states)[:, self.collection_layers]
        # # del outs
        # # gc.collect()

        # multiply by attention mask
        attn_mask = concatenated_batch["concatenated_attention_mask"]

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        chosen_logps_avg = all_logps[:len_chosen] / size_completion[:len_chosen]

        # So we want sum of logprobs or mean of logprobs? Like IPO we will use the log prob per token, https://github.com/eric-mitchell/direct-preference-optimization/issues/40
        if self.loss_type == "ipo":
            all_logps = all_logps / size_completion
            # all_logps = torch.log(torch.exp(all_logps) / size_completion + 1e-12)
            # NOTE for some reason the model is still biased toward longer answers, even though this should neutralise it

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_hs = {k: hs[:len_chosen] for k, hs in reprs.items()}
        rejected_hs = {k: hs[len_chosen:] for k, hs in reprs.items()}

        chosen_attn_mask = attn_mask[:len_chosen]
        rejected_attn_mask = attn_mask[len_chosen:]

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_logps_avg,
            chosen_hs,
            rejected_hs,
            chosen_attn_mask,
            rejected_attn_mask,
        )

    def reprpo_loss(
        self,
        pi_chosen_logps: torch.FloatTensor,
        pi_rejected_logps: torch.FloatTensor,
        pi_cho_hs: torch.FloatTensor,
        pi_rej_hs: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        ref_cho_hs: torch.FloatTensor,
        ref_rej_hs: torch.FloatTensor,
        cho_attn_mask: torch.BoolTensor,
        rej_attn_mask: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            pi_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            pi_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            ref_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            ref_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """

        pi_logratios = pi_chosen_logps - pi_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor(
                [0], dtype=pi_logratios.dtype, device=pi_logratios.device
            )
        else:
            ref_logratios = ref_chosen_logps - ref_rejected_logps

        # log(prob_chosen/prob_rejected) the prob of the chosen strings over the rejected string. 0 is not difference. -ve means rejected is larger
        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = (
            pi_logratios - ref_logratios
        )  # was pi more likely to chose the correct response or the reference model

        # Can we weight by how much better the reference model was
        # in dpo we minimise it, so lower is better, here we are weighting it, so take the -ve to higher is more correct
        # NOTE: -logits is if pi is more correct than ref, and focuses on what model gets wrong, unstable, moving target
        # -ref_logratios is is the reference model lean toward correct, and is stable
        T = 2
        weight_correct = torch.softmax(-ref_logratios * T, 0).detach()

        def _dist_w_attn_mask(
            chosen_hs: Float[Tensor, "b t h"], rejected_hs: Float[Tensor, "b t h"], attn: Float[Tensor ,'b t'],
            eps=1e-12,
        ) -> Float[Tensor ,'b h']:
            assert torch.isfinite(chosen_hs).all()
            assert torch.isfinite(rejected_hs).all()
            dist = rejected_hs - chosen_hs
            assert torch.isfinite(dist).all()
            dist = mean_with_attention(dist, attn.detach()) # reduces over tokens
            assert torch.isfinite(dist).all()
            # loss_rr = symlog(loss_rr)
            # loss_rr = wmean(loss_rr, 1 - weight_correct)
            return (dist**2) + eps # maybe norm or abs?

        def dist_w_attn_mask(
                chosen_hs: Dict[str, Float[Tensor, "b t h"]], 
                rejected_hs, attn
            ) -> Float[Tensor ,'b l h']:
            dists = [
                _dist_w_attn_mask(chosen_hs[k], rejected_hs[k], attn)
                for k in chosen_hs.keys()
            ]
            return dists # torch.stack(dists, dim=1)

        comb_attn_mask = cho_attn_mask * rej_attn_mask

        hs_dist_cho2rej_pi2ref = dist_w_attn_mask(
            detach_hsd(ref_cho_hs), pi_rej_hs, comb_attn_mask
        )

        # the loss is small, express it as a fraction of the reference values
        hs_dist_cho2rej_ref2ref = dist_w_attn_mask(
            ref_cho_hs, ref_rej_hs, comb_attn_mask
        )

        # how much we've reduced the distance between the chosen and rejected responses, compared to reference model
        # reduce(a/b.detach(), 'b h -> b', 'nanmean')
        loss_reroute = torch.stack([
            reduce(a/b.detach(), 'b h -> b', torch.nanmean)
            for a, b in zip(hs_dist_cho2rej_pi2ref, hs_dist_cho2rej_ref2ref)], 1)
        # loss_reroute = (
        #     hs_dist_cho2rej_pi2ref / hs_dist_cho2rej_ref2ref.detach()
        # )

        # this loss measures how much the policy model has retained the information in the chosen responses, compared to the reference model
        hs_dist_cho2cho_pi2ref = dist_w_attn_mask(
            detach_hsd(ref_cho_hs), pi_cho_hs, cho_attn_mask
        )

        # scale it, so that it's expressed as a fraction of the dist between rej2cho on the ref model
        hs_dist_cho2rej_ref2ref = dist_w_attn_mask(
            ref_cho_hs, ref_rej_hs, comb_attn_mask
        )
        # +1 so it start on par with reroute loss, and we can see it diverge?? TODO revisit
        loss_retain = torch.stack([
            reduce(a/b.detach(), 'b h -> b', torch.nanmean)
        for a, b in zip(hs_dist_cho2cho_pi2ref, hs_dist_cho2rej_ref2ref)], 1)
        #     hs_dist_cho2cho_pi2ref / hs_dist_cho2rej_ref2ref.detach() + 1
        # )

        # Weightings
        c_retain, c_reroute = self.get_coeff()
        c_reroute = c_retain = 1
        loss = (
            loss_reroute.nanmean() * c_reroute
            + loss_retain.nanmean() * c_retain * self.alpha
        )

        # difference in logps for chosen responses, between policy and reference model
        # # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        chosen_rewards = (
            self.beta
            * (
                pi_chosen_logps.to(self.accelerator.device)
                - ref_chosen_logps.to(self.accelerator.device)
            ).detach()
        )

        # difference in logps for rejected responses, between policy and reference model
        rejected_rewards = (
            self.beta
            * (
                pi_rejected_logps.to(self.accelerator.device)
                - ref_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        loss_dict = dict(
            loss=loss,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            loss_retain=loss_retain.detach(),
            loss_reroute=loss_reroute.detach(),
            pi_logratios=pi_logratios.detach(),
            ref_logratios=ref_logratios.detach(),
            weighting=weight_correct.nanmean(),
            logits=logits.nanmean().detach(),
            loss_component_rr=(loss_reroute * c_reroute).detach().nanmean(),
            loss_component_retain=(loss_retain * c_retain * self.alpha)
            .detach()
            .nanmean(),
            c_rr=c_reroute,
            c_retain=c_retain,
        )

        loss_dict = {k: normalize_output(v) for k, v in loss_dict.items()}

        return loss, loss_dict


from dataclasses import dataclass


@dataclass
class ReprPOSideChannelConfig2Outs(DPOConfig):
    alpha: int = 1
    print_every: int = 20
    collection_layers: tuple = (11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22)
    collection_keys: tuple = (
        "base_model.model.model.layers.{layer}.self_attn.qkv_proj.base_layer",
        "base_model.model.model.layers.{layer}.mlp.gate_up_proj.base_layer",
    )
    collect_input: bool = False

    # NOTE to self, do not pass both peft_config and model_adapter_name. peft_config creates a new adapter


@dataclass
class ReprPOSideChannelkConfig2Ins(DPOConfig):
    alpha: int = 1
    print_every: int = 20
    collection_layers: tuple = (11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22)
    collection_keys: tuple = (
        "base_model.model.model.layers.{layer}.self_attn.o_proj",
        "base_model.model.model.layers.{layer}.mlp.down_proj",
    )
    collect_input: bool = True


def check_training_args(training_args, model):
    assert training_args.collection_layers is not None
    assert training_args.collection_keys is not None
    layer_paths = get_layer_paths(training_args)
    ps2 = [model.get_submodule(p) for p in layer_paths]

    # also if module in model.peft_config[adapter_name].target_modules

    # target_modules = model.peft_config[adapter_name].target_modules
    # print('target_modules', target_modules)
    # for p in ps:
    #     r = p.rsplit('.', 1)[-1]
    #     if r in target_modules:
    #         print(f"{r} in target_modules {target_modules}. You may want to append .base_layer")
    return training_args
