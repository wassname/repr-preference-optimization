import torch
import torch.nn.functional as F
from reprpo.interventions.pl_base import PL_MODEL
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional
from reprpo.interventions.config import ExperimentConfig
from .dpo_helpers import cross_entropy_loss, compute_ptheta, compute_logprobs, compute_policy_weights
from reprpo.interventions.types import ReprPOModelOutput

def compute_dpo_loss(
    model_chosen_logprobs,
    model_rejected_logprobs,
    reference_chosen_logprobs,
    reference_rejected_logprobs,
    β=0.1,
):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        β: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. We ignore the reference model as β -> 0.
        label_smoothing: conservativeness for DPO loss.

    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
    """

    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    # logits = model_logratios - reference_logratios
    ptheta = compute_ptheta(
        pi_cho_logp=model_chosen_logprobs,
        pi_rej_logp=model_rejected_logprobs,
        ref_cho_logp=reference_chosen_logprobs,
        ref_rej_logp=reference_rejected_logprobs,
    )

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    # Note these are other options 
    losses = -F.logsigmoid(β * ptheta)

    # Optional values to track progress during training
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    dpo_acc = (model_logratios > 0).float()

    # .mean() to average over the samples in the batch
    return losses, dict(
        _chosen_rewards=chosen_rewards.mean(),
        _rejected_rewards=rejected_rewards.mean(),
        _dpo_loss=losses.mean(),
        _model_logratios=model_logratios.mean(),
        _reference_logratios=reference_logratios.mean(),
        _ptheta=ptheta.mean(),
        _dpo_acc=dpo_acc.mean(),
    )

def model_forward_with_logprobs(model, input_ids, attention_mask, prompt_mask=None, special_tokens_mask=None, dpo_agg_type="ipo", return_dict=True, output_hidden_states=True, **kwargs):
    """Forward pass through the model that returns extras."""
    outs = model(input_ids, attention_mask=attention_mask, return_dict=return_dict,
            output_hidden_states=output_hidden_states, **kwargs)

    if prompt_mask is not None:
        attention_mask = attention_mask * (1-prompt_mask)

    if special_tokens_mask is not None:
        attention_mask = attention_mask * (1-special_tokens_mask)

    # Compute log probabilities
    out_lp = compute_logprobs(
        logits=outs.logits,
        input_ids=input_ids,
        selection_mask=attention_mask,
        dpo_agg_type=dpo_agg_type,
    )
    hs = {k: v for k,v in enumerate(outs.hidden_states)} if output_hidden_states else None
    return ReprPOModelOutput(
        hs=hs, logits=outs.logits, label_logprobs=out_lp['label_logp'], mask=attention_mask, log_policy_weights=out_lp['log_policy_weights'],
    )


def dpo_forward_batch(batch, model, β=0.1, use_policy_weights=False, dpo_agg_type="ipo"):
    """Compute the DPO loss on an input batch"""

    model_kwargs = dict(
        use_cache=False,
        return_dict=True,
        output_hidden_states=True,
    )

    model.eval()
    with torch.no_grad():
        with model.disable_adapter():
            ref_cho = model_forward_with_logprobs(
                model, input_ids=batch["chosen_ids"], attention_mask=batch["chosen_mask"], prompt_mask=batch["prompt_mask"], dpo_agg_type=dpo_agg_type, **model_kwargs
            )
            ref_rej = model_forward_with_logprobs(
                model, input_ids=batch["rejected_ids"], attention_mask=batch["rejected_mask"], prompt_mask=batch["prompt_mask"], dpo_agg_type=dpo_agg_type, **model_kwargs
            )

    model.train()
    pi_cho = model_forward_with_logprobs(
        model, input_ids=batch["chosen_ids"], attention_mask=batch["chosen_mask"], prompt_mask=batch["prompt_mask"], dpo_agg_type=dpo_agg_type, **model_kwargs
    )
    pi_rej = model_forward_with_logprobs(
        model, input_ids=batch["rejected_ids"], attention_mask=batch["rejected_mask"], prompt_mask=batch["prompt_mask"], dpo_agg_type=dpo_agg_type, **model_kwargs
    )

    return calc_dpo_loss_w_metrics(
        batch,
        pi_cho,
        pi_rej,
        ref_cho,
        ref_rej,
        β=β,
        use_policy_weights=use_policy_weights,
    )

def calc_dpo_loss_w_metrics(batch, pi_cho: ReprPOModelOutput, pi_rej: ReprPOModelOutput, ref_cho: ReprPOModelOutput, ref_rej: ReprPOModelOutput, β=0.1, use_policy_weights=False):
    """Compute the DPO loss for a batch of policy and reference model log probabilities."""

    loss, info = compute_dpo_loss(
        model_chosen_logprobs=pi_cho.label_logprobs,
        model_rejected_logprobs=pi_rej.label_logprobs,
        reference_chosen_logprobs=ref_cho.label_logprobs,
        reference_rejected_logprobs=ref_rej.label_logprobs,
        β=β,
    )

    policy_weights = compute_policy_weights(pi_cho, pi_rej)
    info["policy_weights"] = policy_weights.mean()
    if use_policy_weights:
        loss = loss * policy_weights.detach()
    
    def cosine_on_hs(hs1: Dict[str, torch.Tensor], hs2: Dict[str, torch.Tensor]):
        """Compute the cosine similarity between two sets of hidden states. Which are lists of tensors from each layer"""
        cosines = []
        for k in hs1.keys():
            cos = F.cosine_similarity(hs1[k], hs2[k], dim=-1).nanmean()
            cosines.append(cos)
        return torch.stack(cosines).mean()

    # compute some metrics

    with torch.no_grad():
        info['_cosine_pi_cho_2_ref_cho'] = cosine_on_hs(pi_cho.hs, ref_cho.hs)
        info['_cosine_pi_rej_2_ref_rej'] = cosine_on_hs(pi_rej.hs, ref_rej.hs)
        info['_cosine_pi_rej_2_ref_cho'] = cosine_on_hs(pi_rej.hs, ref_cho.hs)
        info['_cosine_pi_cho_2_ref_rej'] = cosine_on_hs(pi_cho.hs, ref_rej.hs)

        nll_loss = info["_nll_loss"] = cross_entropy_loss(
            pi_cho.logits, batch["chosen_ids"], batch["chosen_mask"]
        ).mean()
        ref_nll_loss = info["_ref_nll_loss"] = cross_entropy_loss(
            ref_cho.logits, batch["chosen_ids"], batch["chosen_mask"]
        ).mean()
        info["_ref_nll_loss"] = ref_nll_loss
        info["_nll_lratio"] = (nll_loss - ref_nll_loss).mean()

    return loss.mean(), info


class PL_DPO_MODEL(PL_MODEL):
    def __init__(
        self,
        *args,
        use_policy_weights=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hparams.use_policy_weights = use_policy_weights
    
    def _loss_fn(self, batch, model):
        return dpo_forward_batch(batch, model, use_policy_weights=self.hparams.use_policy_weights)


@dataclass
class DPOConfig(ExperimentConfig):
    # 5e-5 https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb
    # 5e-7 https://github.com/eric-mitchell/direct-preference-optimization/blob/main/config/config.yaml
    
    use_policy_weights: bool = False
    """# Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827"""

    _cls = PL_DPO_MODEL

    _model_keys = ["use_policy_weights"]


    @property
    def _name(self):
        return "dpo"
