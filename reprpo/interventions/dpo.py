import torch
import torch.nn.functional as F
from reprpo.interventions.pl_base import PL_MODEL
from dataclasses import dataclass
from reprpo.interventions.config import ExperimentConfig
from .losses.helpers import cross_entropy_loss
from .helpers import compute_logprobs
from reprpo.interventions.losses.helpers import compute_ptheta


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
    logits = compute_ptheta(
        model_chosen_logprobs=model_chosen_logprobs,
        model_rejected_logprobs=model_rejected_logprobs,
        reference_chosen_logprobs=reference_chosen_logprobs,
        reference_rejected_logprobs=reference_rejected_logprobs,
    )

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(β * logits)

    # Optional values to track progress during training
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    dpo_acc = (model_logratios > 0).float()

    # .mean() to average over the samples in the batch
    return losses.mean(), dict(
        _chosen_rewards=chosen_rewards.mean(),
        _rejected_rewards=rejected_rewards.mean(),
        _dpo_loss=losses.mean(),
        _model_logratios=model_logratios.mean(),
        _reference_logratios=reference_logratios.mean(),
        _logits=logits.mean(),
        _dpo_acc=dpo_acc.mean(),
    )


def compute_dpo_loss_batch(batch, model, β=0.1):
    """Compute the DPO loss on an input batch"""

    model_kwargs = dict(
        use_cache=False,
        return_dict=True,
        output_hidden_states=True,
    )

    # FIXME: I need to mask out the prompt?

    model.eval()
    with torch.no_grad():
        with model.disable_adapter():
            ref_cho = model(
                batch["chosen"], attention_mask=batch["chosen_mask"], **model_kwargs
            )
            ref_chosen_log_probas = compute_logprobs(
                logits=ref_cho.logits,
                labels=batch["chosen"],
                selection_mask=batch["chosen_mask"],
            )
            ref_rej = model(
                batch["rejected"], attention_mask=batch["rejected_mask"], **model_kwargs
            )
            ref_rejected_log_probas = compute_logprobs(
                logits=ref_rej.logits,
                labels=batch["rejected"],
                selection_mask=batch["rejected_mask"],
            )

    model.train()
    pi_cho = model(batch["chosen"], attention_mask=batch["chosen_mask"], **model_kwargs)
    policy_chosen_log_probas = compute_logprobs(
        logits=pi_cho.logits,
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"],
    )
    pi_rej = model(
        batch["rejected"], attention_mask=batch["rejected_mask"], **model_kwargs
    )
    policy_rejected_log_probas = compute_logprobs(
        logits=pi_rej.logits,
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"],
    )

    loss, info = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        β=β,
    )

    # def cosine_on_keys(hs1, hs2):
    #     hs1 = collect_hs(hs1)
    #     hs2 = collect_hs(hs2)
    #     return F.cosine_similarity(hs1, hs2, dim=-1).nanmean()

    with torch.no_grad():
        # info['retain_cosine'] = cosine_on_keys(pi_cho.hidden_states, ref_cho.hidden_states)
        # info['rr_cosine'] = cosine_on_keys(pi_rej.hidden_states, ref_cho.hidden_states)

        nll_loss = info["nll_loss"] = cross_entropy_loss(
            pi_cho.logits, batch["chosen"], batch["chosen_mask"]
        ).mean()
        ref_nll_loss = info["ref_nll_loss"] = cross_entropy_loss(
            ref_cho.logits, batch["chosen"], batch["chosen_mask"]
        ).mean()
        info["nll_loss_ratio"] = (nll_loss / ref_nll_loss).mean()

    return loss, info


class PL_DPO_MODEL(PL_MODEL):
    def _loss_fn(self, batch, model):
        return compute_dpo_loss_batch(batch, model)


@dataclass
class DPOConfig(ExperimentConfig):
    lr: float = 6e-5
    # 5e-5 https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb
    # 5e-7 https://github.com/eric-mitchell/direct-preference-optimization/blob/main/config/config.yaml

    _cls = PL_DPO_MODEL

    _model_keys = ["lr"]

    @property
    def _name(self):
        return "dpo"
