import torch
import torch.nn.functional as F
from reprpo.train.pl_base import PL_MODEL
from dataclasses import dataclass
from .pl_base import TrainingArguments, cross_entropy_loss
from einops import rearrange




def collect_hs(hs):
    """The residual stream or hs of the diff of the hs."""
    hs = rearrange(list(hs), "l b t h -> l b t h")
    return rearrange(hs, "l b t h -> b l t h")

def compute_logprobs(logits, labels, selection_mask=None):
    """
    Compute log probabilities.

    Args:
      logits: Tensor of shape (batch_size, num_tokens, vocab_size)
      labels: Tensor of shape (batch_size, num_tokens)
      selection_mask: Tensor for shape (batch_size, num_tokens)

    Returns:
      mean_log_prob: Mean log probability excluding padding tokens.
    """

    # Labels are the inputs shifted by one
    labels = labels[:, 1:].clone()

    # Truncate logits to match the labels num_tokens
    logits = logits[:, :-1, :]

    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the actual labels
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone()

        # Apply the mask to filter out padding tokens
        selected_log_probs = selected_log_probs * mask

        # Calculate the average log probability excluding padding tokens
        # This averages over the tokens, so the shape is (batch_size, num_tokens)
        avg_log_prob = selected_log_probs.sum(-1) #/ mask.sum(-1)

        return avg_log_prob

    else:
        return selected_log_probs.mean(-1)



def compute_dpo_loss(
      model_chosen_logprobs,
      model_rejected_logprobs,
      reference_chosen_logprobs,
      reference_rejected_logprobs,
      beta=0.1,
    ):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss.

    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
    """

    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(beta * logits)

    # Optional values to track progress during training
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    # .mean() to average over the samples in the batch
    return losses.mean(), dict(
        chosen_rewards=chosen_rewards.mean(), rejected_rewards=rejected_rewards.mean(),
        # model_logratios=model_logratios.mean(), reference_logratios=reference_logratios.mean(),
        # logits=logits.mean()
    )


def compute_dpo_loss_batch(batch, model, beta=0.1):
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
                batch["chosen"], 
                attention_mask=batch["chosen_mask"], 
                **model_kwargs
            )
            ref_chosen_log_probas = compute_logprobs(
                logits=ref_cho.logits,
                labels=batch["chosen"],
                selection_mask=batch["chosen_mask"]
            )
            ref_rej = model(batch["rejected"], attention_mask=batch["rejected_mask"],**model_kwargs)
            ref_rejected_log_probas = compute_logprobs(
                logits=ref_rej.logits,
                labels=batch["rejected"],
                selection_mask=batch["rejected_mask"]
            )
    
    model.train()
    pi_cho = model(batch["chosen"], attention_mask=batch["chosen_mask"],**model_kwargs)
    policy_chosen_log_probas = compute_logprobs(
        logits=pi_cho.logits,
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"]
    )
    pi_rej = model(batch["rejected"], attention_mask=batch["rejected_mask"],**model_kwargs)
    policy_rejected_log_probas = compute_logprobs(
        logits=pi_rej.logits,
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"]
    )

    loss, info = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta
    )
    
    # def cosine_on_keys(hs1, hs2):
    #     hs1 = collect_hs(hs1)
    #     hs2 = collect_hs(hs2)
    #     return F.cosine_similarity(hs1, hs2, dim=-1).nanmean()
    
    with torch.no_grad():
        # info['retain_cosine'] = cosine_on_keys(pi_cho.hidden_states, ref_cho.hidden_states)
        # info['rr_cosine'] = cosine_on_keys(pi_rej.hidden_states, ref_cho.hidden_states)

        nll_loss = info['nll_loss'] = cross_entropy_loss(pi_cho.logits, batch["chosen"])
        ref_nll_loss = info['ref_nll_loss'] = cross_entropy_loss(ref_cho.logits, batch["chosen"])
        info['nll_loss_ratio'] = nll_loss / ref_nll_loss

    
    return loss, info

class PL_DPO_MODEL(PL_MODEL):
    def _loss_fn(self, batch, model):
        return compute_dpo_loss_batch(batch, model)

@dataclass
class DPO(TrainingArguments):
    lr: float = 6e-5

    _reprpo_class = PL_DPO_MODEL
    _model_keys = []
