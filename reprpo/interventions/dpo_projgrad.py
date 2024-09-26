import torch
import torch.nn.functional as F
from reprpo.interventions.pl_base import PL_MODEL
from dataclasses import dataclass
from reprpo.interventions.config import ExperimentConfig
from .losses.helpers import cross_entropy_loss
from .helpers import compute_logprobs
from .dpo import compute_dpo_loss

# https://huggingface.co/docs/peft/developer_guides/custom_models
from torch import nn

from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear

class ProjGradLinear(LoraLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = "normal"
        self._cache = {}

    def forward(self, x):
        h = super().forward(x)
        if self._mode != "normal":
            assert self._mode in ["cho", "rej"]
            self._cache[self._mode] = h.detach()

    def custom_backward(self, grad_output):
        grad = super().backward(grad_output)

        # now take only the preferred direction
        eps = 1e-12
        c = self._cache["ref_cho"]
        r = self._cache["ref_rej"]
        pref_dir = c-r
        pref_dir_unit = pref_dir / pref_dir.norm(dim=-1, keepdim=True).clamp(eps)
        
        # get projection of `a` along ref_dir
        grad_proj = (pref_dir_unit * grad).sum(dim=-1, keepdim=True) * pref_dir_unit
        grad_orth = grad - grad_proj

        # free cach
        self._cache = {}

        # only use that part
        return grad_proj

from contextlib import contextmanager

@contextmanager
def set_projgrad_mode(model: nn.Module, mode: str):
    found = False
    for module in model.modules():
        if isinstance(module, ProjGradLinear):
            module._old_mode = module._mode
            module._mode = mode
            found = True
    assert found, "No ProjGradLinear found in model"
    yield model
    for module in model.modules():
        if isinstance(module, ProjGradLinear):
            module._mode = module._old_mode





def compute_gradproj_loss_batch(batch, model, β=0.1):
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
            with set_projgrad_mode(model, "ref_cho"):
                ref_cho = model(
                    batch["chosen"], attention_mask=batch["chosen_mask"], **model_kwargs
                )
                ref_chosen_log_probas = compute_logprobs(
                    logits=ref_cho.logits,
                    labels=batch["chosen"],
                    selection_mask=batch["chosen_mask"],
                )
            with set_projgrad_mode(model, "ref_rej"):
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


class PL_GradProj_MODEL(PL_MODEL):
    def _loss_fn(self, batch, model):
        return compute_gradproj_loss_batch(batch, model)
    

    @staticmethod
    def setup_grad_proj(config: LoraConfig):
        custom_module_mapping = {nn.Linear: ProjGradLinear}
        # register the new mapping
        config._register_custom_module(custom_module_mapping)
        return config


@dataclass
class DPOProjGradConfig(ExperimentConfig):
    lr: float = 5e-5
    # 5e-5 https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb
    # 5e-7 https://github.com/eric-mitchell/direct-preference-optimization/blob/main/config/config.yaml

    _cls = PL_GradProj_MODEL

    _model_keys = ["lr"]

    @property
    def _name(self):
        return "dpo"
