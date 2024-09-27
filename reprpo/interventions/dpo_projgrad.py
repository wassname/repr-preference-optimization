import torch
from torch import nn
import torch.nn.functional as F

# https://huggingface.co/docs/peft/developer_guides/custom_models
from loguru import logger
from peft.tuners.tuners_utils import BaseTunerLayer

from dataclasses import dataclass
from contextlib import contextmanager

from typing import Optional
from einops import reduce, rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor

from reprpo.interventions.pl_base import PL_MODEL
from reprpo.interventions.config import ExperimentConfig
from .losses.helpers import cross_entropy_loss
from .helpers import compute_logprobs
from .dpo import compute_dpo_loss




class ProjGradHooks:
    def __init__(self, model, β=0.1, reverse_pref=False, ignore_direction=False, negative_slope=0.0, magnitude_clip=None):
        self.model = model
        self.β = β
        self.reverse_pref = reverse_pref
        self.ignore_direction = ignore_direction
        self.negative_slope = negative_slope
        self.magnitude_clip = magnitude_clip
        self.register_hooks()

        logger.info(f'β={β}')

    def enabled_modules(self):
        """Yield peft modules that have requires_grad=True"""
        modules = {}
        for module_name, module in self.model.named_modules():

            # only peft layers
            if not isinstance(module, (BaseTunerLayer,)):
                continue

            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    modules[module_name] = module
                    break

        return modules      

    def register_hooks(self):
        modules = self.enabled_modules()
        assert len(modules) > 0, "No modules found with requires_grad=True"
        for k, module in modules.items():
            module.register_full_backward_pre_hook(self.backward_prehook_projgrad)
            module.register_forward_hook(self.forward_hook_projgrad)
        logger.info(f"ProjGrad Registered {len(modules)} hooks. {modules.keys()}")

    def backward_prehook_projgrad(self, module, grad_output):
        eps = 1e-12
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
        ref_cho = module._cache['ref_cho']
        if isinstance(ref_cho, tuple):
            ref_cho = ref_cho[0]
        if module._cache['ref_cho_mask'] is not None:
            ref_cho = ref_cho * module._cache['ref_cho_mask'].unsqueeze(-1)
        ref_rej = module._cache['ref_rej']
        if isinstance(ref_rej, tuple):
            ref_rej = ref_rej[0]
        if module._cache['ref_rej_mask'] is not None:
            ref_rej = ref_rej * module._cache['ref_rej_mask'].unsqueeze(-1)
        
        # now take only the preferred direction in hs-space
        preference_dir = (ref_cho.mean(1) - ref_rej.mean(1)).unsqueeze(1)
        if self.reverse_pref:
            preference_dir *= -1

        def _norm(x, dims):
            return (x ** 2).sum(dims).sqrt()

        def hs_norm(t: Float[Tensor, 'b t h']) -> Float[Tensor, 'b t h']:
            # we mean over tokens, as  the dimensions are not comparable
            # and norm over hs
            t = reduce(t, 'b t h -> b h', 'mean')
            t = reduce(t, 'b h -> b', _norm)
            return repeat(t, 'b -> b t h', t=1, h=1)
        
        pref_dir_unit = preference_dir / hs_norm(preference_dir).clamp(eps)
        
        # get projection of `grad` along ref_dir in hs space
        # grad is typically 'batch token hs'
        # FIXME, do we want to use the mask, and also mean by token?
        # pref_dir_unit = pref_dir_unit.mean(dim=1, keepdim=True)

        def proj_grad(grad: Float[Tensor, 'b t h']) -> Float[Tensor, 'b t h']:

            # magnitude_clip, similar to PPO, we limit the update in hs-space. Simlar to PPO which does it per logit, we do it per hs
            if self.magnitude_clip is not None:
                # per sampler
                # ratios = hs_norm(grad)/(hs_norm(preference_dir).clamp(eps)*self.magnitude_clip)
                # per hs?
                ratios = (grad.mean(1).abs()/(self.magnitude_clip*preference_dir.mean(1).abs().clamp(eps))).unsqueeze(1)
                ratios = ratios.clamp(1, None)
                grad = grad / ratios

            grad_proj_onto_pref = (pref_dir_unit * grad).sum(dim=-1, keepdim=True) * pref_dir_unit
            grad_orthogonal = grad - grad_proj_onto_pref

            if self.ignore_direction:
                grad_proj_onto_pref = torch.abs(grad_proj_onto_pref)

            # only take the part that goes in the direction we want, and some portion of the orthogonal movement
            grad2 = F.leaky_relu(grad_proj_onto_pref, negative_slope=self.negative_slope) + self.β * grad_orthogonal

            return grad2
        res = tuple(proj_grad(g) for g in grad_output)
        return res

    def forward_hook_projgrad(self, m, inputs, output):
        if not hasattr(m, "_cache"):
            m._cache = {}
        
        if m._mode != "normal":
            # FIXME ideally mask the hs
            assert m._mode in ["ref_cho", "ref_rej"]
            m._cache[m._mode+'_mask'] = m._mask.detach()
            m._cache[m._mode] = output # (o.detach() for o in output)
        elif m._mode == "normal":
            assert ('ref_cho' in m._cache and 'ref_rej' in m._cache)
        elif m._mode == "clear":
            m._cache = {}
        else:
            raise ValueError(f"Invalid mode {m._mode}")
        return output


    @contextmanager
    def set_projgrad_mode(self, mode: str, mask=None):

        # TODO consider temp hooks https://github.com/huggingface/peft/blob/ccc350151f95a9ff95da046bae5671da75eab52f/src/peft/tuners/lora/model.py#L438 https://github.com/davidbau/baukit/blob/main/baukit/nethook.py
        # but it wont work with ba

        modules = self.enabled_modules()
        assert len(modules) > 0, "No modules found with requires_grad=True"
        for k, module in modules.items():
            module._old_mode = getattr(module, '_mode', "normal") + ""
            module._mode = mode
            module._mask = mask
        yield self.model
        for k, module in modules.items():
            module._mode = module._old_mode
            module._mask = None

def compute_gradproj_loss_batch(batch, model, projgrad, β=0.1):
    """Compute the DPO loss on an input batch"""

    model_kwargs = dict(
        use_cache=False,
        return_dict=True,
        output_hidden_states=True,
    )
    # with projgrad.set_projgrad_mode("clear"):
    #     pass

    model.eval()
    with projgrad.set_projgrad_mode("ref_cho", batch["chosen_mask"]):
        with model.disable_adapter():
            with torch.no_grad():
                ref_cho = model(
                    batch["chosen"], attention_mask=batch["chosen_mask"], **model_kwargs
                )
                ref_chosen_log_probas = compute_logprobs(
                    logits=ref_cho.logits,
                    labels=batch["chosen"],
                    selection_mask=batch["chosen_mask"],
                )
    with projgrad.set_projgrad_mode("ref_rej", batch["rejected_mask"]):
        with model.disable_adapter():
            with torch.no_grad():
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


class PL_ProjGrad_MODEL(PL_MODEL):
    def __init__(self, *args, β, reverse_pref, ignore_direction, negative_slope, magnitude_clip, **kwargs):
        super().__init__(*args, **kwargs)
        self.projgrad = ProjGradHooks(self._model, β=β, reverse_pref=reverse_pref, ignore_direction=ignore_direction, negative_slope=negative_slope, magnitude_clip=magnitude_clip)


    def _loss_fn(self, batch, model):
        return compute_gradproj_loss_batch(batch, model, self.projgrad)


@dataclass
class DPOProjGradConfig(ExperimentConfig):
    """
    This is DPO, with a hard constrain on the gradient, so only move in the preference direction in hidden-state space

    It takes all the Lora layers, and on the backward pass it clip the gradient to only move in the cho_hs-ref_hs vector
    """

    lr: float = 5e-5
    # 5e-5 https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb
    # 5e-7 https://github.com/eric-mitchell/direct-preference-optimization/blob/main/config/config.yaml

    β: float = 1.0

    reverse_pref: bool = False
    """
    reverse the preference direction
    """

    ignore_direction: bool = False
    """
    ignore the preference direction, and just move in the hidden-state space.
    """

    negative_slope: float = 0.0
    """
    When clipping the gradient in the negative preference direction, we can use leakly relu with this slope. 0 is relu. 0.01 is leaky relu.
    """

    magnitude_clip: Optional[float] = None
    """
    Clip the magnitude of the gradient in the hs space, to ensure a proximal policy in hs space. Value is a fraction of the distance of the preference vector
    """

    _cls = PL_ProjGrad_MODEL

    _model_keys = ["lr", "β", "reverse_pref", "ignore_direction", "negative_slope", "magnitude_clip"]

    @property
    def _name(self):
        return "projgrad"
