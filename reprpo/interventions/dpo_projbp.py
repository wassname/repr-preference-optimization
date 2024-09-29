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
from typing import Optional
from loguru import logger
from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear

from torch.autograd import Function
from .dpo_projgrad import hs_norm, ProjGradHooks, compute_gradproj_loss_batch


class ProjBPHooks(ProjGradHooks):
    def register_hooks(self):
        modules = self.enabled_modules()
        assert len(modules) > 0, "No modules found with requires_grad=True"
        for k, module in modules.items():
            module.register_full_backward_pre_hook(self.backward_prehook_projgrad)
            module.register_forward_hook(self.forward_hook_projgrad)
        logger.debug(f"ProjGrad Registered {len(modules)} hooks. {modules.keys()}")

# class GradProjFunction(Function):
#     @staticmethod
#     # ctx is the first argument to forward
#     def forward(ctx, input, ref_cho, ref_rej, β: float):
#         ctx.save_for_backward(input, ref_cho, ref_rej)
#         ctx.β = β # https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html
#         return input

#     @staticmethod
#     def backward(ctx, grad_output):
#         print(grad_output.norm())
#         input, ref_cho, ref_rej = ctx.saved_tensors
#         # now take only the preferred direction
#         eps = 1e-12
#         preference_dir = ref_cho-ref_rej
#         pref_dir_unit = preference_dir / preference_dir.norm(dim=-1, keepdim=True).clamp(eps)
        
#         # get projection of `grad` along ref_dir
#         # grad is 'b t h'
#         grad_proj_onto_pref = (pref_dir_unit * grad_output).sum(dim=-1, keepdim=True) * pref_dir_unit
#         grad_orthogonal = grad_output - grad_proj_onto_pref

#         grad = grad_proj_onto_pref.clamp(0, None) + ctx.β * grad_orthogonal
#         # print(grad_output.norm(), grad_proj_onto_pref.norm(), grad_orthogonal.norm())

#         # only use that part
#         return grad, None, None, None
    

def backward_prehook_projgrad(module, grad_output):
    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook

    ref_cho = module._cache['ref_cho']
    if isinstance(ref_cho, tuple):
        ref_cho = ref_cho[0]
    
    ref_rej = module._cache['ref_rej']
    if isinstance(ref_rej, tuple):
        ref_rej = ref_rej[0]
    
    # sum over tokens either using mask [b t h] -> [b h]
    mask = module._cache['ref_cho_mask'].unsqueeze(-1)
    ref_cho = (ref_cho * mask).sum(1) / mask.sum(1)
    mask = module._cache['ref_rej_mask'].unsqueeze(-1)
    ref_rej = (ref_rej * mask).sum(1) / mask.sum(1)

    # now take only the preferred direction
    eps = 1e-12
    preference_dir = ref_cho-ref_rej
    pref_dir_unit = preference_dir / preference_dir.norm(dim=-1, keepdim=True).clamp(eps)
    
    # get projection of `grad` along ref_dir
    # grad is 'b t h'
    def proj_grad(grad):
        grad_proj_onto_pref = (pref_dir_unit * grad).sum(dim=-1, keepdim=True) * pref_dir_unit
        grad_orthogonal = grad - grad_proj_onto_pref

        grad2 = grad_proj_onto_pref.clamp(0, None) + module.β * grad_orthogonal
        # print(1, grad.norm(), grad_proj_onto_pref.norm(), grad_orthogonal.norm())
        grad2 = grad2 * grad.norm(dim=1) / grad2.norm(dim=1)

        # TODO scale the magnitude up to the original grad
        return grad2
    res = tuple(proj_grad(g) for g in grad_output)
    # print(2,module)
    # print(3,[aa.shape for aa in grad_output], [aa.shape for aa in res])


    return grad_output



class PL_GradBP_MODEL(PL_MODEL):
    def __init__(self, *args, β, reverse_pref, scale_orth, neg_slope, mag_clip, weight_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.projgrad = ProjGradHooks(self._model, β=β, reverse_pref=reverse_pref, scale_orth=scale_orth, neg_slope=neg_slope, mag_clip=mag_clip, weight_dim=weight_dim)

    def _loss_fn(self, batch, model):
        return compute_gradproj_loss_batch(batch, model, self.projgrad)
    

@dataclass
class ProjBPConfig(ExperimentConfig):
    """
    Project gradient onto a preference direction during backprop. This means that later layer will change the gradint for earleir layers.
    """
    lr: float = 5e-5
    # 5e-5 https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb
    # 5e-7 https://github.com/eric-mitchell/direct-preference-optimization/blob/main/config/config.yaml

    β: float = 0.

    reverse_pref: bool = True
    """
    reverse the preference direction
    """

    weight_dim: int = 0
    """
    1 for forward, 0 for back, 2 for both
    """

    scale_orth: bool = True
    """
    scale the orthogonal movement to be β proportion of the prefered direction movement
    """

    neg_slope: float = 0.5
    """
    When clipping the gradient in the negative preference direction, we can use leakly relu with this slope. 0 is relu. 0.01 is leaky relu.
    """

    mag_clip: Optional[float] = None
    """
    Clip the magnitude of the gradient in the hs space, to ensure a proximal policy in hs space. Value is a fraction of the distance of the preference vector
    """

    _cls = PL_GradBP_MODEL

    _model_keys = ["lr", "β", "reverse_pref", "scale_orth", "neg_slope", "mag_clip", "weight_dim"]

    @property
    def _name(self):
        return "projbp"
