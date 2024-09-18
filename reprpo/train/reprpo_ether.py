import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from einops import rearrange, repeat, reduce
import math
import warnings

from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from reprpo.train.pl_base import PL_MODEL, TrainingArgumentswCollection, cross_entropy_loss
from reprpo.train.dpo import compute_logprobs, compute_dpo_loss
from types import SimpleNamespace
from baukit.nethook import TraceDict
from dataclasses import dataclass
import itertools
from ..layers.ether import ETHERLinear, ETHERLinearSmall, _ETHERConfig
from .reprpo_hra import compute_reprpo_hra_loss_batch


class PL_REPRPO_ETHER_MODEL(PL_MODEL):
    def __init__(self, *args, alpha, collection_layers, nb, Htype, ether_dropout, flip_side, rel_loss=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        self.hparams.collection_layers = collection_layers

        dim_hs = self._model.config.hidden_size
        self.transform = ETHERLinear(dim_hs, dim_hs, nb=nb,
                                                      Htype=Htype,
                                                      ether_dropout=ether_dropout,
                                                      flip_side=flip_side,)

    def _loss_fn(self, batch, model):
        return compute_reprpo_hra_loss_batch(
            batch,
            model,
            self.hparams.alpha,
            self.hparams.collection_layers,
            self.transform,
            rel_loss=self.hparams.rel_loss,
        )


@dataclass
class ETHER(_ETHERConfig, TrainingArgumentswCollection):
    """
    Transform: ETHER along which to reroute the hidden states associated with the rejected responses. (see https://arxiv.org/pdf/2405.20271v1)
    """
    
    alpha: int = 0.001
    """balancing retrain and reroute losses"""

    Htype: str = 'etherplus'

    nb: int = 32

    collection_layers: tuple = (10, 20)
    """The layers to collect the hidden states from. HRA operates on the residual stream so only needs a couple of points of collection"""

    rel_loss: bool = True

    _reprpo_class = PL_REPRPO_ETHER_MODEL

    _model_keys = ['alpha', 'collection_layers',  'nb', 'Htype', 'ether_dropout', 'flip_side', 'rel_loss',]
