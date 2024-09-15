import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from torch import Tensor
from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from reprpo.train.pl_base import PL_MODEL, TrainingArguments, cross_entropy_loss
from reprpo.train.dpo import compute_logprobs, compute_dpo_loss
from types import SimpleNamespace
from baukit.nethook import TraceDict, get_module
from dataclasses import dataclass
import itertools

from ..layers.ether import ETHERLinear
from .reprpo_hra import HRA
from .reprpo_side import Sidein, Sideout, get_layer_paths, validate_layer_paths
from .reprpo_side_hra import compute_reprpo_side_hra_loss_batch



class PL_REPRPO_SIDE_ETHER_MODEL(PL_MODEL):
    def __init__(self, *args, alpha, collection_layers, 
                 nb, Htype, ether_dropout, flip_side, 
                 collect_input, collection_keys_in: list=None, collection_keys_out: list=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams.alpha = alpha
        collection_keys = collection_keys_in if collect_input else collection_keys_out
        self.hparams.layer_paths = get_layer_paths(collection_keys, collection_layers)
        validate_layer_paths(self._model, self.hparams.layer_paths)
        self.hparams.collect_input = collect_input

        # we do one learnable householder roation per layer
        if collect_input:
            hra_sizes = {p:get_module(self._model, p).in_features for p in self.hparams.layer_paths}
        else:
            hra_sizes = {p:get_module(self._model, p).out_features for p in self.hparams.layer_paths}
        self.transforms = torch.nn.ParameterDict({k: 
                                                  ETHERLinear(
                                                      dim_hs, dim_hs, 
                                                      nb=nb,
                                                      Htype=Htype,
                                                      ether_dropout=ether_dropout,
                                                      flip_side=flip_side,
                                                      ) for k,dim_hs in hra_sizes.items()})
        self.transforms = self.transforms.to(self._model.dtype).to(self._model.device)
        # TODO check dtype etc

    def _loss_fn(self, batch, model):
        return compute_reprpo_side_hra_loss_batch(
            batch,
            model,
            self.hparams.layer_paths,
            self.hparams.alpha,
            collect_input=self.hparams.collect_input,
            transforms=self.transforms
        )


@dataclass
class ETHER:
    """ETHER parameters"""

    nb: int = 32
    """number of diagonal blocks"""

    Htype: Literal['ether', 'etherplus', 'oft', 'etherplusHH'] = 'etherplus'
    """type of transformation 

    ether: like HRA but allowing a negative unit vector (reflection)
    etherplus: relaxing distance and orthogonality constraints
    oft: Orthogonal Finetuning: https://arxiv.org/abs/2306.07280
    HH: front and back transform
    
    see https://arxiv.org/pdf/2405.20271v1
    """

    ether_dropout: float = 0.0

    flip_side: bool = False
    """apply ETHER on the other (smaller) side to reduce computational overhead"""

    _model_keys = ['alpha', 'collection_layers', 'collect_input' ,'collection_keys_in', 'nb', 'Htype', 'ether_dropout', 'flip_side']


@dataclass
class SideinETHER(ETHER, Sidein):
    """Transform: ETHER. Target: activations from layer.out.input
    """

    _reprpo_class = PL_REPRPO_SIDE_ETHER_MODEL


@dataclass
class SideoutETHER(ETHER, Sideout):
    """Transform: ETHER. Target: activations from layer.in.output."""



    _reprpo_class = PL_REPRPO_SIDE_ETHER_MODEL
