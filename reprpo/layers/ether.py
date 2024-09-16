"""
https://github.dev/mwbini/ether
"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Literal, Callable

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass


class ETHERLayer(nn.Module):
    def __init__(self, nb: int, 
                 Htype: str,
                 ether_dropout: float,
                 ):
        """Store ETHER specific attributes in a class.

        Args:
            nb: number of diagonal blocks
            ether_dropout: dropout that is applied on the input in the ETHER branch
        """
        super().__init__()
        assert nb >= 0
        self.nb = nb
        self.Htype = Htype
        # Optional dropout
        if ether_dropout > 0.0:
            self.ether_dropout = nn.Dropout(p=ether_dropout)
        else:
            self.ether_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False


class ETHERLinear(ETHERLayer):
    # ETHER implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        bias: bool = False,
        # ↓ the remaining part is for ETHER
        nb: int = 0,
        Htype: str = 'ether',
        ether_dropout: float = 0.0,
        flip_side: bool = False,
        **kwargs,
    ):
        """ETHER wrapper around linear class.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            nb: number of diagonal blocks
            Htype: type of transformation
            ether_dropout: dropout that is applied on the input in the ETHER branch
            flip_side: apply ETHER on the other (smaller) side to reduce computational overhead
        """
        super().__init__(nb=nb, Htype=Htype, ether_dropout=ether_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias, **kwargs)
        self.Htype = Htype
        if 'HH' in self.Htype:
            self.is_HtransposeH = True
        else:
            self.is_HtransposeH = False
        self.flip_side = flip_side and not self.is_HtransposeH


        if nb>0:
            # get R
            self.nb = nb

            if self.flip_side:
                tmp_features = in_features
                in_features = out_features
                out_features = tmp_features
                
            if self.Htype == 'ether':
                R_shape = [nb, in_features // nb]
                ether_R = torch.rand(R_shape[-1])
                ether_R = torch.stack([ether_R] * self.nb)
                self.ether_R = nn.Parameter(ether_R)
                nn.init.kaiming_uniform_(self.ether_R, a=math.sqrt(5))
            elif self.Htype == 'etherplus':
                R_shape = [nb, in_features // nb]
                ether_R = torch.rand(R_shape[-1])
                ether_R = torch.stack([ether_R] * nb)
                self.ether_R = nn.Parameter(ether_R)
                nn.init.kaiming_uniform_(self.ether_R, a=math.sqrt(5))
                ether_R2 = - torch.empty_like(ether_R).copy_(ether_R)
                self.ether_R2 = nn.Parameter(ether_R2)
            elif self.Htype == 'oft':
                R_shape = [nb, in_features // nb, in_features // nb]
                ether_R = torch.zeros(R_shape[-1], R_shape[-1])
                ether_R = torch.stack([ether_R] * self.nb)
                self.ether_R = nn.Parameter(ether_R)
            # HH models
            elif self.Htype == 'etherplusHH':
                # front
                R_shape = [nb, in_features // nb]
                ether_R = torch.rand(R_shape[-1])
                ether_R = torch.stack([ether_R] * nb)
                self.ether_R = nn.Parameter(ether_R)
                nn.init.kaiming_uniform_(self.ether_R, a=math.sqrt(5))
                ether_R2 = - torch.empty_like(ether_R).copy_(ether_R)
                self.ether_R2 = nn.Parameter(ether_R2)

                # back
                R34_shape = [nb, out_features // nb]
                ether_R3 = torch.rand(R34_shape[-1])
                ether_R3 = torch.stack([ether_R3] * nb)
                self.ether_R3 = nn.Parameter(ether_R3)
                nn.init.kaiming_uniform_(self.ether_R3, a=math.sqrt(5))
                ether_R4 = - torch.empty_like(ether_R3).copy_(ether_R3)
                self.ether_R4 = nn.Parameter(ether_R4)
            else:
                raise ValueError(f"Unknown Htype: {self.Htype}")


    def reset_parameters(self):
        """Reset ETHER weights"""
        if hasattr(self, "ether_R"):
            nn.init.kaiming_uniform_(self.ether_R, a=math.sqrt(5))
            if hasattr(self, "ether_R2"):
                self.ether_R2.data = - torch.empty_like(self.ether_R).copy_(self.ether_R)
        if hasattr(self, "ether_R3"):
            nn.init.kaiming_uniform_(self.ether_R3, a=math.sqrt(5))
            if hasattr(self, "ether_R4"):
                self.ether_R4.data = - torch.empty_like(self.ether_R3).copy_(self.ether_R3)


    def get_H(self):
        if self.Htype == 'ether':
            H = self.ether(self.ether_R)
        elif self.Htype == 'etherplus':
            H = self.etherplus(self.ether_R, self.ether_R2)
        elif self.Htype == 'oft':
            H = self.oft(self.ether_R)
        # or get HH
        elif self.Htype == 'etherplusHH':
            H = self.etherplus(self.ether_R, self.ether_R2)
            H2 = self.etherplus(self.ether_R3, self.ether_R4)

        if self.is_HtransposeH:
            return H, H2
        else:
            return H, None


    def forward(self, x: torch.Tensor):
        # do the forward pass with the pretrained weights multiplied by the ETHER weights
        
        # - weights
        # get H
        H, H2 = self.get_H()
        
        # pretrained weights
        filt = self.linear.weight.data

        # - shapes
        nb,m,n = H.shape  #> [4,512,512]
        f,d = filt.shape  #> [8192,2048] or [2048,2048]

        # - direct transformation
        if not self.flip_side:
            # split in nb blocks
            filt = filt.reshape(nb, f, d//nb)

            # multiply
            filt = torch.einsum('rfm,rmn->rfn', filt, H)

            # rebuild in one block
            filt = filt.reshape(f, d)

        # - transposed transformation
        if self.flip_side or self.is_HtransposeH:
            # split in nb blocks
            filt = filt.reshape(nb, f//nb, d)

            # multiply
            if self.is_HtransposeH:
                filt = torch.einsum('rnm,rmd->rnd', H2, filt)
            else:
                filt = torch.einsum('rnm,rmd->rnd', H, filt)

            # rebuild in one block
            filt = filt.reshape(f, d)

        # - bias
        bias_term = self.linear.bias.data if self.linear.bias is not None else None

        # Apply the trainable identity matrix
        ether = nn.functional.linear(input=self.ether_dropout(x), weight=filt, bias=bias_term)
        return ether

    def ether(self, R):
        nb, r = R.shape
        I = torch.eye(r, device=R.device, dtype=R.dtype).unsqueeze(0).expand(nb, r, r)
        R = R.unsqueeze(1)
        H = I - 2 * torch.bmm(R.transpose(1,2), R) / torch.bmm(R, R.transpose(1,2))
        return H

    def etherplus(self, R1, R2):
        nb, r = R1.shape
        I = torch.eye(r, device=R1.device).unsqueeze(0).expand(nb, r, r)
        R1 = R1.unsqueeze(1)
        R2 = R2.unsqueeze(1)
        H = I - torch.bmm(R1.transpose(1,2), R1) / torch.bmm(R1, R1.transpose(1,2)) +  torch.bmm(R2.transpose(1,2), R2) / torch.bmm(R2, R2.transpose(1,2))
        return H
    
    def oft(self, R):
        nb, r, c = R.shape
        skew = 0.5 * (R - R.transpose(1, 2))
        I = torch.eye(r, device=R.device).unsqueeze(0).expand(nb, r, c)
        H = torch.bmm(I + skew, torch.inverse(I - skew))
        return H
    

class ETHERLinearSmall(ETHERLinear):
    """To save params this projects onto a smaller space then back up."""
    # ETHER implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        bias: bool = False,
        reduction: int = 4,
        # ↓ the remaining part is for ETHER
        nb: int = 0,
        Htype: str = 'ether',
        ether_dropout: float = 0.0,
        flip_side: bool = False,
        **kwargs,
    ):
        super().__init__(in_features, out_features//reduction, bias, nb, Htype, ether_dropout, flip_side, **kwargs)
        self.linear_up = torch.nn.Linear(out_features//reduction, out_features, bias=bias, **kwargs)

    def forward(self, x: torch.Tensor):
        super_out = super().forward(x)
        return self.linear_up(super_out)

@dataclass
class _ETHERConfig:
    """ETHER parameters"""

    nb: int = 4
    """number of diagonal blocks"""

    Htype: Literal['ether', 'etherplus', 'oft', 'etherplusHH'] = 'ether'
    """type of transformation 

    - ether: like HRA but allowing a negative unit vector (reflection)
    - etherplus: relaxing distance and orthogonality constraints
    - oft: Orthogonal Finetuning: https://arxiv.org/abs/2306.07280
    - HH: front and back transform
    
    see https://arxiv.org/pdf/2405.20271v1
    """

    ether_dropout: float = 0.0

    flip_side: bool = False
    """apply ETHER on the other (smaller) side to reduce computational overhead"""

    lr: float = 1e-3

    _model_keys = ['alpha', 'collection_layers', 'collect_input' ,'collection_keys_in', 'nb', 'Htype', 'ether_dropout', 'flip_side']
