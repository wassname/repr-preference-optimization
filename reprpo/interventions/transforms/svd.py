# TODO

from reprpo.helpers.svd_decomposer import (
    SoftSVDDecomposer,
    DualSVDDecomposer,
    SVDDecomposer,
)
from dataclasses import dataclass, asdict
from torch import nn


class SVDLayer(nn.Module):
    def __init__(
        self, in_dim, out_dim, dual_svd: bool, quantile: float, model: nn.Module
    ) -> None:
        super().__init__()

        W_e = model.get_input_embeddings().weight.clone().float()
        W_o = model.lm_head.weight.clone()

        if dual_svd:
            self.decomposer = DualSVDDecomposer(
                W_e,
                W_o,
                quantile=quantile,
            )
        else:
            if quantile < 1:
                self.decomposer = SoftSVDDecomposer(W_o, quantile=quantile)
            else:
                self.decomposer = SVDDecomposer(W_o)

    def forward(self, hs):
        hs_io = self.decomposer(hs)
        return hs - hs_io  # .detach() # FIXME, should I not detach this?


@dataclass
class SVDConfig:
    """
    This intervention does not seem to work of converge. It attempt to remove the parts of the hs that are used by lm_head, but because hs is low rank, and lm_head is highrank, it is all used. Hence we try to train on a tiny noisy residual, and cannot.

    It's left in here to show a negative finding, and the question: where do transformer store the working internal memory?
    """

    quantile: float = 0.3
    """What quantile of top singular values to remove from from hs
    - 0.9 would remove 90% of the singular values that contribute to the input and output layers of the model
    - 1
    """

    dual_svd: bool = True
    """if true, will use the embedding and lm_head, if false only lm_head"""

    def c(self, *args, **kwargs):
        return SVDLayer(
            *args,
            **kwargs,
            **asdict(self),
        )
