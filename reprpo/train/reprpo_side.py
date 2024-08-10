import torch.nn.functional as F
from reprpo.train.lightning import PL_MODEL


def compute_reprpo_side_loss_batch(batch, model):
    pass


class PL_REPRPO_MODEL(PL_MODEL):
    def _loss_fn(self, batch, model):
        return compute_reprpo_side_loss_batch(batch, model)
