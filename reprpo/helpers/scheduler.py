# from transformers
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_constant_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1
):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
