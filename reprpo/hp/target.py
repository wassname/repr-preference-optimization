import copy
from reprpo.experiments import experiment_configs
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms
from reprpo.training import train
from loguru import logger
import gc
import torch


def setattrattr(cfg, k, v):
    """
    Sets an attr even it's like o.a.b
    """
    if "." in k:
        k, k2 = k.split(".")
        # print(k, k2)
        # print(getattr(cfg, k))
        return setattrattr(getattr(cfg, k), k2, v)
    else:
        # print(cfg, k, v)
        if hasattr(cfg, k):
            return setattr(cfg, k, v)
        else:
            raise ValueError(f"{k} not found in config. {type(cfg).__name__}")


# quick 2m per run
default_tuner_kwargs = dict(
    verbose=0,
    batch_size=32,
    eval_samples=128,
    n_samples=1800 * 2, # to make sure it converges
    save=False,
    wandb=False,
)


def override(cfg, overrides):
    for k, v in overrides.items():
        try:
            setattrattr(cfg, k, v)
        except ValueError:
            logger.warning(f"WARNING: {k} not found in config")
    return cfg


def objective_func(kwargs, trial):
    cfg = copy.deepcopy(experiment_configs["side-ether-prefvec"][1])
    override(cfg, default_tuner_kwargs)

    # # so first we do the ones high in the heirarchy
    # if "loss" in kwargs:
    #     loss = kwargs.pop("loss")
    #     cfg.loss = getattr(Losses, loss).value()
    # if "transform" in kwargs:
    #     transform = kwargs.pop("transform")
    #     cfg.transform = getattr(Transforms, transform).value()

    # now subcommands
    override(cfg, kwargs)
    r = train(cfg, trial=trial)
    gc.collect()
    torch.cuda.empty_cache()
    return r
