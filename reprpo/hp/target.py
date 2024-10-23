import copy
# from reprpo.experiments import experiment_configs
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms
from reprpo.training import train
from reprpo.hp.space import search_spaces, experiment_configs
from loguru import logger
import gc
import torch
import functools
import optuna
import wandb

key_metric = "acc_gain_vs_ref/oos"

def setattrattr(cfg, k, v):
    """
    Sets an attr even it's like o.a.b
    """
    if "." in k:
        k, k2 = k.split(".")
        return setattrattr(getattr(cfg, k), k2, v)
    else:
        # print(cfg, k, v)
        if hasattr(cfg, k):
            return setattr(cfg, k, v)
        else:
            raise ValueError(f"{k} not found in config. {type(cfg).__name__}")


# quick 2m per run
default_tuner_kwargs = dict(
    verbose=1,
    batch_size=16*3,
    eval_samples=128,
    n_samples=1800 * 6, # to make sure it converges
    save=False,
    wandb=True,
    #  https://huggingface.co/datasets/wassname/genies_preferences
    # dataset='code_easy',
    dataset='alpaca_low_quality',
    # dataset='raven_matrices',
    # dataset='math',

)


def override(cfg, overrides):
    for k, v in overrides.items():
        try:
            setattrattr(cfg, k, v)
        except ValueError:
            logger.warning(f"WARNING: {k} not found in config")
    return cfg

from reprpo.training import get_display_name_from_args


def list2tuples(d):
    for k, v in d.items():
        if isinstance(v, list):
            d[k] = tuple(v)
    return d

def objective_func(kwargs, trial, starter_experiment_name):
    cfg = copy.deepcopy(experiment_configs[starter_experiment_name][1])
    override(cfg, default_tuner_kwargs)
    override(cfg, kwargs)
    r = train(cfg, trial=trial)
    return r

def objective(trial: optuna.Trial, starter_experiment_name, trial2args, key_metric:str) -> float:
    kwargs = trial2args(trial)
    r = objective_func(kwargs, trial, starter_experiment_name)
    for k,v in r.items():
        trial.set_user_attr(k, v)
    return r[key_metric]
