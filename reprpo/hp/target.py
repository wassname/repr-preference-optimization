import copy
from reprpo.experiments import experiment_configs
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms
from reprpo.training import train


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
        return setattr(cfg, k, v)


# quick 2m per run
default_tuner_kwargs = dict(
    verbose=0,
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # ideally would be SFT
    batch_size=32,
    # load_in_4bit=True, # doesn't quite halve the memory, speed it about the same
    collection_layers_side=(8, 10, 12, 14, 16, 18),
    eval_samples=128,
    save=False,
    # collect_hs=True,
)


def override(cfg, overrides):
    for k, v in overrides.items():
        try:
            setattrattr(cfg, k, v)
        except ValueError:
            print(f"WARNING: {k} not found in config")
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
    return r
