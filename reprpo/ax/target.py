import copy
from reprpo.experiments import experiment_configs
from reprpo.interventions import Interventions, DPOConfig, ReprPOConfig
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
tuner_kwargs = dict(
    verbose=0,
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # ideally would be SFT
    batch_size=64,
    load_in_4bit=True,
    collection_layers_side=[8, 10, 12, 14, 16, 18],
    eval_samples=64,
)


def override(cfg, overrides):
    for k, v in overrides.items():
        try:
            setattrattr(cfg, k, v)
        except ValueError:
            print(f"WARNING: {k} not found in config")
    return cfg


def objective_func(**kwargs):
    cfg = copy.deepcopy(experiment_configs["side-ether-prefvec"][1])
    if isinstance(cfg.loss, str):
        cfg.loss = getattr(Losses, cfg.loss).value()
    if isinstance(cfg.transform, str):
        cfg.transform = getattr(Transforms, cfg.transform).value()
    override(cfg, tuner_kwargs)
    override(cfg, kwargs)
    r = train(cfg)
    return r
