import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pytest
import yaml

from reprpo import silence
import hydra
from reprpo.config import register_configs
from omegaconf import OmegaConf
silence.test()

from reprpo.training import train


from reprpo.interventions.dpo import DPOConfig
cs = register_configs()

def get_options_from_hydra_cs(p):
    """get a list of options from the hydra structured configstore."""
    ps = cs.list(p)
    return [f"+{p}={v.replace('.yaml', '')}" for v in ps]

def get_overrides():

    # hard to fully automate this as some require +, some don't
    groups = [s for s in cs.list('') if not s.endswith('yaml')]

    overrides = [
        ['+intervention=dpo'],
    ]
    for o in get_options_from_hydra_cs('intervention.loss'):
        overrides.append(['+intervention=reprpo','+intervention.transform=ether', o])
    for o in get_options_from_hydra_cs('intervention.transform'):
        overrides.append(['+intervention=reprpo','+intervention.loss=prefvec', o])
    return overrides

overrides = get_overrides()
print('overrides', overrides)

hydra.initialize(config_path=".", job_name="unittest_main")

@pytest.mark.parametrize("overriders", overrides)
def test_train_method_dev(overriders):
    """test all methods in dev mode
    ~10s
    """

    overrides=["+experiment=unit_test"]+overriders

    # load in unit test overrides
    # FIXME: proper hydra config
    f = "./configs/dev.yaml"
    new_overrides = yaml.safe_load(open(f))
    # new_overrides = {k: tuple(v) if isinstance(v, list) else v for k,v in new_overrides.items()}
    # overrides += [f'{k}={v}' for k,v in new_overrides.items()]

    # filter out any reprpo specific overrides. TODO improve
    if 'dpo' in overriders[0]:
        new_overrides = {k:v for k,v in new_overrides.items() if 'intervention' not in k}
    
    print(f"loaded default config from {f}")
    print(overrides)

    # set up exact config
    training_args = hydra.compose(
        config_name="expconf",
        overrides=overrides,
    )
    training_args = OmegaConf.merge(training_args, new_overrides)
    # just do merge? OmegaConf.merge

    print(training_args)
    train(training_args)

# @pytest.mark.parametrize("name,config", configs)
# def test_train_method_dev1b(name, config):
#     """test all methods in dev with a small model
#     ~50s
#     """

#     f = "./configs/dev1b.yaml"
#     overrides = yaml.safe_load(open(f))
#     training_args = config
#     if f is not None:
#         overrides = yaml.safe_load(open(f))
#         for k, v in overrides.items():
#             setattr(training_args, k, v)

#     print(f"loaded default config from {f}")
#     print(training_args)
#     train(training_args)
