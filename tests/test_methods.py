import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pytest
import yaml

from reprpo import silence
import hydra
from reprpo.config import register_configs

silence.test()

from reprpo.training import train


from reprpo.interventions.dpo import DPOConfig
cs = register_configs()

# # TODO need to do combinations of each
# paths = ['intervention', 'reprpo.loss_fn', 'reprpo.transform']
# overrides = {}
# for p in paths:
#     ps = cs.list(p)
#     ps = [f"+{p}={v.replace('.yaml', '')}" for v in ps]
#     overrides[p] = ps

# # now do combinations
# print('overrides', overrides)

# TODO automate parts of this
overrides = [
    ['+intervention=dpo'],
    # ['+intervention=reprpo','+intervention.loss_fn=mse','+intervention.transform=ether'],
    # ['+intervention=reprpo','+intervention.loss_fn=prefvec','+intervention.transform=none'],
    # ['+intervention=reprpo','+intervention.loss_fn=rank','+intervention.transform=svd'],
    # ['+intervention=reprpo','+intervention.loss_fn=rank','+intervention.transform=svd'],
]
def get_overrides(p):
    ps = cs.list(p)
    return [f"+{p}={v.replace('.yaml', '')}" for v in ps]
for o in get_overrides('reprpo.loss_fn'):
    overrides.append(['+intervention=reprpo','+reprpo.transform=ether', o])
for o in get_overrides('reprpo.transform'):
    overrides.append(['+intervention=reprpo','+reprpo.loss_fn=prefrank', o])
print('overrides', overrides)


# @jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("overriders", overrides)
def test_train_method_dev(overriders):
    """test all methods in dev mode
    ~10s
    """

    hydra.initialize(config_path=".", job_name="unittest_main")
    overrides=["+experiment=unit_test"]+overriders

    # load in unit test overrides
    f = "./configs/dev.yaml"
    new_overrides = yaml.safe_load(open(f))
    overrides += [f'{k}={v}' for k,v in new_overrides.items()]
    print(f"loaded default config from {f}")
    print(overrides)

    # set up exact config
    training_args = hydra.compose(
        config_name="expconf",
        overrides=overrides,
    )

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
