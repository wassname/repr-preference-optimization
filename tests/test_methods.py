import os
import pytest
import yaml

from reprpo import silence

silence.test()

from reprpo.experiments import experiment_configs
from reprpo.training import train

configs = [(k, v[1]) for k, v in experiment_configs.items()]
print("configs", configs)


# @jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("name,config", configs)
def test_train_method_dev(name, config):
    """test all methods in dev mode"""

    f = "./configs/dev.yaml"
    overrides = yaml.safe_load(open(f))
    training_args = config
    if f is not None:
        overrides = yaml.safe_load(open(f))
        for k, v in overrides.items():
            setattr(training_args, k, v)

    print(f"loaded default config from {f}")
    print(training_args)
    train(training_args)
