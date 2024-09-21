import os
import pytest
import yaml

from reprpo import silence
silence.test()

from reprpo.interventions import Interventions, InterventionType
from reprpo.training import train


# @jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("Method", Interventions)
def test_train_method_dev(Method):
    """test all methods in dev mode"""

    f="./configs/dev.yaml"
    overrides = yaml.safe_load(open(f))
    training_args = Method.value()
    if f is not None:
        overrides = yaml.safe_load(open(f))
        for k, v in overrides.items():
            setattr(training_args, k, v)
    
    print(f"loaded default config from {f}")
    print(training_args)
    train(training_args)
