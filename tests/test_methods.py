import os
os.environ["WANDB_MODE"] = "disabled"
# os.environ["WANDB_SILENT"] = "true"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TQDM_DISABLE"] = "true"

# from jaxtyping import jaxtyped
# from typeguard import typechecked as typechecker
import pytest
import yaml

from reprpo.train import Methods, MethodsUnion
from reprpo.training import train

# @jaxtyped(typechecker=typechecker)
@pytest.mark.parametrize("Method", Methods)
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
