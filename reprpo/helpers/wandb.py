import os
from pathlib import Path

def init_wandb(__vsc_ipynb_file__ : str) -> str:
    # bridge vscode notebooks and wandb
    nb_file = Path(__vsc_ipynb_file__)
    os.environ['WANDB_NOTEBOOK_NAME'] = nb_file.name # must be a valid file
    os.environ["WANDB_PROJECT"] = nb_file.parent.name
    nb_name = nb_file.stem.replace(' ', '_') # used to trl run name

    # enable wandb service (experimental, https://github.com/wandb/client/blob/master/docs/dev/wandb-service-user.md)
    # this hopefully fixes issues with multiprocessing
    import wandb
    wandb.require(experiment='service')
    return nb_name
