from lightning.pytorch.loggers import WandbLogger
import wandb
import os
import pandas as pd

def flatten_dict(d):
    return pd.json_normalize(d, sep='_').to_dict(orient='records')[0]

def init_wandb(config, save_dir, group_name, run_name, project="reprpo2"):
    config = flatten_dict(config)  # flatten the config dict
    # you can also add a “post” sub-dict here, just like you do today
    wandb_kwargs = dict(
        name=run_name,
        save_dir=save_dir,
        project=project,
        group=group_name,
        config=config,
        mode="disabled" if os.getenv("WANDB_MODE")=="disabled" else "online",
    )
    logger = WandbLogger(**wandb_kwargs)
    # tag & rename existing run if needed
    if wandb.run:
        wandb.config.update(config, allow_val_change=True)
        wandb.run.name = run_name
        wandb.settings.quiet=True
        # add whatever tags you need
    return logger
