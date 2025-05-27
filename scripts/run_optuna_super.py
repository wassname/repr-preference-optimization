# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html#sphx-glr-download-tutorial-10-key-features-005-visualization-py

# %reload_ext autoreload
# %autoreload 2

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] ="expandable_segments:True" # seems to stop gpu mem from filling up despite clearing

import torch
import pandas as pd
from pathlib import Path
import optuna
from reprpo.hp.helpers import optuna_df

from reprpo.hp import space_super


from reprpo.hp.target import  default_tuner_kwargs, key_metric
from reprpo.hp.space_super import objective_super
import wandb

import optuna.pruners
from optuna_integration.wandb import WeightsAndBiasesCallback
# -

# ## Opt

import warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning) 

from optuna.study.study import get_all_study_names

SEED=42
dev = False # put to True for unit test
torch.manual_seed(SEED)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# +
# silence please
import os
from loguru import logger
import logging
from reprpo.interventions.config import ExperimentConfig
from reprpo.hp.target import default_tuner_kwargs, key_metric, override_cfg, override
from dataclasses import asdict

cfg = ExperimentConfig()
override(cfg, default_tuner_kwargs)
default_args = asdict(cfg)

ds_name = default_args['dataset']
model_name = default_args['base_model'].replace("/", "-")
# os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_DATASETS_OFFLINE"] = "1"

os.environ["WANDB_SILENT"] = "true"
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

os.environ["TQDM_DISABLE"] = "true"
ts = pd.Timestamp.now().strftime("%H%M%S")
os.environ["WANDB_GROUP"] = f"optuna4_{ts}_{model_name}_{ds_name}"

if dev:
    f_db = f"sqlite:///:memory:"
else:
    f_db = f"sqlite:///outputs/optuna_super.db"
    f = f_db.replace('sqlite:///', './')
    Path(f).parent.mkdir(parents=True, exist_ok=True)
print(f_db)

study_names = get_all_study_names(storage=f_db)

for study_name in study_names:
    print(study_name)
    study = optuna.load_study(study_name=study_name, storage=f_db)
    try:    
        df_res = optuna_df(study, key_metric)
        print(df_res.to_markdown())
        print()
    except ValueError as e:
        print('-')

from reprpo.hp.wandb import WeightsAndBiasesCallback2

wandb_kwargs = {
    "project": f"reprpo2-optuna_{ds_name}", 
    "group": os.environ.get("WANDB_GROUP"),
    "tags": [ds_name, model_name, "optuna", ],
}
wandb.require(experiment="core")
wandbc = WeightsAndBiasesCallback2(
    wandb_kwargs=wandb_kwargs, as_multirun=True
    )
import numpy as np

# pruner = optuna.pruners.PatientPruner(patience=2, wrapped_pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)

study_name = f"super_{ds_name}_{model_name}"
study = optuna.create_study(
    study_name=study_name,
    direction="maximize",
    load_if_exists=True,
    storage=f_db,
    sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
    pruner=pruner,
)

max_trials = 2000
n = 0
try:
    df = study.trials_dataframe().sort_values('value', ascending=False)
    n = len(df)
except Exception as e:
    print(e)
    pass
if n>0:
    print(f"loaded {n} {study_name} trials")

    df_res = optuna_df(study, key_metric)
    print(df_res.to_markdown())

if wandb.run is not None:
    wandb.run._quiet = True

# track each trial in wandb so we can check if they needed more time to converge and so on
@wandbc.track_in_wandb()
def _objective(trial):
    r =  objective_super(trial, key_metric=key_metric, dev=dev)
    return r

# HACK, the best way to get defaults is sample some, and change them manually
# from reprpo.hp.space_super import superspace
# def sample(study, n=1):
#     study = optuna.create_study()
#     for _ in range(n):
#         trial = study.ask()
#         args, cfg = superspace(trial)
#         print([trial._cached_frozen_trial.params, args, cfg])
# sample(study, 10)
# 1/0

# TODO need cfg and args grr
defaults = [
    # side-none-InnerPO, this uses the most mem python scripts/train.py side-none-rank
    {'space': 'reprpo', 'reprpo.lr': 3e-4,
      'transform': 'none', 
      'loss': 'InnerPO', 'InnerPO.β': 3, 'InnerPO.use_orth_loss': False, 'InnerPO.use_angle_loss': True, 'InnerPO.use_dpo_loss': False, 'InnerPO.use_nll_loss': False, 'InnerPO.use_proj_rel': False},
    # ether-mse
    {'space': 'reprpo', 'reprpo.lr': 2e-4, 
     'transform': 'ether', 'ether.nb': 16, 'ether.Htype': 'ether', 'ether.flip_side': False, 'ether.reduction': 60, 
     'loss': 'mse', 'mse.α': 0.6},
     # baseline
    {'space': 'dpo', 'dpo.lr': 6e-5},
    # {'space': 'projbp', 'projbp.lr': 1e-5, 'projbp.β': 0.9035496929952938, 'projbp.reverse_pref': False, 'projbp.scale_orth': False, 'projbp.neg_slope_value': 0.01, 'projbp.mag_clip': None},
    {'space': 'projgrad', 'projgrad.lr': 6e-5, 'projgrad.β': 7.938142168093467, 'projgrad.reverse_pref': False, 'projgrad.scale_orth': True, 'projgrad.weight_dim': 0, 'projgrad.neg_slope_value': 0.014520194983786325, 'projgrad.mag_clip': None},
    # supr-rank
    {'space': 'reprpo', 'reprpo.lr': 3e-4, 
     'transform': 'supr', 
     'loss': 'rank', 'rank.α': 0.25, 'rank.β': 1.0, 'rank.use_dpo_loss': True, 'rank.use_nll_loss': False, 'rank.use_rank_retain': False},
    # hra-InnerPO
    {
        'space': 'reprpo', 'reprpo.lr': 1e-4, 
        'transform': 'hra', 'r': 38, 'apply_GS': True, 
        'loss': 'InnerPO', 'InnerPO.β': 5, 'InnerPO.use_orth_loss': False, 'InnerPO.use_angle_loss': True, 'InnerPO.use_dpo_loss': False, 'InnerPO.use_nll_loss': False, 'InnerPO.use_proj_rel': False},
]

for params in defaults:
    study.enqueue_trial(params, 
                    user_attrs={"starter_experiment": True},
                    skip_if_exists=True
                    )

study.optimize(_objective, 
            gc_after_trial=True, 
            catch=(AssertionError, OSError, RuntimeError, KeyError, torch.OutOfMemoryError),
            callbacks=[wandbc],
)

print('='*80)
wandb.finish(quiet=True)
