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

from reprpo.training import train
from reprpo.experiments import experiment_configs
from reprpo.hp.space import search_spaces

# ## Objective

#

SEED=42
key_metric = "acc_gain_vs_ref/oos"
torch.manual_seed(SEED)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# +
# silence please
import os
from loguru import logger
# logger.remove()
# logger.remove()
# logger.add(os.sys.stderr, level="WARNING")

# os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TQDM_DISABLE"] = "true"
ts = pd.Timestamp.now().strftime("%H%M%S")
os.environ["WANDB_GROUP"] = "optuna4_{ts}"
# -

f_db = f"sqlite:///outputs/optuna4.db"
f = f_db.replace('sqlite:///', './')
print(f)
Path(f).parent.mkdir(parents=True, exist_ok=True)
f_db

# +
# print(f'to visualise run in cli\ncd nbs\noptuna-dashboard {f_db}')

# +
from reprpo.hp.target import override, default_tuner_kwargs, list2tuples, objective
import functools
from reprpo.experiments import experiment_configs
import copy
import wandb

import optuna.pruners
from optuna_integration.wandb import WeightsAndBiasesCallback
# -

# ## Opt

# Note on pruning. It's only really usefull with validation metrics and for long jobs over many epochs. I've got a small proxy job so there is no need.

# from reprpo.experiments import experiment_configs
from reprpo.hp.space import experiment_configs
experiment_configs.keys()

import warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning) 

# +
from optuna.study.study import storages, get_all_study_names
study_names = get_all_study_names(storage=f_db)

for study_name in study_names:
    print(study_name)
    study = optuna.load_study(study_name=study_name, storage=f_db)
    try:
        df_res = optuna_df(study, key_metric)
        print(df_res)
        print()
    except ValueError as e:
        print('-')

# +
# unit test, moved to pytest
# for exp_name, (N, trial2args) in search_spaces.items():
#     study = optuna.create_study(direction="maximize")
#     cfg = copy.deepcopy(experiment_configs[exp_name][1])
#     print('exp_name', exp_name)
#     for _ in range(10):
#         trial = study.ask()
#         kwargs = trial2args(trial)
#         override(cfg, default_tuner_kwargs)
#         override(cfg, kwargs)
#         kwargs = list2tuples(kwargs)

#     # try one dev run
#     kwargs['dev'] = True
#     _objective = functools.partial(objective, key_metric=key_metric, starter_experiment_name=exp_name, trial2args=trial2args)

#     study.optimize(_objective, 
#                 n_trials=1, # do 20 at a time, round robin, untill done
#                 gc_after_trial=True, 
#                 # catch=(AssertionError, OSError, RuntimeError, KeyError, torch.OutOfMemoryError)
#     )
# -


from optuna.integration.wandb import WeightsAndBiasesCallback
wandb_kwargs = {"project": "reprpo2-optuna"}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)



MAX_TRIALS= 250
import numpy as np
spaces = list(search_spaces.items())
while True:
    np.random.shuffle(spaces)
    for exp_name, (max_trials, trial2args) in spaces:
        try:
            study_name = f"{exp_name}"
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                load_if_exists=True,
                storage=f_db,
                sampler=optuna.samplers.TPESampler(seed=SEED),
                pruner=optuna.pruners.MedianPruner(),
            )

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

            
            if n < max_trials:
                @wandbc.track_in_wandb()
                def _objective(trial):
                    return objective(trial, key_metric=key_metric, starter_experiment_name=exp_name, trial2args=trial2args)
                # _objective = functools.partial(objective, key_metric=key_metric, starter_experiment_name=exp_name, trial2args=trial2args)
                # _objective = wandbc.track_in_wandb(_objective)

                study.optimize(_objective, 
                            n_trials=20, # do 20 at a time, round robin, untill done
                            gc_after_trial=True, 
                            catch=(AssertionError, OSError, RuntimeError, KeyError, torch.OutOfMemoryError),
                            callbacks=[wandbc],
                )

            print('='*80)
            wandb.finish(quiet=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception(e)
