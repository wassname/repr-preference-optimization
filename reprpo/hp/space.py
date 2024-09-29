import optuna
import torch
from torch import nn

import optuna

def base_reprpo_params(trial):
    return {
        "learning-rate": trial.suggest_float("learning-rate", 1e-6, 1e-3, log=True),
        "collect_input": trial.suggest_categorical("collect_input", [True, False]),
        "collect_hs": trial.suggest_categorical("collect_hs", [True, False]),
    }

def ortho_params(trial):
    return {
        "orthogonal_map": trial.suggest_categorical("orthogonal_map", ('householder', 'cayley', 'matrix_exp')),
    }

def ether_params(trial):
    return {
        "nb": trial.suggest_int("nb", 1, 32),
        "Htype": trial.suggest_categorical("Htype", ["ether", "etherplus", "oft", "etherplusHH"]),
        "flip_side": trial.suggest_categorical("flip_side", [True, False]),
        "reduction": trial.suggest_int("reduction", 1, 200),
    }

def prefvec_params(trial):
    return {
        "loss.β": trial.suggest_float("loss.β", 1e-6, 2.0, log=True),
        "use_orth_loss": trial.suggest_categorical("use_orth_loss", [True, False]),
        "use_angle_loss": trial.suggest_categorical("use_angle_loss", [True, False]),
        "use_dpo_loss": trial.suggest_categorical("use_dpo_loss", [True, False]),
        "use_nll_loss": trial.suggest_categorical("use_nll_loss", [True, False]),
        "weight_tokens": trial.suggest_categorical("weight_tokens", [True, False]),
    }

def hra_params(trial):
    return {
        "r": trial.suggest_int("r", 2, 128),
        "apply_GS": trial.suggest_categorical("apply_GS", [True, False]),
    }

def rank_params(trial):
    return {
        "α": trial.suggest_float("α", 0, 10.0),
    }

def mse_params(trial):
    return {
        "α": trial.suggest_float("α", 0, 10.0),
    }

def svd_params(trial):
    return {
        "quantile": trial.suggest_categorical("quantile", [0.1, 0.5, 0.75, 1]),
        "dual_svd": trial.suggest_categorical("dual_svd", [True, False]),
    }


# Define other parameter groups similarly

def projgrad(trial):
    args = {
        "learning-rate": trial.suggest_float("learning-rate", 1e-6, 1e-3, log=True),
        "β": trial.suggest_float("β", 0.0, 1.0, log=False),
        "reverse_pref": trial.suggest_categorical("reverse_pref", [True, False]),
        "scale_orth": trial.suggest_categorical("scale_orth", [True, False]),
        "weight_dim": trial.suggest_int("weight_dim", 0, 2),
        "neg_slope": trial.suggest_categorical("neg_slope",[0, 0.01, 0.1, 0.5, 1]),
        "mag_clip": trial.suggest_categorical("mag_clip", [None, 0.03, 0.1, 0.5, 1.0, 10, 100]),
    }
    return args

def ether_prefvec(trial):
    args = base_reprpo_params(trial)
    args.update(ether_params(trial))
    args.update(prefvec_params(trial))
    return args

def hra_rank(trial):
    args = base_reprpo_params(trial)
    args.update(hra_params(trial))
    args.update(rank_params(trial))
    return args

def svd_mse(trial):
    args = base_reprpo_params(trial)
    args.update(svd_params(trial))
    args.update(mse_params(trial))
    return args


def dpo(trial):
    args = {"learning-rate": trial.suggest_float("learning-rate", 1e-6, 1e-3, log=True)}
    return args

# Define other search space functions similarly

search_spaces = {
    # starter experiment name, search space function
    'projgrad': projgrad,
    'side-ether-prefvec': ether_prefvec,
    'side-svd-mse': svd_mse,
    'side-hra-rank': hra_rank,
}

