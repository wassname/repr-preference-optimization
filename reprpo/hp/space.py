import optuna
import torch
from torch import nn

import optuna

def base_reprpo_params(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-7, 1e-2, log=True),
        "collect_input": trial.suggest_categorical("collect_input", [True, False]),
        "collect_hs": trial.suggest_categorical("collect_hs", [True, False]),
    }

def ortho_params(trial):
    return {
        "orthogonal_map": trial.suggest_categorical("orthogonal_map", ('householder', 'cayley', 'matrix_exp')),
    }

def ether_params(trial):
    return {
        "nb": trial.suggest_int("nb", 1, 32, log=True),
        "Htype": trial.suggest_categorical("Htype", ["ether", "etherplus", "oft", "etherplusHH"]),
        "flip_side": trial.suggest_categorical("flip_side", [True, False]),
        "reduction": trial.suggest_int("reduction", 1, 512, log=True),
    }

def prefvec_params(trial):
    return {
        "β": trial.suggest_float("β", 1e-6, 2.0, log=True),
        "use_orth_loss": trial.suggest_categorical("use_orth_loss", [True, False]),
        "use_angle_loss": trial.suggest_categorical("use_angle_loss", [True, False]),
        "use_dpo_loss": trial.suggest_categorical("use_dpo_loss", [True, False]),
        "use_nll_loss": trial.suggest_categorical("use_nll_loss", [True, False]),
        "weight_tokens": trial.suggest_categorical("weight_tokens", [True, False]),
        "use_proj_rel": trial.suggest_categorical("use_proj_rel", [True, False]),
    }



def hra_params(trial):
    return {
        "r": trial.suggest_int("r", 2, 512, log=True),
        "apply_GS": trial.suggest_categorical("apply_GS", [True, False]),
    }

def rank_params(trial):
    return {
        "α": trial.suggest_float("α", 1e-4, 1e4, log=True),
        "β": trial.suggest_float("β", 1e-1, 1e2, log=True),
    }

def mse_params(trial):
    return {
        "α": trial.suggest_float("α", 0, 10.0),
    }

def svd_params(trial):
    args = {
        "quantile": trial.suggest_categorical("quantile", ["float", 1]),
        "dual_svd": trial.suggest_categorical("dual_svd", [True, False]),
    }
    if args["quantile"] == "float":
        args["quantile"] = trial.suggest_float("quantile_value", 0.1, 0.9, step=0.1)  
    return args


# Define other parameter groups similarly

def projgrad(trial):
    args = {
        "lr": trial.suggest_float("lr", 1e-7, 1e-2, log=True),
        "β": trial.suggest_float("β", 0.0, 1.0, log=False),
        "reverse_pref": trial.suggest_categorical("reverse_pref", [True, False]),
        "scale_orth": trial.suggest_categorical("scale_orth", [True, False]),
        "weight_dim": trial.suggest_int("weight_dim", 0, 2),
        "neg_slope": trial.suggest_categorical("neg_slope",[0, 'float']),
        "mag_clip": trial.suggest_categorical("mag_clip", [None, "float"]),
    }
    if args["mag_clip"] == "float":
        args["mag_clip"] = trial.suggest_float("mag_clip_value", 1e-2, 1e4, log=True)
    if args["neg_slope"] == "float":
        args["neg_slope"] = trial.suggest_float("neg_slope_value", 0, 1)
    # args = {f"loss.{k}": v for k, v in args.items()}
    # args.update(base_reprpo_params(trial))
    return args

def projbp(trial):
    args = {
        "lr": trial.suggest_float("lr", 1e-7, 1e-2, log=True),
        "β": trial.suggest_float("β", 0.0, 1.0, log=False),
        "reverse_pref": trial.suggest_categorical("reverse_pref", [True, False]),
        "scale_orth": trial.suggest_categorical("scale_orth", [True, False]),
        "neg_slope": trial.suggest_categorical("neg_slope",[0, 'float']),
        "mag_clip": trial.suggest_categorical("mag_clip", [None, "float"]),
    }
    if args["mag_clip"] == "float":
        args["mag_clip"] = trial.suggest_float("mag_clip_value", 1e-2, 1e4, log=True)
    if args["neg_slope"] == "float":
        args["neg_slope"] = trial.suggest_float("neg_slope_value", 0, 1)
    # args = {f"loss.{k}": v for k, v in args.items()}
    # args.update(base_reprpo_params(trial))
    return args

def ether_prefvec(trial):
    args = base_reprpo_params(trial)
    args.update({f"transform.{k}": v for k, v in ether_params(trial).items()})
    args.update({f"loss.{k}": v for k, v in prefvec_params(trial).items()})
    return args

def hra_rank(trial):
    args = base_reprpo_params(trial)
    args.update({f"transform.{k}": v for k, v in hra_params(trial).items()})
    args.update({f"loss.{k}": v for k, v in rank_params(trial).items()})
    return args

def svd_mse(trial):
    args = base_reprpo_params(trial)
    args.update({f"transform.{k}": v for k, v in svd_params(trial).items()})
    args.update({f"loss.{k}": v for k, v in mse_params(trial).items()})
    return args


def dpo(trial):
    args = {"lr": trial.suggest_float("lr", 1e-7, 1e-2, log=True)}
    return args

def ortho_prefvec(trial):
    args = base_reprpo_params(trial)
    args.update({f"transform.{k}": v for k, v in ortho_params(trial).items()})
    args.update({f"loss.{k}": v for k, v in prefvec_params(trial).items()})
    return args

# Define other search space functions similarly

search_spaces = {
    # starter experiment name, search space function
    'side-svd-mse': svd_mse,
    'side-hra-rank': hra_rank,
    'side-ether-prefvec': ether_prefvec,
    "hs-ortho-prefvec": ortho_prefvec, 
    'projgrad': projgrad,
    'projbp': projbp,
    'dpo': dpo,
}

