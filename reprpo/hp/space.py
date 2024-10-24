import optuna
import torch
from torch import nn
from reprpo.interventions import DPOConfig, ReprPOConfig, ProjGradConfig, ProjBPConfig
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms
import optuna

# components

def base_reprpo_params(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-7, 1e-2, log=True),
        "collect_input": trial.suggest_categorical("collect_input", [False, True]),
        "collect_hs": trial.suggest_categorical("collect_hs", [False, True]),
    }

def ortho_params(trial):
    return {
        "orthogonal_map": trial.suggest_categorical("orthogonal_map", ('householder', 'cayley', 'matrix_exp')),
    }

def ether_params(trial):
    return {
        "nb": trial.suggest_int("nb", 1, 32, log=True),
        "Htype": trial.suggest_categorical("Htype", ["ether", "etherplus", "oft", "etherplusHH"]),
        "flip_side": trial.suggest_categorical("flip_side", [False, True]),
        "reduction": trial.suggest_int("reduction", 1, 512, log=True),
    }

def supr_params(trial):
    return {
    }

def prefvec_params(trial):
    return {
        "β": trial.suggest_float("β", 1e-6, 2.0, log=True),
        "use_orth_loss": trial.suggest_categorical("use_orth_loss", [False, True]),
        "use_angle_loss": trial.suggest_categorical("use_angle_loss", [False, True]),
        "use_dpo_loss": trial.suggest_categorical("use_dpo_loss", [False, True]),
        "use_nll_loss": trial.suggest_categorical("use_nll_loss", [False, True]),
        # "weight_tokens": trial.suggest_categorical("weight_tokens", [False, True]),
        "use_proj_rel": trial.suggest_categorical("use_proj_rel", [False, True]),
    }



def hra_params(trial):
    return {
        "r": trial.suggest_int("r", 2, 512, log=True),
        "apply_GS": trial.suggest_categorical("apply_GS", [False, True]),
    }

def rank_params(trial):
    return {
        "α": trial.suggest_float("α", 1e-4, 1e4, log=True),
        "β": trial.suggest_float("β", 1e-1, 1e2, log=True),
        "use_dpo_loss": trial.suggest_categorical("use_dpo_loss", [False, True]),
        "use_nll_loss": trial.suggest_categorical("use_nll_loss", [False, True]),
        "use_rank_retain": trial.suggest_categorical("use_rank_retain", [False, True]),
    }

def mse_params(trial):
    return {
        "α": trial.suggest_float("α", 1e-4, 1e4, log=True),
    }

def svd_params(trial):
    args = {
        "quantile": trial.suggest_categorical("quantile", ["float", 1]),
        "dual_svd": trial.suggest_categorical("dual_svd", [False, True]),
    }
    if args["quantile"] == "float":
        args["quantile"] = trial.suggest_float("quantile_value", 0.1, 0.9, step=0.1)
    return args


# Define other parameter groups similarly

def projgrad_params(trial):
    args = {
        "lr": trial.suggest_float("lr", 1e-7, 1e-2, log=True),
        "β": trial.suggest_float("β", 1e-2, 1e3, log=True),
        "reverse_pref": trial.suggest_categorical("reverse_pref", [False, True]),
        "scale_orth": trial.suggest_categorical("scale_orth", [False, True]),
        "mag_clip": trial.suggest_categorical("mag_clip", [0, 1]),
        "weight_dim": trial.suggest_int("weight_dim", 0, 2),
        "neg_slope": trial.suggest_float("neg_slope_value", 1e-8, 1, log=True),
        "mag_clip": trial.suggest_categorical("mag_clip", [None, "float"]),
    }
    if args["mag_clip"] == "float":
        args["mag_clip"] = trial.suggest_float("mag_clip_value", 1e-2, 1e4, log=True)
    return args

def projbp_params(trial):
    args = {
        "lr": trial.suggest_float("lr", 1e-7, 1e-2, log=True),
        "β": trial.suggest_float("β", 0.0, 1.0, log=False),
        "reverse_pref": trial.suggest_categorical("reverse_pref", [False, True]),
        "scale_orth": trial.suggest_categorical("scale_orth", [False, True]),
        "neg_slope": trial.suggest_float("neg_slope_value", 1e-8, 1, log=True),
        "mag_clip": trial.suggest_categorical("mag_clip", [None, "float"]),
    }
    if args["mag_clip"] == "float":
        args["mag_clip"] = trial.suggest_float("mag_clip_value", 1e-2, 1e4, log=True)
    return args

def dpo(trial):
    args = {"lr": trial.suggest_float("lr", 1e-6, 1e-4, log=True)}
    # beta TODO
    # ipo vs dpo vs others
    return args

## experiments

def ether_prefvec(trial):
    args = base_reprpo_params(trial)
    # args.update({f"transform.{k}": v for k, v in ether_params(trial).items()})
    args.update({f"loss.{k}": v for k, v in prefvec_params(trial).items()})
    return args

def hs_ether_rank(trial):
    args = base_reprpo_params(trial)
    # args = {"lr": trial.suggest_float("lr", 1e-7, 1e-2, log=True)}
    # args.update({f"transform.{k}": v for k, v in hra_params(trial).items()})
    args.update({f"loss.{k}": v for k, v in rank_params(trial).items()})
    return args

def hs_ether_mse(trial):
    # args = base_reprpo_params(trial)
    # args = {
    #     "lr": trial.suggest_float("lr", 1e-7, 1e-2, log=True),
    # }    
    args = base_reprpo_params(trial)
    # args.update({f"transform.{k}": v for k, v in svd_params(trial).items()})
    args.update({f"loss.{k}": v for k, v in mse_params(trial).items()})
    return args




def hs_ether_prefvec(trial):
    args = base_reprpo_params(trial)
    # args = {"lr": trial.suggest_float("lr", 1e-7, 1e-2, log=True)}
    # args.update({f"transform.{k}": v for k, v in ortho_params(trial).items()})
    args.update({f"loss.{k}": v for k, v in prefvec_params(trial).items()})
    return args

def hs_supr_mse(trial):
    args = base_reprpo_params(trial)
    args.update({f"loss.{k}": v for k, v in mse_params(trial).items()})
    return args


def superspace(trial):
    intervention = trial.suggest_categorical("space", ['dpo', 'projgrad', 'projbp', 'reprpo'])
    if intervention == 'dpo':
        return dpo(trial)
    elif intervention == 'projgrad':
        return projgrad_params(trial)
    elif intervention == 'projbp':
        return projbp_params(trial)
    elif intervention == 'reprpo':
        args = base_reprpo_params(trial)
        transform = trial.suggest_categorical("transform", ['ether', 'hra', 'none', 'svd', 'supr'])

        if transform == 'ether':
            args.update({f"transform.{k}": v for k, v in ether_params(trial).items()})
        elif transform == 'hra':
            args.update({f"transform.{k}": v for k, v in hra_params(trial).items()})
        elif transform == 'none':
            pass
        elif transform == 'svd':
            args.update({f"transform.{k}": v for k, v in svd_params(trial).items()})
        elif transform == 'supr':
            args.update({f"transform.{k}": v for k, v in supr_params(trial).items()})

        loss = trial.suggest_categorical("loss", ['prefvec', 'rank', 'mse'])
        if loss == 'prefvec':
            args.update({f"loss.{k}": v for k, v in prefvec_params(trial).items()})
        elif loss == 'rank':
            args.update({f"loss.{k}": v for k, v in rank_params(trial).items()})
        elif loss == 'mse':
            args.update({f"loss.{k}": v for k, v in mse_params(trial).items()})
        else:
            raise ValueError("Invalid loss")
        return args
    else:
        raise ValueError("Invalid intervention")


    


# Define other search space functions similarly

# TODO replace with custom experiments
search_spaces = {
    # starter experiment name, search space function
    # number: rougly 50 per float, 10 per bool
    'hs-ether-mse': (150, hs_ether_mse),
    'hs-ether-rank': (150, hs_ether_rank),
    "hs-ether-prefvec": (550, hs_ether_prefvec), 
    "hs-supr-mse": (10, hs_supr_mse), 
    # 'ether-prefvec': (250, ether_prefvec),
    'projgrad2': (350, projgrad_params),
    # 'projbp': (500, projbp),
    'dpo': (5, dpo),
}


experiment_configs = {
    "hs-supr-mse": (
        "",
        ReprPOConfig(
            collect_hs=True, # OOM on sides, to many layers
            transform=Transforms.supr.value(),
            loss=Losses.mse.value(),
        ),
    ),
    "hs-ether-mse": (
        "",  # unstable due to svd
        ReprPOConfig(
            collect_hs=True, # OOM on sides, to many layers
            transform=Transforms.ether.value(),
            loss=Losses.mse.value(),
        ),

    ),
    "hs-ether-rank": (
        "",
        ReprPOConfig(
            collect_hs=True, # OOM on sides, to many layers
            transform=Transforms.ether.value(),
            loss=Losses.rank.value(),
        ),
    ),
    "hs-ether-prefvec": (
        "",  
        ReprPOConfig(
            collect_hs=True,# OOM on sides, to many layers
            transform=Transforms.ether.value(),
            loss=Losses.prefvec.value(),
        ),
    ),

    "ether-prefvec": (
        "",  
        ReprPOConfig(
            transform=Transforms.ether.value(),
            loss=Losses.prefvec.value(),
        ),
    ),
    "dpo": ("DPO experiment.", DPOConfig()),
    "projbp": ("DPO experiment.", ProjBPConfig()),
    "projgrad2": ("DPO experiment.", ProjGradConfig()),
    # TODO also some side ones with no transform
}

