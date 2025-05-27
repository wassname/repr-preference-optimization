from reprpo.interventions import DPOConfig, ReprPOConfig, ProjGradConfig, ProjBPConfig
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms
from .space import base_reprpo_params, ether_params, hra_params, svd_params, supr_params, InnerPO_params, rank_params, mse_params, dpo_params, projgrad_params, projbp_params
from .target import override_cfg
from reprpo.training import train
import optuna


def superspace(trial):
    intervention = trial.suggest_categorical("space", ['dpo', 'projgrad', 'projbp', 'reprpo'])
    if intervention == 'dpo':
        args =  dpo_params(trial)
        cfg = DPOConfig(**args)
    elif intervention == 'projgrad':
        args = projgrad_params(trial)
        cfg = ProjGradConfig(**args)
    elif intervention == 'projbp':
        args = projbp_params(trial)
        cfg = ProjBPConfig(**args)
    elif intervention == 'reprpo':
        base_args = base_reprpo_params(trial)
        
        transform = trial.suggest_categorical("transform", ['ether', 'hra', 'none', 'svd', 'supr'])

        base_args['collect_hs'] = True
        if transform == 'none':
            base_args["collect_input"] = trial.suggest_categorical("collect_input", [False, True])
            transform_args = {}
        elif transform == 'ether':
            transform_args = ether_params(trial)
        elif transform == 'hra':
            transform_args = hra_params(trial)
        elif transform == 'svd':
            transform_args = svd_params(trial)
        elif transform == 'supr':
            transform_args = supr_params(trial)
        base_args['collect_hs'] = transform != 'none'

        loss = trial.suggest_categorical("loss", ['InnerPO', 'rank', 'mse'])
        if loss == 'InnerPO':
            loss_args = InnerPO_params(trial)
        elif loss == 'rank':
            loss_args = rank_params(trial)
        elif loss == 'mse':
            loss_args = mse_params(trial)
        else:
            raise ValueError("Invalid loss")

        cfg = ReprPOConfig(
            transform=Transforms[transform].value(**transform_args), 
            loss=Losses[loss].value(**loss_args), 
            **base_args
        )
        transform_args2 = {f"transform.{k}": v for k, v in transform_args.items()}
        loss_args2 = {f"loss.{k}": v for k, v in loss_args.items()}
        args = {**base_args, **transform_args2, **loss_args2}
    else:
        raise ValueError("Invalid intervention")    
    return args, cfg

def objective_super(trial: optuna.Trial, key_metric:str, **kwargs) -> float:

    # set params
    kwargs2, cfg = superspace(trial)
    kwargs3 = {**kwargs2, **kwargs}
    cfg2 = override_cfg(kwargs3, cfg)
    print(cfg2)

    # run
    r = train(cfg2, trial=trial)

    # store res
    for k,v in r.items():
        trial.set_user_attr(k, v)
    return r[key_metric]
