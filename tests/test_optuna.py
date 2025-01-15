import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pytest
import yaml

from reprpo import silence

silence.test()

from reprpo.experiments import experiment_configs
from reprpo.training import train
from reprpo.hp.target import override, default_tuner_kwargs
from reprpo.experiments import experiment_configs
from reprpo.hp.space import search_spaces
import optuna
from reprpo.hp.target import override, default_tuner_kwargs, list2tuples, objective, key_metric, override_cfg
import functools
import copy

spaces = [(name,N,f) for name, (N, f) in search_spaces.items()]

@pytest.mark.parametrize("exp_name,N,trial2args", spaces)
def test_optuna_spaces(exp_name, N, trial2args):
    # exp_name, (N, trial2args) = row
    study = optuna.create_study(direction="maximize")
    # cfg = copy.deepcopy(experiment_configs[exp_name][1])
    # print('exp_name', exp_name)
    # for _ in range(10):
    #     trial = study.ask()
    #     kwargs = trial2args(trial)
    #     override(cfg, default_tuner_kwargs)
    #     override(cfg, kwargs)


    # f = "./configs/dev.yaml"
    # dev_overrides = yaml.safe_load(open(f))
    # TODO list to tuple

    dev_overrides = {**default_tuner_kwargs, **{'dev':True}}

    # try one dev run
    trial = study.ask()
    kwargs = trial2args(trial)
    kwargs = {**kwargs, **dev_overrides}
    r = override_cfg(kwargs, None, exp_name)
    print(r['key_metric'])
    print(r)
