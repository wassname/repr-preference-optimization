from reprpo.interventions import DPOConfig, ReprPOConfig, ProjGradConfig, ProjBPConfig
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms


def get_default_bool(c):
    """get the bool attrs from a dataclass"""
    for k in dir(c):
        v = getattr(c, k)
        if isinstance(v, bool):
            yield k, v

experiment_configs = {

    # this will use lots of memory, so good to have it as the first one
    "side-none-InnerPO2": ("",
        ReprPOConfig(
            collect_hs=False,
            transform=Transforms.none.value(),
            loss=Losses.InnerPO.value(β=3.),
        ),
    ),


    # baseline
    "dpo": ("DPO experiment.", DPOConfig()),

    # gradient based methods
    "projgrad": ("projgrad experiment.", ProjGradConfig()),

    "hs-ether-rank2": ("",
        ReprPOConfig(
            collect_hs=True,
            transform=Transforms.ether.value(),
            loss=Losses.rank.value(use_nll_loss=True, β=0.1, α=100),
        ),
    ),
    
    "projbp": ("projbp experiment.", ProjBPConfig()),


}

# first all the reprpo experiments
experiment_configs2 = {}
for loss in Losses:
    l_name = loss.name
    experiment_configs2.update({
        f"side-none-{l_name}": (
            f"No transform one side activations and use {l_name} loss.",
            ReprPOConfig(
                collect_hs=False,
                transform=Transforms.none.value(),
                loss=loss.value(),
            ),
        )
    })
for transform in Transforms:
    for loss in Losses:
        t_name = transform.name
        l_name = loss.name
        experiment_configs2.update({
            f"hs-{t_name}-{l_name}": (
                f"Apply {t_name} transform on hs and use {l_name} loss.",
                ReprPOConfig(
                    collect_hs=True,
                    transform=transform.value(),
                    loss=loss.value(),
                ),
            )
        })

# shuffle experiment_configs2
import random
keys = list(experiment_configs2.keys())
random.shuffle(keys)
experiment_configs2 = {k: experiment_configs2[k] for k in keys}
experiment_configs.update(experiment_configs2)

for Htype in ["ether", "etherplus", "oft", "etherplusHH"]:
    experiment_configs.update({
            f"hs-ether-InnerPO-Htype_{Htype}": ('', 
            ReprPOConfig(
                collect_hs=True,
                transform=Transforms.ether.value(Htype=Htype),
                loss=Losses.InnerPO.value(),
            ),
        ),
    })


for k,v in list(get_default_bool(Losses.InnerPO.value)):
    # variants, with bools flipped
    experiment_configs.update({   
        f"side-none-InnerPO-{k}_{not v}": ("No transform one side activations and use InnerPO loss.",
            ReprPOConfig(     
                collect_hs=False,             
                transform=Transforms.none.value(),
                loss=Losses.InnerPO.value(**{k:not v}),
            ),
        ),
    })

for k,v in list(get_default_bool(Losses.rank.value)):
    # variants, with bools flipped
    experiment_configs.update({   
        f"side-none-rank-{k}_{not v}": ("No transform one side activations and use rank loss.",
            ReprPOConfig(      
                collect_hs=False,      
                transform=Transforms.none.value(),
                loss=Losses.rank.value(**{k:not v}),
            ),
        ),
    })

experiment_configs.update({   

    # variants of the supression transform with only the last two layers
    "hs-supr-InnerPO2": ('', 
        ReprPOConfig(
            collect_hs=True,
            collection_layers='-2,1',
            transform=Transforms.supr.value(),
            loss=Losses.InnerPO.value(),
            # lr=1e-5,
        ),
    ),
    "hs-supr-rank2": ('', 
        ReprPOConfig(
            collect_hs=True,
            collection_layers='-2,1',
            transform=Transforms.supr.value(),
            loss=Losses.rank.value(),
        ),
    ),

})
