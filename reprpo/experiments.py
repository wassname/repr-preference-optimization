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

    "hs-ether-prefvec2": ("",
        ReprPOConfig(
            transform=Transforms.ether.value(),
            loss=Losses.prefvec.value(β=3.),
        ),
    ),

    # baseline
    "dpo": ("DPO experiment.", DPOConfig()),

    # gradient based methods
    "projgrad": ("projgrad experiment.", ProjGradConfig()),

    "projbp": ("projbp experiment.", ProjBPConfig()),
}

# first all the reprpo experiments
for loss in Losses:
    l_name = loss.name
    experiment_configs.update({
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
        experiment_configs.update({
            f"hs-{t_name}-{l_name}": (
                f"Apply {t_name} transform on hs and use {l_name} loss.",
                ReprPOConfig(
                    collect_hs=True,
                    transform=transform.value(),
                    loss=loss.value(),
                ),
            )
        })

for Htype in ["ether", "etherplus", "oft", "etherplusHH"]:
    experiment_configs.update({
            f"hs-ether-prefvec-Htype={Htype}": ('', 
            ReprPOConfig(
                collect_hs=True,
                transform=Transforms.ether.value(Htype=Htype),
                loss=Losses.prefvec.value(),
            ),
        ),
    })


for k,v in list(get_default_bool(Losses.prefvec.value)):
    # variants, with bools flipped
    experiment_configs.update({   
        f"side-none-prefvec-{k}={not v}": ("No transform one side activations and use prefvec loss.",
            ReprPOConfig(     
                collect_hs=False,             
                transform=Transforms.none.value(),
                loss=Losses.prefvec.value(**{k:not v}),
            ),
        ),
    })

for k,v in list(get_default_bool(Losses.rank.value)):
    # variants, with bools flipped
    experiment_configs.update({   
        f"side-none-rank-{k}={not v}": ("No transform one side activations and use prefvec loss.",
            ReprPOConfig(      
                collect_hs=False,      
                transform=Transforms.none.value(),
                loss=Losses.rank.value(**{k:not v}),
            ),
        ),
    })

experiment_configs.update({   

    # variants of the supression transform with onyl the last two layers
    "hs-supr-prefvec2": ('', 
        ReprPOConfig(
            collect_hs=True,
            collection_layers_side=(-2, -1),
            transform=Transforms.supr.value(),
            loss=Losses.prefvec.value(),
            # lr=1e-5,
        ),
    ),
    "hs-supr-rank2": ('', 
        ReprPOConfig(
            collect_hs=True,
            collection_layers_side=(-2, -1),
            transform=Transforms.supr.value(),
            loss=Losses.rank.value(),
        ),
    ),
    "hs-supr-mse2": ('', 
        ReprPOConfig(
            collect_hs=True,
            collection_layers_side=(-2, -1),
            transform=Transforms.supr.value(),
            loss=Losses.mse.value(),
        ),
    ),


})
