from reprpo.interventions import DPOConfig, ReprPOConfig, ProjGradConfig, ProjBPConfig
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms

experiment_configs = {}

# first all the reprpo experiments
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
    experiment_configs.update({
        f"side-none-{l_name}": (
            f"No transform one side activations and use {l_name} loss.",
            ReprPOConfig(
                transform=Transforms.none.value(),
                loss=loss.value(),
            ),
        )
    })


experiment_configs.update({   
    # variants
    "hs-oft-prefvec": ('', 
        ReprPOConfig(
            collect_hs=True,
            transform=Transforms.ether.value(Htype="oft"),
            loss=Losses.prefvec.value(β=0.1),
        ),
    ),

    "side-none-prefvec2": ("No transform one side activations and use prefvec loss.",
        ReprPOConfig(
            transform=Transforms.none.value(),
            loss=Losses.prefvec.value(β=0.5,),
        ),
    ),
    "projbp": ("DPO experiment.", ProjBPConfig()),

    "hs-ether-prefvec2": ("No transform one side activations and use prefvec loss.",
        ReprPOConfig(
            transform=Transforms.ether.value(),
            loss=Losses.prefvec.value(β=3),
        ),
    ),

    # baseline
    "dpo": ("DPO experiment.", DPOConfig()),

    # gradient based methods
    "projgrad": ("DPO experiment.", ProjGradConfig()),

})
