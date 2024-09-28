from reprpo.interventions import DPOConfig, ReprPOConfig, DPOProjGradConfig
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms

experiment_configs = {
    
    # TODO remame to collect-transform-loss
    "side-ether-mse": (
        "Collect hs from the side channels, apply an ETHER transform and use MSE loss.",
        ReprPOConfig(
            transform=Transforms.ether.value(),
            loss=Losses.mse.value(),
        ),
    ),
    "side-ether-rank": (
        "2",
        ReprPOConfig(
            transform=Transforms.ether.value(),
            loss=Losses.rank.value(),
        ),
    ),
    "side-ether-prefvec": (
        "3",  # unstable?
        ReprPOConfig(
            transform=Transforms.ether.value(),
            loss=Losses.prefvec.value(),
            lr=1e-5,
        ),
    ),

    "side-none-mse": (
        "Collect hs from the side channels, apply an ETHER transform and use MSE loss.",
        ReprPOConfig(
            transform=Transforms.none.value(),
            loss=Losses.mse.value(),
        ),
    ),
    "side-none-rank": (
        "2",
        ReprPOConfig(
            transform=Transforms.none.value(),
            loss=Losses.rank.value(),
        ),
    ),
    "side-none-prefvec": (
        "3",  # unstable?
        ReprPOConfig(
            transform=Transforms.none.value(),
            loss=Losses.prefvec.value(),
            lr=1e-5,
        ),
    ),
    "hs-none-prefvec": (
        "Collect hs and use PreferenceVector loss.",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.none.value(),
            loss=Losses.prefvec.value(),
        ),
    ),
    "hs-none-rank": (
        "Collect hs and use ranking loss.",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.none.value(),
            loss=Losses.rank.value(),
        ),
    ),
    "hs-none-mse": (
        "Collect hs and use ranking loss.",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.none.value(),
            loss=Losses.mse.value(),
        ),
    ),
    "hs-ether-rank": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.ether.value(),
            loss=Losses.rank.value(),
        ),
    ),
    # "hs-ether-mse": (
    #     "",  # unstable with tinyllama, no grad otherwise?
    #     ReprPOConfig(
    #         collection_keys_in=(),
    #         transform=Transforms.ether.value(),
    #         loss=Losses.mse.value(),
    #     ),
    # ),
    "hs-ether-prefvec": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.ether.value(),
            loss=Losses.prefvec.value(),
        ),
    ),
    "hs-hra-prefvec": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.hra.value(),
            loss=Losses.prefvec.value(),
        ),
    ),
    "hs-ortho-prefvec": (
        "Collect hs, apply Orthogonal transform and use PreferenceVector loss.",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.ortho.value(),
            loss=Losses.prefvec.value(),
        ),
    ),
    "hs-svd-prefvec": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.svd.value(),
            loss=Losses.prefvec.value(),
        ),
    ),
    # baseline
    "dpo": ("DPO experiment.", DPOConfig()),
    "projgrad": ("DPO experiment.", DPOProjGradConfig()),
    "projgrad_right": ("DPO experiment.", DPOProjGradConfig(weight_dim=1, neg_slope=1.0)),
    "projgrad_left": ("DPO experiment.", DPOProjGradConfig(weight_dim=0, neg_slope=1.0)),
    "projgrad_fb": ("DPO experiment.", DPOProjGradConfig(β=0.0, neg_slope=1.0)),
    "projgrad_fs": ("DPO experiment.", DPOProjGradConfig(β=0.03, )),
    "projgrad_fs2": ("DPO experiment.", DPOProjGradConfig(β=1.0, scale_orth=True)),
    "projgrad_fs3": ("DPO experiment.", DPOProjGradConfig(β=1.0, )),
    "projgrad_bs3": ("DPO experiment.", DPOProjGradConfig(β=1.0, reverse_pref=True)),
    "projgrad_fs4": ("DPO experiment.", DPOProjGradConfig(β=1.0, mag_clip=0.02)),
    # variants
    "hs-svd-prefvec-dual": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.svd.value(dual_svd=True),
            loss=Losses.prefvec.value(),
        ),
    ),
    "hs-svd-prefvec-dualhard": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.svd.value(dual_svd=True, quantile=1.),
            loss=Losses.prefvec.value(),
        ),
    ),
    "hs-svd-prefvec-hard": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.svd.value(quantile=1.),
            loss=Losses.prefvec.value(),
        ),
    ),
    "side-ether1-prefvec": (
        "3",  # unstable?
        ReprPOConfig(
            transform=Transforms.ether.value(Htype="ether"),
            loss=Losses.prefvec.value(),
            lr=1e-5,
        ),
    ),
    "side-oft-prefvec": (
        "3",  # unstable?
        ReprPOConfig(
            transform=Transforms.ether.value(Htype="oft"),
            loss=Losses.prefvec.value(),
            lr=1e-5,
        ),
    ),
    # TODO ether with Htype=ether and oft
}
