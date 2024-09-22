from reprpo.interventions import Interventions, DPOConfig, ReprPOConfig
from reprpo.interventions.losses import Losses
from reprpo.interventions.transforms import Transforms

experiment_configs = {
    "ether-side-mse": (
        "1",
        ReprPOConfig(
            transform=Transforms.Ether.value(),
            loss_fn=Losses.mse.value(),
        ),
    ),
    "ether-side-rank": (
        "2",
        ReprPOConfig(
            transform=Transforms.Ether.value(),
            loss_fn=Losses.rank.value(),
        ),
    ),
    "ether-side-prefvec": (
        "3",  # unstable?
        ReprPOConfig(
            transform=Transforms.Ether.value(),
            loss_fn=Losses.prefvec.value(),
            lr=1e-5,
        ),
    ),
    "none-side-prefvec": (
        "4",  # unstable?
        ReprPOConfig(
            transform=Transforms.none.value(),
            loss_fn=Losses.prefvec.value(),
            lr=1e-5,
        ),
    ),
    "none-hs-prefvec": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.none.value(),
            loss_fn=Losses.prefvec.value(),
        ),
    ),
    "none-hs-rank": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.none.value(),
            loss_fn=Losses.rank.value(),
        ),
    ),
    "ether-hs-rank": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.Ether.value(),
            loss_fn=Losses.rank.value(),
        ),
    ),
    "ether-hs-mse": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.Ether.value(),
            loss_fn=Losses.mse.value(),
        ),
    ),
    "ether-hs-prefvec": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.Ether.value(),
            loss_fn=Losses.prefvec.value(),
        ),
    ),
    "hra-hs-prefvec": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.HRA.value(),
            loss_fn=Losses.prefvec.value(),
        ),
    ),
    "ortho-hs-prefvec": (
        "",
        ReprPOConfig(
            collection_keys_in=(),
            transform=Transforms.Ortho.value(),
            loss_fn=Losses.prefvec.value(),
        ),
    ),
    "dpo": ("DPO experiment.", DPOConfig()),
}
