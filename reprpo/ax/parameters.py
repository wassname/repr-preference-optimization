parameters_ether_prefvec = [
    # main
    {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
    {"name": "collect_input", "type": "choice", "values": [False, True]},
    {"name": "collect_hs", "type": "choice", "values": [False, True]},
    # prefvec
    {
        "name": "loss.β",
        "type": "range",
        "bounds": [1.e-6, 2.],
        "log_scale": True,
    },
    {
        "name": "loss.use_dpo_loss",
        "type": "choice",
        "values": [False, True],
    },
    {
        "name": "loss.use_nll_loss",
        "type": "choice",
        "values": [False, True],
    },
    {
        "name": "loss.use_angle_loss",
        "type": "choice",
        "values": [False, True],
    },
    {
        "name": "loss.weight_tokens",
        "type": "choice",
        "values": [False, True],
    },
    {
        "name": "loss.use_orth_loss",
        "type": "choice",
        "values": [False, True],
    },
    # ether
    {
        "name": "transform.nb",
        "type": "range",
        "bounds": [1, 64],
    },
    {
        "name": "transform.Htype",
        "type": "choice",
        "values": ["ether", "etherplus", "oft", "etherplusHH"],
    },
    {
        "name": "transform.reduction",
        "type": "range",
        "bounds": [1, 128],
    },
]

parameters_transform = [
    {
        "name": "transform",
        "type": "choice",
        "values": ["ether", "svd", "ortho", "none", "hra"],
        "dependents": {
            "ether": ["transform.nb", "transform.Htype", "transform.reduction"],
            "svd": ["transform.quantile", "transform.dual_svd"],
        },
    },
    # ether
    {
        "name": "transform.nb",
        "type": "range",
        "bounds": [1, 64],
    },
    {
        "name": "transform.Htype",
        "type": "choice",
        "values": ["ether", "etherplus", "oft", "etherplusHH"],
    },
    {
        "name": "transform.reduction",
        "type": "range",
        "bounds": [1, 1024],
    },
    # # SVD
    {
        "name": "transform.quantile",
        "type": "choice",
        "values": [0.1, 0.25, 0.5, 0.75, 1.0],
    },
    {
        "name": "transform.dual_svd",
        "type": "choice",
        "values": [False, True],
    },
]


parameters_loss = [
    {
        "name": "loss",
        "type": "choice",
        "values": ["mse", "prefvec", "rank"],
        "dependents": {
            "prefvec": ["loss.β", "loss.use_dpo_loss","loss.use_nll_loss","loss.weight_tokens","loss.use_orth_loss",],
            "rank": ["loss.α"],
            "mse": ["loss.α"],
        },
    },
    {
        "name": "loss.α",
        "type": "range",
        "bounds": [1.e-6, 2.],
        "log_scale": True,
    },

    {
        "name": "loss.use_dpo_loss",
        "type": "choice",
        "values": [False, True],
    },
    {
        "name": "loss.use_nll_loss",
        "type": "choice",
        "values": [False, True],
    },
    {
        "name": "loss.use_angle_loss",
        "type": "choice",
        "values": [False, True],
    },
    {
        "name": "loss.weight_tokens",
        "type": "choice",
        "values": [False, True],
    },
    {
        "name": "loss.use_orth_loss",
        "type": "choice",
        "values": [False, True],
    },
]
