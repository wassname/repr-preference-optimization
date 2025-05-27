import modal
cuda_version = "12.6.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "cd /root && git clone https://github.com/wassname/repr-preference-optimization",
        "cd /root/repr-preference-optimization && uv sync --no-build-isolation",
    )
    # .pip_install(  # add flash-attn
    #     "flash-attn==2.7.4.post1", extra_options="--no-build-isolation"
    # )
)

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

volume = modal.Volume.from_name(
    "repr", create_if_missing=True
)
MODEL_DIR = "/root/repr-preference-optimization/outputs"

@app.function(gpu="A100-80GB", 
              volumes={MODEL_DIR: volume},  # stores fine-tuned model
              image=image, timeout=60 * 60,
              secrets=[modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])],
)
def run_flash_attn():
    import torch
    from flash_attn import flash_attn_func

    train()
    # The trained model information has been output to the volume mounted at `MODEL_DIR`.
    # To persist this data for use in our web app, we 'commit' the changes
    # to the volume.
    volume.commit()
