set shell := ["zsh", "-c"]

# settings
# set dotenv-load

# Export all just variables as environment variables.
# set export

export CUDA_VISIBLE_DEVICES := "0"
export TOKENIZERS_PARALLELISM := "false"

[private]
default:
    @just --list

# run one method, by argument
run METHOD='reprpo_ortho':
    . ./.venv/bin/activate
    python nbs/train2.py {{METHOD}} --verbose


run_all:
    #!/usr/bin/zsh
    echo "REPR_CONFIG=$REPR_CONFIG"
    # export WANDB_GROUP=$(date +%Y%m%d_%H%M%S)
    export WANDB_GROUP=${WANDB_GROUP:-$(date +%Y%m%d_%H%M%S)}
    echo "WANDB_GROUP=$WANDB_GROUP"

    # export HF_DATASETS_OFFLINE=1
    # export WANDB_MODE=offline

    . ./.venv/bin/activate
    python nbs/train2.py sidein-ether
    python nbs/train2.py dpo
    python nbs/train2.py sidein-ether --Htype oft
    python nbs/train2.py sidein-ether --Htype ether
    # biggest first so we find out about OOM first
    python nbs/train2.py sideout-hra
    python nbs/train2.py ortho
    python nbs/train2.py sidein
    python nbs/train2.py hra
    python nbs/train2.py sidein-hra
    python nbs/train2.py sideout
    python nbs/train2.py hs
    python nbs/train2.py svd
    python nbs/train2.py svd --quantile 1.0
    python nbs/train2.py hra --no-apply_GS

run_llama:
    #!/usr/bin/zsh
    export REPR_CONFIG=./configs/llama3_7b.yaml
    just run_all


dev:
    #!/usr/bin/zsh
    export REPR_CONFIG=./configs/dev.yaml
    . ./.venv/bin/activate
    python nbs/train2.py -m pdb sidein-hra
