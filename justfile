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
    python scripts/train.py {{METHOD}} --verbose

run_ds:
    #!/usr/bin/zsh
    export REPR_CONFIG=./configs/llama3_7b.yaml
    . ./.venv/bin/activate

    python scripts/train.py hrakl

    export WANDB_GROUP=${WANDB_GROUP:-ds-$(date +%Y%m%d_%H%M%S)}
    export DS=(alpaca_easy alpaca_mmlu alpaca_low_quality alpaca_short code_easy alpaca_mmlu math raven_matrices  us_history_textbook)
    for ds in $DS; do
        echo "DS=$ds"
        . ./.venv/bin/activate
        # python scripts/train.py sideout-ether --dataset $ds
        python scripts/train.py ether --dataset $ds
        python scripts/train.py sidein --dataset $ds
        python scripts/train.py dpo --dataset $ds
        python scripts/train.py hrakl --dataset $ds
    done

run_all:
    #!/usr/bin/zsh
    echo "REPR_CONFIG=$REPR_CONFIG"
    # export WANDB_GROUP=$(date +%Y%m%d_%H%M%S)
    export WANDB_GROUP=${WANDB_GROUP:-mdl-$(date +%Y%m%d_%H%M%S)}
    echo "WANDB_GROUP=$WANDB_GROUP"

    # export HF_DATASETS_OFFLINE=1
    # export WANDB_MODE=offline
    # export WANDB_SILENT"]=true
    # export HF_DATASETS_DISABLE_PROGRESS_BARS=1

    . ./.venv/bin/activate
    python scripts/train.py sidein-ether
    python scripts/train.py dpo
    python scripts/train.py ether
    python scripts/train.py hra --no-rel-loss --verbose --lr 1e-5
    # biggest first so we find out about OOM first
    python scripts/train.py sideout-hra
    python scripts/train.py ortho
    python scripts/train.py sidein
    python scripts/train.py hra
    python scripts/train.py sidein-hra
    python scripts/train.py sideout
    python scripts/train.py hs
    python scripts/train.py svd
    python scripts/train.py sidein-ether --Htype oft
    python scripts/train.py sidein-ether --Htype ether
    python scripts/train.py svd --quantile 1.0
    python scripts/train.py hra --no-apply_GS

run_llama:
    #!/usr/bin/zsh
    export REPR_CONFIG=./configs/llama3_7b.yaml
    just run_all


dev:
    #!/usr/bin/zsh
    export REPR_CONFIG=./configs/dev.yaml
    . ./.venv/bin/activate
    python scripts/train.py -m pdb sidein-hra

# copy trained models from runpod
cp:
    rsync -avz --ignore-existing runpod:/workspace/repr-preference-optimization/ouputs/ ./ouputs/


run_temp:
    #!/usr/bin/zsh
    export REPR_CONFIG=./configs/llama3_7b.yaml
    . ./.venv/bin/activate
    python scripts/train.py sidein-ether
    python scripts/train.py ether
