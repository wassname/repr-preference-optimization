# set shell := ["zsh", "-uc"]


export CUDA_VISIBLE_DEVICES := "0"
export TOKENIZERS_PARALLELISM := "false"

[private]
default:
    @just --list




# run one method, with args
run +args='':
    #!/usr/bin/zsh
    source ./.venv/bin/activate
    python scripts/train.py {{ args }}


run_all +args='':
    #!/usr/bin/bash

    . ./.venv/bin/activate
    export WANDB_GROUP=${WANDB_GROUP:-mdl-$(date +%y%m%d_%H%M)}
    echo "REPR_CONFIG=$REPR_CONFIG"
    echo "WANDB_GROUP=$WANDB_GROUP"
    echo "EXTRA_ARGS={{args}}"

    readarray -t EXPERIMENTS <<< "$(python ./scripts/export_experiments.py)"

    set -x
    # echo EXPERIMENTS $EXPERIMENTS $S
    for METHOD in $EXPERIMENTS; do
        echo "python scripts/train.py $METHOD {{args}}"
        python scripts/train.py $METHOD {{args}}
    done


run_ds +args='':
    #!/usr/bin/zsh
    # export REPR_CONFIG=./configs/llama-3-7b_a100.yaml
    export WANDB_GROUP=${WANDB_GROUP:-ds-$(date +%y%m%d_%H%M)}
    source ./.venv/bin/activate
    export DS=(
        alpaca_easy
        alpaca_mmlu 
        alpaca_low_quality 
        alpaca_short 
        code_easy 
        alpaca_mmlu 
        math 
        raven_matrices  
        us_history_textbook
    )
    export METHODS=(
        dpo
        hs-ether-prefvec
        side-none-prefvec
        projgrad
    )
    set -x
    echo $DS
    for ds in $DS; do
        echo "DS=$ds"
        for METHOD in $METHODS; do
            echo python scripts/train.py $METHOD --dataset $ds {{ args }}
            python scripts/train.py $METHOD --dataset $ds {{ args }}
        done
    done

run_sizes +args='':
    #!/usr/bin/zsh
    export METHODS=(
        dpo
        hs-ether-prefvec
        side-none-prefvec
        projgrad
    )
    export WANDB_GROUP=${WANDB_GROUP:-sz-$(date +%y%m%d_%H%M)}
    for METHOD in $EXPERIMENTS; do
        REPR_CONFIG=./configs/llama-3-7b_a100.yaml python scripts/train.py $METHOD {{ args }}
        REPR_CONFIG=./configs/llama-3-2-3b_a100.yaml python scripts/train.py $METHOD {{ args }}
        REPR_CONFIG=./configs/llama-3-2-1b_a100.yaml python scripts/train.py $METHOD {{ args }}
    done


run_llama +args='':
    #!/usr/bin/zsh
    export REPR_CONFIG=./configs/llama-3-7b_a100.yaml
    just run_all {{ args }}
    just run_ds {{ args }}
    just run_sizes {{ args }}

# run pytest
test:
    . ./.venv/bin/activate
    pytest --pdb -x -s -v \
        --jaxtyping-packages=reprpo,beartype.beartype --beartype-packages='reprpo'

# run in dev mode with pdb
dev:
    #!/usr/bin/zsh
    export REPR_CONFIG=./configs/dev.yaml
    . ./.venv/bin/activate
    python scripts/train.py -m pdb sidein-hra

# copy trained models from runpod
cp:
    rsync -avz --ignore-existing runpod:/workspace/repr-preference-optimization/ouputs/ ./ouputs/

