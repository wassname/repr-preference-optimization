# set shell := ["zsh", "-uc"]


export CUDA_VISIBLE_DEVICES := "0"
export TOKENIZERS_PARALLELISM := "false"

[private]
default:
    @just --list


# run pytest
test:
    . ./.venv/bin/activate
    pytest --pdb -x -s -v \
        --jaxtyping-packages=reprpo,beartype.beartype --beartype-packages='reprpo'


# run one method, with args
run METHOD='reprpo_ortho' +EXTRA_ARGS='':
    #!/usr/bin/zsh
    export EXTRA_ARGS=${EXTRA_ARGS:-}
    source ./.venv/bin/activate
    python scripts/train.py {{METHOD}} $EXTRA_ARGS


run_all EXTRA_ARGS='':
    #!/usr/bin/bash

    . ./.venv/bin/activate
    export WANDB_GROUP=${WANDB_GROUP:-mdl-$(date +%y%m%d_%H%M)}
    echo "REPR_CONFIG=$REPR_CONFIG"
    echo "WANDB_GROUP=$WANDB_GROUP"
    echo "EXTRA_ARGS=$EXTRA_ARGS"

    # export EXPERIMENTS=(hs-ortho-rank hs-ortho-prefvec hs-ortho-mse side-none-mse hs-ether-rank hs-ether-prefvec hs-ether-mse hs-hra-rank hs-hra-prefvec hs-hra-mse hs-none-rank hs-none-prefvec hs-none-mse hs-svd-rank hs-svd-prefvec hs-svd-mse dpo projbp projgrad hs-oft-prefvec)
    # export EXPERIMENTS="${EXPERIMENTS[@]}"

    readarray -t EXPERIMENTS <<< "$(python ./scripts/export_experiments.py)"

    set -x
    echo EXPERIMENTS $EXPERIMENTS $S
    for METHOD in $EXPERIMENTS; do
        echo "python scripts/train.py $METHOD $EXTRA_ARGS"
        python scripts/train.py $METHOD $EXTRA_ARGS
    done


run_ds:
    #!/usr/bin/zsh
    export REPR_CONFIG=./configs/llama-3-7b_a100.yaml
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

    . ./.venv/bin/activate
    export METHODS=(
        dpo
        hs-ether-prefvec
        side-none-prefvec
        projgrad
    )
    echo $DS
    for ds in $DS; do
        echo "DS=$ds"
        for METHOD in $EXPERIMENTS; do
            echo python scripts/train.py $METHOD --dataset $ds
            python scripts/train.py $METHOD --dataset $ds
        done
    done

run_sizes:
    #!/usr/bin/zsh -x
    export METHODS=(
        dpo
        hs-ether-prefvec
        side-none-prefvec
        projgrad
    )
    export WANDB_GROUP=${WANDB_GROUP:-sz-$(date +%y%m%d_%H%M)}
    for METHOD in $EXPERIMENTS; do
        REPR_CONFIG=./configs/llama-3-7b_a100.yaml python scripts/train.py $METHOD
        REPR_CONFIG=./configs/llama-3-2-3b_a100.yaml python scripts/train.py $METHOD
        REPR_CONFIG=./configs/llama-3-2-1b_a100.yaml python scripts/train.py $METHOD


run_llama:
    #!/usr/bin/zsh
    export REPR_CONFIG=./configs/llama-3-7b_a100.yaml
    just run_all
    just run_ds
    just run_sizes


# dev:
#     #!/usr/bin/zsh
#     export REPR_CONFIG=./configs/dev.yaml
#     . ./.venv/bin/activate
#     python scripts/train.py -m pdb sidein-hra

# copy trained models from runpod
cp:
    rsync -avz --ignore-existing runpod:/workspace/repr-preference-optimization/ouputs/ ./ouputs/

