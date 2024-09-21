# set shell := ["zsh", "-uc"]

# settings
# set dotenv-load

# Export all just variables as environment variables.
# set export

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

test_notyping:
    . ./.venv/bin/activate
    pytest --pdb -x -s -v

# run one method, with args
run METHOD='reprpo_ortho' +EXTRA_ARGS='':
    #!/usr/bin/zsh
    export EXTRA_ARGS=${EXTRA_ARGS:-}
    source ./.venv/bin/activate
    python scripts/train.py {{METHOD}} $EXTRA_ARGS

run2 METHOD='reprpo_ortho' +EXTRA_ARGS='':
    #!/usr/bin/zsh
    export EXTRA_ARGS="{{EXTRA_ARGS}}"
    export METHOD="{{METHOD}}"
    echo "METHOD={{METHOD}} $METHOD EXTRA_ARGS=${EXTRA_ARGS}"

run_all +EXTRA_ARGS='':
    #!/usr/bin/zsh
    echo "REPR_CONFIG=$REPR_CONFIG"
    export WANDB_GROUP=${WANDB_GROUP:-mdl-$(date +%Y%m%d_%H%M%S)}
    echo "WANDB_GROUP=$WANDB_GROUP"

    # export HF_DATASETS_OFFLINE=1
    # export WANDB_MODE=offline
    # export WANDB_SILENT"]=true
    # export HF_DATASETS_DISABLE_PROGRESS_BARS=1

    export EXTRA_ARGS=${EXTRA_ARGS:-}
    echo "EXTRA_ARGS=$EXTRA_ARGS"

    . ./.venv/bin/activate
    for METHOD in sidein-ether dpo ether hra sideout-hra ortho sidein hra sidein-hra sideout hs svd sideout-ether hs-kl hs-dist side-dist; do
        echo "METHOD=$METHOD"
        python scripts/train.py $METHOD $EXTRA_ARGS
    done
    python scripts/train.py hra --no-rel-loss --verbose --lr 1e-5  $EXTRA_ARGS
    python scripts/train.py sidein-ether --Htype oft  $EXTRA_ARGS
    python scripts/train.py sidein-ether --Htype ether  $EXTRA_ARGS
    python scripts/train.py svd --quantile 1.0  $EXTRA_ARGS
    python scripts/train.py hra --no-apply_GS  $EXTRA_ARGS


run_ds:
    #!/usr/bin/zsh -x
    export REPR_CONFIG=./configs/llama3_7b.yaml
    source ./.venv/bin/activate

    export WANDB_GROUP=${WANDB_GROUP:-ds-$(date +%Y%m%d_%H%M%S)}
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
    for ds in $DS; do
        echo "DS=$ds"
        . ./.venv/bin/activate
        # python scripts/train.py sideout-ether --dataset $ds
        python scripts/train.py ether --dataset $ds
        python scripts/train.py sidein --dataset $ds
        python scripts/train.py dpo --dataset $ds
        python scripts/train.py hs-kl --dataset $ds
        python scripts/train.py side-dist --dataset $ds
        python scripts/train.py hs-dist --dataset $ds
    done



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
    python scripts/train.py hs-dist --verbose --n_samples=5000
