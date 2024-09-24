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

hp:
    WANDB_GROUP=prefvec python reprpo/training.py -m \
        +intervention=reprpo \
        +intervention.transform=ether \
        +intervention.loss=prefvec \
        'intervention.loss.use_nll_loss=True,False' \
        'intervention.loss.use_dpo_loss=True,False' \
        'intervention.loss.β=0.1,1.,2.' \
        'intervention.loss.use_angle_loss=True,False' \
        'intervention.loss.use_orth_loss=True,False'


    WANDB_GROUP=intervention.loss.use_nll_loss  python reprpo/training.py -m \
        '+intervention=reprpo' \
        '+intervention.transform=ether' \
        '+intervention.loss=glob(*)' \
        'intervention.loss.use_nll_loss=True,False'

    hydra/sweeper=ax


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

run2 METHOD='reprpo_ortho' +EXTRA_ARGS='':
    #!/usr/bin/zsh
    export EXTRA_ARGS="{{EXTRA_ARGS}}"
    export METHOD="{{METHOD}}"
    echo "METHOD={{METHOD}} $METHOD EXTRA_ARGS=${EXTRA_ARGS}"

run_all EXTRA_ARGS='':
    #!/usr/bin/zsh
    echo "REPR_CONFIG=$REPR_CONFIG"
    export WANDB_GROUP=${WANDB_GROUP:-mdl-$(date +%Y%m%d_%H%M%S)}
    echo "WANDB_GROUP=$WANDB_GROUP"

    # export HF_DATASETS_OFFLINE=1
    # export WANDB_MODE=offline
    # export WANDB_SILENT"]=true
    # export HF_DATASETS_DISABLE_PROGRESS_BARS=1

    export EXTRA_ARGS=${EXTRA_ARGS:---verbose=0}
    echo "EXTRA_ARGS=$EXTRA_ARGS"

    . ./.venv/bin/activate
    # for METHOD in ether-side-mse ether-side-rank ether-side-prefvec none-side-mse none-side-rank none-side-prefvec none-hs-prefvec none-hs-rank none-hs-mse ether-hs-rank ether-hs-prefvec hra-hs-prefvec ortho-hs-prefvec svd-hs-prefvec dpo; do
    for METHOD in none-hs-prefvec none-hs-rank none-hs-mse ether-hs-rank ether-hs-prefvec hra-hs-prefvec ortho-hs-prefvec svd-hs-prefvec dpo; do
        echo "METHOD=$METHOD"
        python scripts/train.py $METHOD $EXTRA_ARGS
    done
    # python scripts/train.py hra --no-rel-loss --verbose --lr 1e-5  $EXTRA_ARGS
    # python scripts/train.py sidein-ether --Htype oft  $EXTRA_ARGS
    # python scripts/train.py sidein-ether --Htype ether  $EXTRA_ARGS
    # python scripts/train.py svd --quantile 1.0  $EXTRA_ARGS
    # python scripts/train.py hra --no-apply_GS  $EXTRA_ARGS


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
        python scripts/train.py none-side-prefvec --dataset $ds
        python scripts/train.py ether-side-prefvec --dataset $ds
        python scripts/train.py ether-hs-prefvec --dataset $ds
        # python scripts/train.py dpo --dataset $ds
    done



run_llama:
    #!/usr/bin/zsh
    export REPR_CONFIG=./configs/llama3_7b.yaml
    just run_all
    just run_ds


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
