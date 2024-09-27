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
    for METHOD in \
        projgrad \
        dpo \
        side-ether-prefvec \
        hs-ether-prefvec \
        side-ether-mse \
        side-ether-rank \
        side-none-mse \
        side-none-rank \
        side-none-prefvec \
        hs-none-prefvec \
        hs-none-rank \
        hs-none-mse \
        hs-ether-rank \
        hs-hra-prefvec \
        hs-ortho-prefvec \
        hs-svd-prefvec \
        dpo; do
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
        python scripts/train.py dpo --dataset $ds
        python scripts/train.py side-ether-prefvec --dataset $ds
        python scripts/train.py side-none-rank --dataset $ds
        python scripts/train.py side-none-mse --dataset $ds
        python scripts/train.py hs-ether-prefvec --dataset $ds
        python scripts/train.py side-none-prefvec --dataset $ds
    done


run_hp:
    export WANDB_GROUP=${WANDB_GROUP:-ds-$(date +%Y%m%d_%H%M%S)}
    python scripts/train.py side-ether-prefvec --loss.β 0.04
    python scripts/train.py side-ether-prefvec --loss.β 0.08 --loss.no-use_orth_loss --loss.use_angle_loss
    python scripts/train.py side-ether-prefvec --loss.β 0.3 --loss.no-use_orth_loss --loss.no-use_angle_loss
    python scripts/train.py side-ether-prefvec --loss.no-use_dpo_loss --loss.no-use_orth_loss 
    python scripts/train.py side-ether-prefvec --loss.no-use_dpo_loss --loss.no-use_orth_loss --loss.weight_tokens
    python scripts/train.py side-ether-prefvec --loss.no-use_nll_loss --loss.no-use_orth_loss --loss.weight_tokens --loss.use_angle_loss
    python scripts/train.py side-ether-prefvec --transform.Htype=ether
    python scripts/train.py side-ether-prefvec --transform.Htype=oft
    python scripts/train.py side-ether-prefvec --transform.Htype=HH

    python scripts/train.py hs-ether-prefvec --transform.Htype=ether --transform.nb=16 --transform.reduction=1
    python scripts/train.py hs-ether-prefvec --transform.Htype=oft --transform.nb=16 --transform.reduction=1
    python scripts/train.py hs-ether-prefvec --transform.Htype=HH --transform.nb=16 --transform.reduction=1
    just run_ds





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

run_pg:
    export WANDB_GROUP=${WANDB_GROUP:-ds-$(date +%Y%m%d_%H%M%S)}
    python scripts/train.py projgrad --n-samples=6000 --verbose=1
    python scripts/train.py dpo --verbose=1
    # python scripts/train.py projgrad
    python scripts/train.py projgrad --lr=1e-6 --verbose=1
    python scripts/train.py projgrad --β=0.0 --negative-slope=1.0 --verbose=1
    python scripts/train.py projgrad --β=0.0
    python scripts/train.py projgrad --β=0.0
    python scripts/train.py projgrad --β=0.1
    python scripts/train.py projgrad --β=0.5
    # python scripts/train.py projgrad --β=1.0 --ignore-direction 
    python scripts/train.py projgrad --β=0.5 --negative-slope=0.05
    python scripts/train.py projgrad --β=1.0 --negative-slope=1.0 # should be like dpo. yes
    python scripts/train.py projgrad --lr=1e-7
    python scripts/train.py projgrad --lr=1e-4
    python scripts/train.py projgrad --lr=1e-3
    python scripts/train.py projgrad --lr=1e-3

    python scripts/train.py projgrad --β=0.8 --negative-slope=0.1 --magnitude-clip=0.2 # soft constraint
