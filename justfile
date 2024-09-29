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
    #!/usr/bin/bash
    # export HF_DATASETS_OFFLINE=1
    # export WANDB_MODE=offline
    # export WANDB_SILENT"]=true
    # export HF_DATASETS_DISABLE_PROGRESS_BARS=1

    . ./.venv/bin/activate
    export WANDB_GROUP=${WANDB_GROUP:-mdl-$(date +%Y%m%d_%H%M%S)}
    export EXTRA_ARGS=${EXTRA_ARGS:---verbose=0}
    echo "REPR_CONFIG=$REPR_CONFIG"
    echo "WANDB_GROUP=$WANDB_GROUP"
    echo "EXTRA_ARGS=$EXTRA_ARGS"

    readarray -t EXPERIMENTS <<< "$(python ./scripts/export_experiments.py)"

    echo EXPERIMENTS $EXPERIMENTS $S
    for METHOD in $EXPERIMENTS; do
        echo "python scripts/train.py $METHOD $EXTRA_ARGS"
        python scripts/train.py "$METHOD" "$EXTRA_ARGS"
    done

# export_exp:
#     #!/usr/bin/bash
#     . ./.venv/bin/activate
#     export EXPERIMENTS=$(python ./scripts/export_experiments.py)
#     IFS=' ' read -r -a DS <<< "$STRING"
    
#     # echo $EXPERIMENTS

#     export DS=(
#         alpaca_easy
#         alpaca_mmlu 
#         alpaca_low_quality 
#         alpaca_short 
#         code_easy 
#         alpaca_mmlu 
#         math 
#         raven_matrices  
#         us_history_textbook
#     )


run_ds:
    #!/usr/bin/zsh
    export REPR_CONFIG=./configs/llama3_7b.yaml
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

    export WANDB_GROUP=${WANDB_GROUP:-ds-$(date +%Y%m%d_%H%M%S)}
    . ./.venv/bin/activate
    export METHODS=(
        dpo
        side-ether-prefvec
        prefvec
    )
    echo $DS
    for ds in $DS; do
        echo "DS=$ds"
        for METHOD in $EXPERIMENTS; do
            echo python scripts/train.py $METHOD --dataset $ds
            python scripts/train.py $METHOD --dataset $ds
        done
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
    # -x
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
    python scripts/train.py projgrad
    python scripts/train.py dpo
    python scripts/train.py side-ether-prefvec

run_pg:
    #!/usr/bin/zsh
    export WANDB_GROUP=${WANDB_GROUP:-ds-$(date +%Y%m%d_%H%M%S)}
    #python scripts/train.py projgrad --n-samples=6000 --verbose=1
    # python scripts/train.py projgrad
    export REPR_CONFIG=./configs/llama3_7b.yaml
    python scripts/train.py projgrad_right --verbose=1
    python scripts/train.py projgrad_left --verbose=1
    python scripts/train.py projgrad_fb
    python scripts/train.py projgrad_fs
    python scripts/train.py projgrad_fs2
    python scripts/train.py projgrad_fs3
    python scripts/train.py projgrad_fs4
    python scripts/train.py projgrad_bs3
    python scripts/train.py projgrad --β=0.0 --neg-slope=1.0 --verbose=1
    python scripts/train.py dpo --verbose=1
    # python scripts/train.py projgrad --β=1.0 --ignore-direction 
    python scripts/train.py projgrad --β=0.5 --neg-slope=0.05
    python scripts/train.py projgrad --β=1.0 --neg-slope=1.0 # should be like dpo. yes
    python scripts/train.py projgrad --no-scale-orth --no-reverse_pref
    python scripts/train.py projgrad --no-reverse-pref
    python scripts/train.py projgrad --weight-dim=1
    python scripts/train.py projgrad --weight-dim=2

    python scripts/train.py projgrad --β=0.8 --neg-slope=0.1 --mag-clip=0.2 # soft constraint
