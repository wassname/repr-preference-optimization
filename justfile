# set shell := ["zsh", "-uc"]


export CUDA_VISIBLE_DEVICES := "0"
export TOKENIZERS_PARALLELISM := "false"

[private]
default:
    @just --list

sweep:
    #!/usr/bin/zsh
    rm -f sweep.sh
    mv outputs outputs_$(date +%Y-%m-%d_%H-%M-%S) || true
    python scripts/sweep.py > sweep.sh
    unbuffer bash sweep.sh  2>&1 | tee sweep.txt



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

        # extreme
        arc_easy
        math_easy
        ranking_logic_easy
        raven_easy
        code
        cooking
        pursue_goals
        creative_writing
        code_low_quality
        shp_low_quality
        math_make_questions
        math_textbook
        math_fiction
        us_history_make_questions
        change_my_view
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
    rsync -avz --ignore-existing runpod:/workspace/repr-preference-optimization/outputs/ ./ouputs/

scratch2:
    #!/usr/bin/zsh
    set -x
    export CUDA_VISIBLE_DEVICES=0
    export WANDB_GROUP=${WANDB_GROUP:-prefvecexp-$(date +%y%m%d_%H%M)}
    # export ARGS='--batch-size=10'
    # baseline
    # couple of quick ones to test
    python scripts/train.py hs-none-InnerDPO --verbose=2 --n_samples=4000 --eval_samples=200
    python scripts/train.py dpo --verbose=2 --n_samples=4000 --eval_samples=200

    python scripts/train.py hs-none-InnerDPO --loss.align-method=direct_projection
    python scripts/train.py dpo
    python scripts/train.py hs-none-InnerDPO --loss.align-method=para_orth
    python scripts/train.py hs-none-InnerDPO --loss.align-method=orth
    python scripts/train.py dpo --use-policy-weights
    python scripts/train.py hs-none-InnerDPO --loss.align-method=para
    python scripts/train.py hs-none-InnerDPO --loss.align-method=para_orth2 --verbose=2
    python scripts/train.py hs-none-InnerDPO --loss.align-method=angle_mag

    python scripts/train.py hs-none-InnerDPO --loss.no-norm-before-reduce
    python scripts/train.py hs-none-InnerDPO --use-policy-weights
    python scripts/train.py hs-none-InnerDPO --collection-layers='range(0.3, 0.6, 4)'

    python scripts/train.py hs-none-InnerDPO --loss.align-method=cosine_similarity
    python scripts/train.py hs-none-InnerDPO --loss.align-method=abs
    python scripts/train.py hs-none-InnerDPO --loss.align-method=log_ratio

    python scripts/train.py side-none-InnerDPO 
    python scripts/train.py hs-ether-InnerDPO
    python scripts/train.py hs-surp-InnerDPO

    python scripts/train.py hs-ether-rank
    python scripts/train.py hs-ether-mse

    python scripts/train.py projgrad --no-use-pref-ref
    python scripts/train.py projgrad
    python scripts/train.py projbp

    just sweep

