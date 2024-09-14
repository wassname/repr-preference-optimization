set shell := ["zsh", "-cu"]

# settings
set dotenv-load

# Export all just variables as environment variables.
set export

export CUDA_VISIBLE_DEVICES := "0"
export TOKENIZERS_PARALLELISM := "false"

[private]
default:
    @just --list

# run one method, by argument
run method='reprpo_ortho':
    . ./.venv/bin/activate
    python nbs/train.py {{method}} --verbose

run_all:
    # TODO set group by env var and time
    . ./.venv/bin/activate
    # python nbs/train2.py sidein-hra
    # python nbs/train2.py sideout
    python nbs/train2.py hra
    python nbs/train2.py sideout-hra
    python nbs/train2.py sidein
    python nbs/train2.py hs
    python nbs/train2.py ortho
    python nbs/train2.py svd
    python nbs/train2.py svd --quantile 1.0
    python nbs/train2.py dpo


llama_all:
    export REPR_CONFIG=../configs/llama3_7b.yaml
    . ./.venv/bin/activate
    python nbs/train2.py reprpo_side_hra
    python nbs/train2.py reprpo_sideout
    python nbs/train2.py reprpo_side
    python nbs/train2.py reprpo_hs
    python nbs/train2.py reprpo_hra
    python nbs/train2.py reprpo_ortho
    python nbs/train2.py reprpo_svd
    python nbs/train2.py reprpo_svd --quantile 1.0
    python nbs/train2.py dpo
