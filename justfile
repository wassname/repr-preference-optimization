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
    python nbs/train.py --method {{method}} --verbose

run_all:
    . ./.venv/bin/activate
    python nbs/train.py --method reprpo_side_hra
    python nbs/train.py --method reprpo_sideout
    python nbs/train.py --method reprpo_hs
    python nbs/train.py --method reprpo_hra
    python nbs/train.py --method reprpo_ortho
    python nbs/train.py --method reprpo_side
    python nbs/train.py --method dpo
    python nbs/train.py --method reprpo_svd
    python nbs/train.py --method reprpo_svd --quantile 1.0
