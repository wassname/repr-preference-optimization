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
    python nbs/train.py --method reprpo_ortho  --verbose
    python nbs/train.py --method reprpo_side  --verbose
    python nbs/train.py --method dpo --verbose
    #python nbs/train.py --method reprpo_svd --verbose
