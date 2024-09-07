
[private]
default:
    @just --list

run:
    . ./.venv/bin/activate
    python nbs/train.py --method reprpo_side  --verbose
    python nbs/train.py --method dpo --verbose
    #python nbs/train.py --method reprpo_svd --verbose
