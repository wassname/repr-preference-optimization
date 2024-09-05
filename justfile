
[private]
default:
    @just --list

run:
    . ./.venv/bin/activate
    python nbs/train.py --method dpo
    python nbs/train.py --method reprpo_side
    python nbs/train.py --method reprpo_svd
