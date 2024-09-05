
[private]
default:
    just --list

# clear outputs
clear:
    rm -rf ./outputs

# run all methods
run:
    . ./.venv/bin/activate
    python nbs/train.py --method reprpo_svd
    python nbs/train.py --method reprpo_side
    python nbs/train.py --method dpo
