#!/bin/bash
. ./.venv/bin/activate
python nbs/train.py --method dpo
python nbs/train.py --method reprpo_svd
python nbs/train.py --method reprpo_side
