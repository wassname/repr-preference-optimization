#!/bin/bash
python nbs/train --method dpo
python nbs/train --method reprpo_svd
python nbs/train --method reprpo_side
