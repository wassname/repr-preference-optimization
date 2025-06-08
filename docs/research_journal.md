
# 2025-06-07 13:54:52

Shower thoughts:
- Taking absolute hs distance is not enougth, bercaus then you are maximising existing distance, and the optimiser will find it hard to flip, and this would involve many steps of negative loss! So it should be abs( hs_diff * learned_sign ), that way the learned sign can flip the direction of the loss, and the optimiser can find it easier to flip the preference, but it wont change the loss, only the direction.
- Also might want to try just a normal sft nll loss


| adapter/distribution_shift   |   in_domain |   difficulty_scaling |   moral_transfer |   orthogonal |
|:-----------------------------|------------:|---------------------:|-----------------:|-------------:|
| none                         |       0.868 |                0.78  |            0.27  |        0.414 |
| hs-None-InnerDPO coh            |       0.944 |                0.796 |            0.242 |        0.438 |
| hs-None-InnerDPO coh            |       0.93  |                0.798 |            0.268 |        0.414 |
| hs-None-InnerDPO inco            |       0.962 |                 0.86 |             0.3  |        0.528 |
| dpo inco                          |       0.948 |                 0.86 |            0.316 |        0.468 |
| ipo? inco                          |       0.966 |                0.798 |            0.366 |        0.558 |
| hs-ETHER-InnerDPO            |       0.964 |                0.868 |            0.302 |        0.518 |
| hs-SupressedHS-InnerDPO  incoh     |       0.96  |                 0.87 |            0.276 |        0.586 |

Ok I had a think and signed loss actually makes sense as we are using the data preference not the models



ok so they don't overfit now, better rscedule helps

```sh
# new signed
python scripts/train.py hs-none-InnerDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=para
python scripts/train.py hs-none-InnerDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=para
python scripts/train.py hs-none-InnerDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=para
loss.
# old
python scripts/train.py hs-none-InnerDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=para_orth2
python scripts/train.py hs-none-InnerDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=para_orth
python scripts/train.py hs-none-InnerDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=orth
python scripts/train.py hs-none-InnerDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=direct_projection
python scripts/train.py hs-none-InnerDPO -direct_projection_log

set -x
export METHODS=(
    "para_signed"
    "para_signed_log"
    "para_orth_signed"
    "para_orth_signed_logpara_orth_signed_logpara_orth_signed_log"
    "logodds"
    "cosine_policy_margin"
    "cosine_cross_model"
)
for method in "${METHODS[@]}"; do
    python scripts/train.py hs-none-InnerDPO --loss.align_method="$method"
done
```
