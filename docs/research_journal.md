
# 2025-06-07 13:54:52

Shower thoughts:
- Taking absolute hs distance is not enougth, bercaus then you are maximising existing distance, and the optimiser will find it hard to flip, and this would involve many steps of negative loss! So it should be abs( hs_diff * learned_sign ), that way the learned sign can flip the direction of the loss, and the optimiser can find it easier to flip the preference, but it wont change the loss, only the direction.
- Also might want to try just a normal sft nll loss


| adapter/distribution_shift     | in_domain | difficulty_scaling | moral_transfer | orthogonal |
| :----------------------------- | --------: | -----------------: | -------------: | ---------: |
| none                           |     0.868 |               0.78 |           0.27 |      0.414 |
| hs-None-InnerDPO coh           |     0.944 |              0.796 |          0.242 |      0.438 |
| hs-None-InnerDPO coh           |      0.93 |              0.798 |          0.268 |      0.414 |
| hs-None-InnerDPO inco          |     0.962 |               0.86 |            0.3 |      0.528 |
| dpo inco                       |     0.948 |               0.86 |          0.316 |      0.468 |
| ipo? inco                      |     0.966 |              0.798 |          0.366 |      0.558 |
| hs-ETHER-InnerDPO              |     0.964 |              0.868 |          0.302 |      0.518 |
| hs-SupressedHS-InnerDPO  incoh |      0.96 |               0.87 |          0.276 |      0.586 |

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


| adapter/distribution_shift       | in_domain | difficulty_scaling | moral_transfer | orthogonal |
| :------------------------------- | --------: | -----------------: | -------------: | ---------: |
| none                             |     0.868 |               0.78 |           0.27 |      0.414 |
| hs-None-InnerDPO para_signed inch ovefit     |      0.96 |              0.876 |          0.258 |       0.64 |
| hs-None-InnerDPO para_signed_log inch v.overfit |     0.966 |              0.594 |           0.35 |      0.646 |
| dpo inco                         |     0.948 |               0.86 |          0.316 |      0.468 |
| ipo? inco                        |     0.966 |              0.798 |          0.366 |      0.558 |

250608 03:22:17|INFO|reprpo.training:make_table#429 - Table 1: Absolute accuracy after training with named adapter on ds:`alpaca_easy` compared to base model `llama-3.2-3b-sft` for various distribution shifts [N=500]:
- Shift: difficulty_scaling, made up of:
        - `genies_preferences-alpaca_hard-test[:500]`
- Shift: in_domain, made up of:
        - `genies_preferences-alpaca_easy-test[:500]`
- Shift: moral_transfer, made up of:
        - `ethics_expression_preferences-justice-test[:500]`
- Shift: orthogonal, made up of:
        - `medical-dpo-v2-test-data[:500]`
