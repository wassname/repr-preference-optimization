
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


| adapter/distribution_shift                      | in_domain | difficulty_scaling | moral_transfer | orthogonal |
| :---------------------------------------------- | --------: | -----------------: | -------------: | ---------: |
| hs-None-InnerDPO  para_signed_log inc           |     0.966 |              0.594 |           0.35 |      0.646 |
| hs-None-InnerDPO para_signed_log inch v.overfit |     0.966 |              0.594 |           0.35 |      0.646 |
| none                                            |     0.868 |               0.78 |           0.27 |      0.414 |
| ipo? inco                                       |     0.966 |              0.798 |          0.366 |      0.558 |
| hs-None-InnerDPO logodds inch                   |     0.968 |              0.808 |          0.534 |      0.668 |
| dpo inco                                        |     0.948 |               0.86 |          0.316 |      0.468 |
| hs-None-InnerDPO cosine_pol_mar inch            |      0.96 |              0.876 |          0.264 |      0.634 |
| hs-None-InnerDPO para_signed inch ovefit        |      0.96 |              0.876 |          0.258 |       0.64 |
| hs-None-InnerDPO para_signed α=0.25 coheren     |      0.96 |              0.876 |          0.258 |       0.64 |

250608 03:22:17|INFO|reprpo.training:make_table#429 - Table 1: Absolute accuracy after training with named adapter on ds:`alpaca_easy` compared to base model `llama-3.2-3b-sft` for various distribution shifts [N=500]:
- Shift: difficulty_scaling, made up of:
        - `genies_preferences-alpaca_hard-test[:500]`
- Shift: in_domain, made up of:
        - `genies_preferences-alpaca_easy-test[:500]`
- Shift: moral_transfer, made up of:
        - `ethics_expression_preferences-justice-test[:500]`
- Shift: orthogonal, made up of:
        - `medical-dpo-v2-test-data[:500]`



I want to report nll ratio (should be [0-1]) otherwise incoherent




|                                                     | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb    |  nll |
| :-------------------------------------------------- | --------: | -----------------: | -------------: | ---------: | :------- | ---: |
| none                                                |     0.868 |               0.78 |           0.27 |      0.414 |          |      |
| ReprPO_None_InnerDPO ReprPO_Non α=0.25              |      0.96 |              0.864 |          0.294 |      0.602 | 61pyen4w |    0 |
| ReprPO_None_InnerDPO ReprPO_Non α=0.001             |     0.958 |              0.884 |          0.366 |      0.592 | 1ocenac8 |    0 |
| ReprPO_None_InnerDPO ReprPO_Non α=1                 |     0.954 |                0.9 |          0.322 |      0.598 | bm6zm8gk |    0 |
| ReprPO_None_InnerDPO ReprPO_Non α=10                |     0.954 |              0.858 |          0.318 |       0.51 | rx37mx51 |    0 |
| ReprPO_None_InnerDPO ReprPO_Non                     |     0.958 |              0.886 |          0.326 |      0.604 | kuheisul |    0 |
| dpo inco                                            |     0.948 |               0.86 |          0.316 |      0.468 |
| ipo? inco                                           |     0.966 |              0.798 |          0.366 |      0.558 |
| ReprPO_None_InnerDPO ReprPO_Non AlMe=logodds        |     0.966 |              0.876 |          0.324 |      0.568 | 97geelv4 |    0 |
| ReprPO_None_InnerDPO ReprPO_Non AlMe=cosine_pol_mar |     0.946 |              0.876 |          0.278 |        0.6 |
| hs-SupressedHS-InnerDPO                             |      0.96 |              0.896 |           0.33 |      0.606 |
 ReprPO_None_InnerDPO ReprPO_Non AlMe=stabili α=10 |       0.948 |                0.818 |             0.39 |        0.496 | mp90qyfl | 1.963 |
 | ReprPO_None_InnerDPO ReprPO_Non α=0.01 |        0.96 |                 0.89 |            0.306 |        0.598 | 0y9bq5z9 | 6.665 |

Hm the dff alphas don't seem to have much effect. Is the mostl cheating somehow. Perhaps for the ratio one its finding areas with a very small differen't, and making those very seperate in comparison to the tiny denominator, so the ratio is very large, and this dominated the mean. Hmm

ok! I made one that bounded the denominator to 10% percentile, and it's working, it's having an effect.

|                                                  |   in_domain |   difficulty_scaling |   moral_transfer |   orthogonal | wandb    |   nll |
|:-------------------------------------------------|------------:|---------------------:|-----------------:|-------------:|:---------|------:|
| ReprPO_None_InnerDPO ReprPO_Non AlMe=stabili α=1 |       0.964 |                0.858 |             0.44 |        0.468 | ncpdqhlh | 1.615 |
