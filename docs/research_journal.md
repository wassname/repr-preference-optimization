
# 2025-06-07 13:54:52

Shower thoughts:
- Taking absolute hs distance is not enougth, bercaus then you are maximising existing distance, and the optimiser will find it hard to flip, and this would involve many steps of negative loss! So it should be abs( hs_diff * learned_sign ), that way the learned sign can flip the direction of the loss, and the optimiser can find it easier to flip the preference, but it wont change the loss, only the direction.
- Also might want to try just a normal sft nll loss


| adapter/distribution_shift  | in_domain | difficulty_scaling | moral_transfer | orthogonal |
| :-------------------------- | --------: | -----------------: | -------------: | ---------: |
| none                        |     0.868 |               0.78 |           0.27 |      0.414 |
| hs-None-InDPO coh           |     0.944 |              0.796 |          0.242 |      0.438 |
| hs-None-InDPO coh           |      0.93 |              0.798 |          0.268 |      0.414 |
| hs-None-InDPO inco          |     0.962 |               0.86 |            0.3 |      0.528 |
| dpo inco                    |     0.948 |               0.86 |          0.316 |      0.468 |
| ipo? inco                   |     0.966 |              0.798 |          0.366 |      0.558 |
| hs-ETHER-InDPO              |     0.964 |              0.868 |          0.302 |      0.518 |
| hs-SupressedHS-InDPO  incoh |      0.96 |               0.87 |          0.276 |      0.586 |

Ok I had a think and signed loss actually makes sense as we are using the data preference not the models



ok so they don't overfit now, better rscedule helps

```sh
# new signed
python scripts/train.py hs-none-InDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=para
python scripts/train.py hs-none-InDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=para
python scripts/train.py hs-none-InDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=para
loss.
# old
python scripts/train.py hs-none-InDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=para_orth2
python scripts/train.py hs-none-InDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=para_orth
python scripts/train.py hs-none-InDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=orth
python scripts/train.py hs-none-InDPO --base_model=wassname/llama-3.2-3b-sft --loss.align_method=direct_projection
python scripts/train.py hs-none-InDPO -direct_projection_log

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
    python scripts/train.py hs-none-InDPO --loss.align_method="$method"
done
```


| adapter/distribution_shift                   | in_domain | difficulty_scaling | moral_transfer | orthogonal |
| :------------------------------------------- | --------: | -----------------: | -------------: | ---------: |
| hs-None-InDPO  para_signed_log inc           |     0.966 |              0.594 |           0.35 |      0.646 |
| hs-None-InDPO para_signed_log inch v.overfit |     0.966 |              0.594 |           0.35 |      0.646 |
| none                                         |     0.868 |               0.78 |           0.27 |      0.414 |
| ipo? inco                                    |     0.966 |              0.798 |          0.366 |      0.558 |
| hs-None-InDPO logodds inch                   |     0.968 |              0.808 |          0.534 |      0.668 |
| dpo inco                                     |     0.948 |               0.86 |          0.316 |      0.468 |
| hs-None-InDPO cosine_pol_mar inch            |      0.96 |              0.876 |          0.264 |      0.634 |
| hs-None-InDPO para_signed inch ovefit        |      0.96 |              0.876 |          0.258 |       0.64 |
| hs-None-InDPO para_signed α=0.25 coheren     |      0.96 |              0.876 |          0.258 |       0.64 |

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




|                                         | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb    |   nll |
| :-------------------------------------- | --------: | -----------------: | -------------: | ---------: | :------- | ----: |
| none                                    |     0.868 |               0.78 |           0.27 |      0.414 |          |       |
| RPO_N_InDPO RPO_Non α=0.25              |      0.96 |              0.864 |          0.294 |      0.602 | 61pyen4w |     0 |
| RPO_N_InDPO RPO_Non α=0.001             |     0.958 |              0.884 |          0.366 |      0.592 | 1ocenac8 |     0 |
| RPO_N_InDPO RPO_Non α=1                 |     0.954 |                0.9 |          0.322 |      0.598 | bm6zm8gk |     0 |
| RPO_N_InDPO RPO_Non α=10                |     0.954 |              0.858 |          0.318 |       0.51 | rx37mx51 |     0 |
| RPO_N_InDPO RPO_Non                     |     0.958 |              0.886 |          0.326 |      0.604 | kuheisul |     0 |
| dpo inco                                |     0.948 |               0.86 |          0.316 |      0.468 |
| ipo? inco                               |     0.966 |              0.798 |          0.366 |      0.558 |
| RPO_N_InDPO RPO_Non AlMe=logodds        |     0.966 |              0.876 |          0.324 |      0.568 | 97geelv4 |     0 |
| RPO_N_InDPO RPO_Non AlMe=cosine_pol_mar |     0.946 |              0.876 |          0.278 |        0.6 |
| hs-SupressedHS-InDPO                    |      0.96 |              0.896 |           0.33 |      0.606 |
| RPO_N_InDPO RPO_Non AlMe=stabili α=10   |     0.948 |              0.818 |           0.39 |      0.496 | mp90qyfl | 1.963 |
| RPO_N_InDPO RPO_Non α=0.01              |      0.96 |               0.89 |          0.306 |      0.598 | 0y9bq5z9 | 6.665 |

Hm the dff alphas don't seem to have much effect. Is the mostl cheating somehow. Perhaps for the ratio one its finding areas with a very small differen't, and making those very seperate in comparison to the tiny denominator, so the ratio is very large, and this dominated the mean. Hmm

ok! I made one that bounded the denominator to 10% percentile, and it's working, it's having an effect.

|                                      | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb    |   nll |
| :----------------------------------- | --------: | -----------------: | -------------: | ---------: | :------- | ----: |
| RPO_N_InDPO RPO_Non AlMe=stabili α=1 |     0.964 |              0.858 |           0.44 |      0.468 | ncpdqhlh | 1.615 |
| RPO_N_InDPO RPO_Non AlMe=logodds α=1 |     0.962 |               0.83 |          0.442 |      0.478 | 0z3xsjc3 | 1.534 |


|                                         | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb    |   nll |
| :-------------------------------------- | --------: | -----------------: | -------------: | ---------: | :------- | ----: |
| RPO_N_InDPO RPO_Non AlMe=stabili α=10   |      0.95 |              0.824 |          0.414 |      0.488 | 14rwlkhf | 2.047 |
| RPO_N_InDPO RPO_Non AlMe=logodds α=10   |     0.944 |              0.834 |          0.406 |       0.45 | iy48gij7 | 0.734 |
| RPO_N_InDPO RPO_Non AlMe=odds_no α=1    |     0.956 |               0.89 |          0.292 |      0.584 | x17oaz0r | 6.137 |
| RPO_N_InDPO RPO_Non AlMe=odds_no α=10   |     0.972 |              0.848 |          0.346 |      0.452 | pbig3muv | 1.585 |
| RPO_N_InDPO RPO_Non α=0.01              |      0.96 |               0.89 |          0.306 |      0.598 | 221acpzk | 6.665 |
| RPO_N_InDPO RPO_Non α=0.25              |      0.96 |              0.864 |          0.294 |      0.602 | z3kgnv2d | 6.577 |
| RPO_N_InDPO RPO_Non α=0.001             |     0.958 |              0.884 |          0.366 |      0.592 | 0lwti0vl |  6.31 |
| RPO_N_InDPO RPO_Non α=1                 |     0.954 |                0.9 |          0.322 |      0.598 | a7r8owsa | 6.359 |
| RPO_N_InDPO RPO_Non α=10                |     0.954 |              0.858 |          0.318 |       0.51 | o5uqhidf | 4.739 |
| RPO_N_InDPO RPO_Non AlMe=stabili        |      0.95 |              0.888 |          0.316 |       0.58 | mfrav3nc | 4.935 |
| RPO_N_InDPO RPO_Non AlMe=logodds        |     0.974 |              0.874 |          0.288 |      0.508 | 1ra6k7f2 | 2.969 |
| RPO_N_InDPO RPO_Non AlMe=odds_no        |     0.968 |              0.886 |          0.324 |       0.59 | 255io1os | 5.323 |
| RPO_N_InDPO RPO_Non                     |     0.958 |              0.886 |          0.326 |      0.604 | 6bsokle2 | 6.607 |
| RPO_N_InDPO RPO_Non AlMe=logodds        |      0.96 |              0.816 |           0.42 |      0.454 | mncx88v4 | 1.083 |
| RPO_N_InDPO RPO_Non AlMe=para_si        |      0.96 |               0.88 |          0.298 |        0.6 | l216tdcw | 5.812 |
| RPO_N_InDPO RPO_Non AlMe=para_or        |     0.958 |              0.882 |          0.314 |      0.596 | od447zdl | 5.956 |
| RPO_N_InDPO RPO_Non AlMe=para_or        |     0.962 |              0.882 |          0.316 |      0.592 | wvbcku7k | 6.785 |
| RPO_N_InDPO RPO_Non AlMe=cosine_        |     0.946 |              0.876 |          0.278 |        0.6 | jj0uject | 6.549 |
| RPO_N_InDPO RPO_Non AlMe=cosine_        |     0.954 |               0.88 |          0.314 |      0.606 | zeg3jdx8 | 6.745 |
| ReprPO_SupressedHS_InDPO ReprPO_Sup     |      0.96 |              0.896 |           0.33 |      0.606 | 48ufw1xz | 6.814 |
| ReprPO_ETHER_InDPO ReprPO_ETH           |     0.958 |              0.874 |           0.31 |       0.59 | 2uqub786 | 5.418 |
| RPO_N_InDPO RPO_Non                     |      0.96 |              0.884 |          0.312 |      0.584 | m3u5aklf | 6.277 |
| RPO_N_InDPO RPO_Non NoBeRe=0            |     0.964 |              0.884 |          0.284 |      0.602 | 5c4ah3ut | 5.915 |
| RPO_N_InDPO RPO_Non UsPoWe=1            |     0.966 |              0.882 |          0.326 |      0.608 | jrdabdnj | 6.041 |
| RPO_N_InDPO RPO_Non DpAgTy=dpo          |     0.976 |              0.828 |          0.284 |      0.412 | vkkq6kq3 | 0.228 |
| RPO_N_InDPO RPO_Non                     |     0.958 |              0.886 |          0.326 |      0.604 | 1q8fb4hp | 6.607 |
| DPO DPO                                 |     0.958 |               0.91 |          0.286 |      0.618 | 9iue8gbp | 8.163 |
| none                                    |     0.868 |               0.78 |           0.27 |      0.414 |          |     0 |
| RPO_N_InDPO RPO_Nonv2 AlMe=logodds α=10 |     0.926 |               0.82 |          0.392 |      0.444 | 6jxrx283 | 0.759 |
| RPO_N_InDPO RPO_Nonv2 AlMe=stabili α=10 |     0.964 |              0.822 |          0.464 |      0.464 | ci9ztfj2 | 1.003 |
| RPO_N_InDPO RPO_Non AlMe=odds_no α=1    |     0.964 |               0.85 |          0.396 |      0.426 | bh83z6cf | 1.028 |



Table 1b: Absolute accuracy after training with named adapter on ds:`alpaca_easy` compared to base model `llama-3.2-3b-sft` for various distribution shifts [N=500]:
- Shift: difficulty_scaling, made up of:
        - `genies_preferences-alpaca_hard-test[:500]`
- Shift: in_domain, made up of:
        - `genies_preferences-alpaca_easy-test[:500]`
- Shift: moral_transfer, made up of:
        - `ethics_expression_preferences-justice-test[:500]`
- Shift: orthogonal, made up of:
        - `medical-dpo-v2-test-data[:500]`


|                                        | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb      | nll_cho/ref |
| :------------------------------------- | --------: | -----------------: | -------------: | ---------: | :--------- | ----------: |
| RPO_N_InDPO RPO_Non AlMe=stabili α=10  |     0.964 |              0.822 |          0.464 |      0.464 | wjistjxd   |       1.003 |
| RPO_N_InDPO RPO_Non AlMe=stabili       |     0.964 |              0.854 |           0.43 |      0.452 | t0iglbsy   |       1.484 |
| RPO_N_InDPO RPO_Non AlMe=logodds       |      0.96 |               0.83 |          0.436 |      0.456 | l95sq1fg   |       0.873 |
| RPO_N_InDPO RPO_Non AlMe=para_or       |     0.956 |              0.848 |          0.426 |      0.454 | yfrcoy8t   |       1.423 |
| RPO_N_InDPO RPO_Non AlMe=log_rat       |     0.934 |              0.826 |          0.398 |      0.432 | z5oznicj   |       0.563 |
| RPO_N_InDPO RPO_Non AlMe=ratio_d       |     0.892 |              0.794 |          0.274 |      0.396 | l5dh15wl   |      -0.086 |
| RPO_N_InDPO RPO_Non AlMe=log_rat α=10  |      0.91 |              0.802 |          0.292 |      0.398 | svbkt8jn   |      -0.081 |
| RPO_N_InDPO RPO_Non AlMe=ratio_d α=10  |      0.88 |               0.79 |           0.27 |      0.398 | u02meyve   |       -0.06 |
| RPO_N_InDPO RPO_Non AlMe=log_rat α=0.2 |     0.942 |               0.82 |          0.394 |      0.448 | kfarsdk6   |       0.901 |
| RPO_N_InDPO RPO_Non AlMe=ratio_d α=0.2 |     0.932 |              0.826 |          0.362 |      0.404 | y13nltlv   |      -0.127 |
| RPO_N_InDPO RPO_Non AlMe=logodds α=10  |     0.926 |               0.82 |          0.392 |      0.444 | v7mck3xn   |       0.759 |
| RPO_N_InDPO RPO_Non AlMe=odds_no α=10  |      0.96 |              0.826 |          0.396 |      0.424 | svwpu608 d |       0.716 |
| RPO_N_InDPO RPO_Non α=10               |      0.94 |              0.814 |          0.364 |      0.424 | duz4snn0 d |       1.037 |
| RPO_N_InDPO RPO_Non α=0.01             |     0.958 |              0.888 |          0.318 |       0.61 | hh1ccy5a   |        5.77 |
| RPO_N_InDPO RPO_Non α=0.25             |      0.97 |              0.872 |          0.288 |      0.556 | 8y8b52i9   |       4.503 |
| RPO_N_InDPO RPO_Non α=0.001            |     0.962 |              0.878 |          0.254 |      0.588 | islgc0j4   |       6.644 |
| RPO_N_InDPO RPO_Non                    |     0.966 |              0.864 |          0.312 |      0.488 | cdcoeils d |       2.015 |
| RPO_N_InDPO RPO_Non AlMe=log_rat       |     0.934 |              0.826 |          0.398 |      0.432 | 31tyci5s   |       0.563 |
| RPO_N_InDPO RPO_Non AlMe=ratio_d       |     0.892 |              0.794 |          0.274 |      0.396 | lj4yc37u   |      -0.086 |
| RPO_N_InDPO RPO_Non AlMe=logodds       |      0.94 |              0.824 |          0.394 |      0.436 | o3nn4uwj   |       0.553 |
| RPO_N_InDPO RPO_Non AlMe=odds_no       |     0.964 |               0.85 |          0.396 |      0.426 | xleo40cn   |       1.028 |
| RPO_N_InDPO RPO_Non                    |     0.966 |              0.864 |          0.312 |      0.488 | 4h3zyksf d |       2.015 |
| RPO_N_InDPO RPO_Non AlMe=para_si       |      0.96 |              0.874 |          0.294 |      0.486 | bei4lat3 d |       1.961 |
| RPO_N_InDPO RPO_Non AlMe=para_or       |     0.964 |              0.852 |          0.396 |      0.426 | e04tr8tb   |       1.024 |
| RPO_N_InDPO RPO_Non AlMe=cosine_       |     0.962 |              0.878 |           0.32 |      0.608 | uhf9mjvm   |       6.477 |
| RPO_N_InDPO RPO_Non AlMe=cosine_       |     0.958 |              0.872 |          0.284 |      0.586 | v54fff3l   |       5.587 |
| ReprPO_SupressedHS_InDPO ReprPO_Sup    |     0.976 |              0.896 |           0.31 |       0.49 | 35qri7ve   |       1.948 |
| DPO DPO DpAgTy=dpo                     |      0.98 |               0.83 |           0.21 |      0.548 | qfrdh6z2   |       4.497 |
| DPO DPO                                |     0.958 |               0.91 |          0.286 |      0.618 | 6vja6bei   |       8.163 |
| DPO DPO                                |     0.958 |               0.91 |          0.286 |      0.618 | gqra479b   |       8.163 |
| none                                   |     0.868 |               0.78 |           0.27 |      0.414 |
| RPO_N_InDPO RPO_Non α=10               |      0.94 |              0.814 |          0.364 |      0.424 | kjl2kh1j   |       1.037 |


|                                                | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb      | nll_cho/ref |
| :--------------------------------------------- | --------: | -----------------: | -------------: | ---------: | :--------- | ----------: |
| RPO_N_InDPO RPO_Non para_signed                |     0.966 |              0.864 |          0.312 |      0.488 | cdcoeils d |       2.015 |
| RPO_N_InDPO RPO_Non para_signed                |     0.966 |              0.864 |          0.312 |      0.488 | 4h3zyksf d |       2.015 |
| RPO_N_InDPO  AlMe=para_signed_log              |      0.96 |              0.874 |          0.294 |      0.486 | bei4lat3 d |       1.961 |
| DPO DPO                                        |     0.958 |               0.91 |          0.286 |      0.618 | gqra479b   |       8.163 |
| none                                           |     0.868 |               0.78 |           0.27 |      0.414 |
| RPO_N_InDPO  InPoWe=1 NoLa=1                   |      0.95 |              0.812 |           0.35 |      0.446 | vhlm28dj   |       1.225 |
| RPO_N_InDPO InPoWe=1                           |      0.95 |              0.812 |           0.35 |      0.446 | 2ks26l53   |       1.225 |
| RPO_N_InDPO InPoWe=1 NoBeRe=1 NoLa=1           |     0.958 |              0.836 |          0.308 |      0.494 | 4tiaj1o0   |       4.279 |
| RPO_N_InDPO RPO_Non α=20 para_signed_log long! |      0.89 |              0.802 |          0.354 |      0.296 | 8r67vdo6   |      16.411 |
| RPO_N_InDPO RPO_Non para_signed α=10           |      0.94 |              0.814 |          0.364 |      0.424 | duz4snn0 d |       1.037 |
| RPO_N_InDPO  AlMe=odds_noref α=10              |      0.96 |              0.826 |          0.396 |      0.424 | svwpu608 d |       0.716 |
| RPO_N_InDPO AlMe=odds_noref α=20 long!         |     0.964 |              0.798 |          0.412 |      0.582 | wkeaftud   |      18.611 |

250609 14:58:35|INFO|reprpo.training:make_table#441 - Table 1: Absolute accuracy after training with named adapter on ds:`alpaca_easy` compared to base model `llama-3.2-3b-sft` for various distribution shifts [N=500]:
- Shift: difficulty_scaling, made up of:
        - `genies_preferences-alpaca_hard-test[:500]`
- Shift: in_domain, made up of:
        - `genies_preferences-alpaca_easy-test[:500]`
- Shift: moral_transfer, made up of:
        - `ethics_expression_preferences-justice-test[:500]`
- Shift: orthogonal, made up of:
        - `medical-dpo-v2-test-data[:500]`



ok I tried 1b with 8bit.... incoherent
what about without 8bit?


incoherent again, maybe it's that 1b
it was! damn




Try with other 1b, 34min, bs=16
|                                   | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb    |                       nll_cho/ref |
| :-------------------------------- | --------: | -----------------: | -------------: | ---------: | :------- | --------------------------------: |
| none                              |     0.944 |              0.819 |          0.388 |      0.389 |
| DPO                               |     0.916 |              0.877 |          0.432 |      0.409 | lbijyagd |                             3.713 |
| ReprPO_None_InnerDPO AlMe=logodds |     0.947 |              0.887 |          0.594 |      0.487 | q522xaug | 4.401 no inner learn bc tiny, coh |
| ReprPO_None_InnerDPO AlMe=odds_no |     0.935 |              0.855 |          0.523 |      0.505 | u2d73apj |            5.836 inner learn, coh |
| ReprPO_None_InnerDPO para_s_l     |     0.923 |              0.792 |          0.432 |      0.576 | xbs8008n |                      10.969 incho |
| ReprPO_SupressedHS_para_s_l       |     0.903 |              0.857 |          0.466 |      0.621 | n1mts1jk |                      22.151 incho |
| ReprPO_None_InnerDPO AlMe=odds_no |     0.544 |               0.36 |           0.31 |      0.376 | 74yrnnne |                            -0.002 |
| ReprPO_None_InnerDPO α=1          |     0.925 |              0.849 |          0.341 |      0.501 | 84bm674r |                             7.039 inco |

 met 8bit seems bad
python scripts/train.py hs-none-InnerDPO --loss.align_method=odds_noref --loss.α=20
python scripts/train.py hs-ether-InnerDPO --loss.align_method=odds_noref --loss.α=20
python scripts/train.py hs-supr-InnerDPO --loss.align_method=odds_noref --loss.α=20
python scripts/train.py hs-none-InnerDPO --loss.align_method=logodds_noref --loss.α=100  --batch-size=18
python scripts/train.py hs-none-InnerDPO --loss.align_method=para_signed_log --loss.α=1  --batch-size=18


python scripts/train.py hs-supr-InnerDPO --loss.align_method=odds_noref --loss.α=10
python scripts/train.py hs-supr-InnerDPO --loss.α=10

try 8bit bs=24, logodds,
and higher lpha... no wait it was high

250610 07:24:47|INFO|reprpo.training:make_table#442 - Table 1: Absolute accuracy after training with named adapter on ds:`alpaca_easy` compared to base model `OLMo-2-0425-1B-SFT` for various distribution shifts [N=750]:
- Shift: difficulty_scaling, made up of:
        - `genies_preferences-alpaca_hard-test[:750]`
- Shift: in_domain, made up of:
        - `genies_preferences-alpaca_easy-test[:750]`
- Shift: moral_transfer, made up of:
        - `ethics_expression_preferences-justice-test[:750]`
- Shift: orthogonal, made up of:
        - `medical-dpo-v2-test-data[:750]`


TODO turns out medical is not a good orthogonal shift. I should try obscure ranking task, or logic, or code, or ethics?
<<<<<<< HEAD
<<<<<<< HEAD


TODO It's working but incoherent. Lets try para_signed / mag_ref+eps  Or  para_signed / ref.norm()?


python scripts/train.py hs-supr-InnerDPO --loss.align_method=pars_rat
python scripts/train.py hs-supr-InnerDPO --loss.align_method=pars_ratref
python scripts/train.py hs-supr-InnerDPO --loss.align_method=pars_ratref_log
python scripts/train.py hs-supr-InnerDPO --loss.align_method=pars_rat_log
=======
>>>>>>> origin/vast2a
=======
>>>>>>> origin/vast2a
