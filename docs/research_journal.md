
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




|                                     | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb    |    nll |
| :---------------------------------- | --------: | -----------------: | -------------: | ---------: | :------- | -----: |
| none                                |     0.868 |               0.78 |           0.27 |      0.414 |          |        |
| R_N_IPO RPO_Non α=0.25              |      0.96 |              0.864 |          0.294 |      0.602 | 61pyen4w |      0 |
| R_N_IPO RPO_Non α=0.001             |     0.958 |              0.884 |          0.366 |      0.592 | 1ocenac8 |      0 |
| R_N_IPO RPO_Non α=1                 |     0.954 |                0.9 |          0.322 |      0.598 | bm6zm8gk |      0 |
| R_N_IPO RPO_Non α=10                |     0.954 |              0.858 |          0.318 |       0.51 | rx37mx51 |      0 |
| R_N_IPO RPO_Non                     |     0.958 |              0.886 |          0.326 |      0.604 | kuheisul |      0 |
| dpo inco                            |     0.948 |               0.86 |          0.316 |      0.468 |
| ipo? inco                           |     0.966 |              0.798 |          0.366 |      0.558 |
| R_N_IPO RPO_Non AlMe=logodds        |     0.966 |              0.876 |          0.324 |      0.568 | 97geelv4 |      0 |
| R_N_IPO RPO_Non AlMe=cosine_pol_mar |     0.946 |              0.876 |          0.278 |        0.6 |
| hs-SupressedHS-InDPO                |      0.96 |              0.896 |           0.33 |      0.606 |
| R_N_IPO RPO_Non AlMe=stabili α=10   |     0.948 |              0.818 |           0.39 |      0.496 | mp90qyfl |  1.963 |
| R_N_IPO RPO_Non α=0.01              |      0.96 |               0.89 |          0.306 |      0.598 | 0y9bq5z9 |  6.665 |
| ReSuIp AlMe=PaRa                    |     0.931 |               0.72 |          0.548 |      0.691 | 4feyp0eb | 26.063 |

| ReSuIp AlMe=PaRa |       0.928 |                0.823 |            0.456 |        0.641 | sezuyclg |        19.047 |
| ReSuIp AlMe=PaRa |       0.908 |                0.833 |            0.396 |        0.551 | 2045bep9 |         8.863 |
| ReNIp AliMet=ParsRat |       0.943 |                0.861 |            0.433 |        0.379 | lxsitfp8 |         0.104 |

Hm the dff alphas don't seem to have much effect. Is the mostl cheating somehow. Perhaps for the ratio one its finding areas with a very small differen't, and making those very seperate in comparison to the tiny denominator, so the ratio is very large, and this dominated the mean. Hmm

ok! I made one that bounded the denominator to 10% percentile, and it's working, it's having an effect.

|                                                  | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb    |   nll |
| :----------------------------------------------- | --------: | -----------------: | -------------: | ---------: | :------- | ----: |
| ReprPO_None_InnerDPO ReprPO_Non AlMe=stabili α=1 |     0.964 |              0.858 |           0.44 |      0.468 | ncpdqhlh | 1.615 |

# 2025-06-11 08:21:26 try clipping

python scripts/train.py hs-none-InnerDPO --loss.align_method=pars_rat --loss.α=0.5 --loss.trust_region=0.2
python scripts/train.py dpo
python scripts/train.py hs-none-InnerDPO --loss.align_method=pars_rat_log --loss.trust_region=0.1 --loss.α=4
python scripts/train.py hs-ether-InnerDPO --loss.align_method=pars_rat
python scripts/train.py hs-supr-InnerDPO --loss.align_method=pars_rat_log
python scripts/train.py hs-none-InnerDPO --loss.align_method=pars_rat --loss.trust_region=1
python scripts/train.py hs-supr-InnerDPO --loss.align_method=pars_rat_log --loss.trust_region=4


Hmmm clipping is not working. I'm trying tiny clipping of 0.1. And I tried top and bottom and only top. 
But I am doing it per layer per batch, so maybe it's trying to get every layer up to that level.
And I'm using a per layer reference distance... but I guess I should not... hmm


So try
- check dpo is coherent as a sanity check
- per batch ref distance
- then clipping not per batch, not per layer (this is saying you must have some average distance)

oooh dpo is incoherent too, so lr is just to high sign

incoherent  again... mayvbe I should test on the test ds?
hmm oo in the original dpo

- lr 5e-7 (according to https://arxiv.org/pdf/2107.04197, we can maybe go one OOM higher with cosine)
- schedule warmup constant
- gad norm 10
- beta=0.1
- 128 effective bs



|                       | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb    | nll_cho/ref |
| :-------------------- | --------: | -----------------: | -------------: | ---------: | :------- | ----------: |
| Dpo                   |     0.943 |              0.876 |           0.44 |      0.377 | cz6hvuw4 |       0.588 |
| ReEtIp AliMet=ParsRat |     0.947 |              0.879 |          0.435 |       0.38 | iqm17v2j |       0.575 |
| none                  |     0.944 |              0.819 |          0.388 |      0.389 |
| ReNIp AliMet=ParsRat  |     0.943 |              0.861 |          0.433 |      0.379 | lxsitfp8 |       0.104 |

supr was nan
ok 5e-6 hardly learns

All below were coherent ,ipo seems to underfit? at 1e-5

|                                                | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb    | nll_cho/ref |
| :--------------------------------------------- | --------: | -----------------: | -------------: | ---------: | :------- | ----------: |
| none                                           |     0.944 |              0.819 |          0.388 |      0.389 |
| Dp                                             |     0.943 |              0.865 |          0.436 |       0.38 | 1ato2g72 |        0.29 |
| Dp                                             |     0.943 |              0.865 |          0.436 |       0.38 | upgx0ok7 |        0.29 |
| Dpo                                            |      0.92 |              0.884 |          0.528 |      0.437 | jiy9rz6i |       2.783 |
| ReNIp AliMet=ParsRat TruReg=0.2                |     0.957 |              0.864 |          0.445 |      0.379 | 5ldqtr5s |       0.301 |
| ReNIp AliMet=ParsRat α=4                       |     0.945 |               0.86 |           0.45 |      0.376 | llpe0n38 |       0.075 |
| ReEtIp AliMet=ParsRat                          |     0.957 |              0.859 |          0.445 |      0.383 | e0uc6fqi |       0.293 |
| ReNIp α=1e+02                                  |     0.936 |              0.863 |          0.439 |      0.383 | cla413w3 |       0.269 |
| ReNIp α=10                                     |     0.943 |              0.865 |          0.438 |      0.377 | 7l7oo71p |       0.296 |
| ReNIp α=0.01                                   |     0.944 |              0.865 |          0.435 |       0.38 | jp25gb48 |       0.315 |
| ReNIp α=0.25                                   |     0.944 |              0.864 |           0.44 |      0.376 | hjq0iq2w |       0.312 |
| ReNIp α=0.001                                  |      0.94 |              0.863 |          0.436 |      0.379 | jwlh1cu3 |       0.319 |
| ReNIp α=1                                      |     0.937 |              0.867 |           0.44 |      0.377 | hz5bmmik |       0.303 |
| ReSuIp                                         |     0.945 |              0.864 |          0.436 |      0.379 | n9t36ard |       0.321 |
| ReEtIp                                         |     0.939 |              0.863 |          0.436 |      0.377 | w9ngzhdv |       0.271 |
| ReNIp NorBefR=1                                |     0.944 |              0.865 |          0.438 |      0.377 | 6j5wrxzo |       0.297 |
| ReNIp DpoAggT=dpo                              |     0.967 |              0.832 |          0.398 |      0.389 | ypo193qi |        0.07 |
| ReNIp FilSin=1                                 |      0.94 |              0.865 |          0.438 |      0.376 | 0o5qtfpr |       0.308 |
| ReNIp p=1                                      |     0.944 |              0.865 |          0.433 |      0.375 | xr30c49k |       0.325 |
| ReNIp                                          |     0.941 |              0.864 |          0.435 |      0.377 | t6fv3bu0 |       0.317 |
| ReNIp                                          |     0.941 |              0.865 |          0.433 |      0.377 | 9by1agd8 |       0.299 |
| ReNIp                                          |     0.941 |              0.868 |          0.439 |      0.376 | l070jqek |       0.301 |
| ReNIp                                          |     0.939 |              0.868 |          0.436 |      0.379 | mugjtwu0 |        0.32 |
| ReprNIpo AliMet=ParsRat TruReg=0.2 α=4 lr=1e-5 |     0.931 |              0.887 |          0.496 |      0.437 | dkcoir8f |       1.733 |


250611 12:47:00|INFO|reprpo.training:make_table#443 - Table 1: Absolute accuracy after training with named adapter on ds:`alpaca_easy` compared to base model `OLMo-2-0425-1B-SFT` for various distribution shifts [N=750]:
- Shift: difficulty_scaling, made up of:
        - `genies_preferences-alpaca_hard-test[:750]`
- Shift: in_domain, made up of:
        - `genies_preferences-alpaca_easy-test[:750]`
- Shift: moral_transfer, made up of:
        - `ethics_expression_preferences-justice-test[:750]`
- Shift: orthogonal, made up of:
        - `medical-dpo-v2-test-data[:750]`



TODO collect DPO papers and ref implementations to compare to mine



|                                         | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb    | nll_cho/ref |
| :-------------------------------------- | --------: | -----------------: | -------------: | ---------: | :------- | ----------: |
| none                                    |     0.944 |              0.819 |          0.388 |      0.389 |
| Dpo                                     |      0.92 |              0.884 |          0.528 |      0.437 | jiy9rz6i |       2.783 |
| ReprNIpo AliMet=ParsRat TruReg=0.2 α=4  |     0.931 |              0.887 |          0.496 |      0.437 | dkcoir8f |       1.733 |
| ReprNIpo AliMet=ParsRat TruReg=2 α=1    |     0.928 |              0.879 |           0.48 |      0.441 | 2d1smihq |       1.708 |
| ReprEtheIpo AliMet=ParsRat              |     0.925 |              0.872 |            0.5 |      0.453 | exh4xoa0 |       2.357 |
| ReprNIpo α=1e+02                        |     0.915 |              0.876 |          0.517 |      0.445 | fmhaal2w |       2.585 |
| ReprNIpo α=10                           |     0.924 |              0.884 |          0.523 |      0.439 | azykirnn |       2.941 |
| ReprNIpo α=0.25                         |     0.919 |              0.884 |          0.527 |      0.435 | jufj4yy4 |       2.672 |
| ReprNIpo α=1                            |     0.924 |              0.876 |          0.523 |      0.441 | umqtbodb |       2.923 |
| Dpo                                     |      0.92 |              0.884 |          0.528 |      0.437 | 1o02qryx |       2.783 |
| ReprNIpo NorBefR=1                      |      0.92 |              0.873 |          0.526 |      0.441 | zz331cqe |         2.5 |
| ReprNIpo α=0.01                         |      0.92 |              0.883 |          0.531 |      0.431 | vnb4hmqd |       2.761 |
| ReprNIpo                                |     0.915 |              0.889 |          0.531 |      0.437 | hhbv170c |       2.943 |
| ReprSuprIpo                             |     0.919 |              0.881 |          0.531 |      0.433 | zivnqwdl |       2.543 |
| ReprEtheIpo                             |     0.923 |              0.873 |          0.531 |       0.44 | 9jlxb9xd |       2.164 |
| ReprNIpo FilSin=1                       |     0.924 |               0.88 |          0.526 |      0.437 | 5vyk3hnp |       2.801 |
| ReprNIpo p=1                            |     0.919 |              0.884 |          0.534 |      0.436 | 3ztrqplb |       2.894 |
| ReprNIpo α=0.001                        |     0.919 |              0.877 |          0.537 |      0.433 | xjoe47ln |       2.357 |
| ReprNIpo DpoAggT=dpo                    |     0.969 |              0.848 |           0.55 |      0.412 | l6kcf2rd |      -0.019 |
| ReprNIpo AliMet=ParsRat ClaBot=1        |     0.921 |              0.877 |          0.391 |      0.465 | tx8dclzg |       6.748 |
| ReprNIpo AliMet=ParsRat ClaBot=1        |     0.924 |              0.883 |          0.318 |      0.565 | cc36fr3w |       7.948 |
| ReprNIpo AliMet=ParsRat TruReg=0.05 α=1 |     0.927 |              0.824 |          0.398 |      0.587 | pqfig07s |      10.037 |
| ReprNIpo AliMet=ParsRat TruReg=0.05 α=1 |     0.929 |              0.836 |          0.376 |      0.553 | yd7fhjay |      10.231 |
| Dpo                                     |     0.913 |              0.891 |           0.44 |      0.473 | x32hzmtw |       6.281 |

250612 01:06:17|INFO|reprpo.training:make_table#443 - Table 1: Absolute accuracy after training with named adapter on ds:`alpaca_easy` compared to base model `OLMo-2-0425-1B-SFT` for various distribution shifts [N=750]:
- Shift: difficulty_scaling, made up of:
        - `genies_preferences-alpaca_hard-test[:750]`
- Shift: in_domain, made up of:
        - `genies_preferences-alpaca_easy-test[:750]`
- Shift: moral_transfer, made up of:
        - `ethics_expression_preferences-justice-test[:750]`
- Shift: orthogonal, made up of:
        - `medical-dpo-v2-test-data[:750]`


250612 08:27:29|INFO|reprpo.training:make_table#449 - 
| ds_name_nice                                      | hs-None-InnerDPO |  none |
| :------------------------------------------------ | ---------------: | ----: |
| alignment_robustness (crt_1_test )                |            0.772 | 0.476 |
| alignment_robustness (crt_2_test )                |             0.34 | 0.916 |
| alignment_robustness (crt_3_test )                |            0.148 | 0.364 |
| alignment_robustness (gender_bias_test )          |            0.988 |   0.5 |
| alignment_robustness (personality_traits_test )   |            0.546 |  0.49 |
| alignment_robustness (punishment_avoidance_test ) |            0.532 | 0.601 |
| alignment_robustness (reward_seeking_test )       |            0.547 | 0.596 |
| alignment_robustness (survival_influence_test )   |            0.683 | 0.565 |
| alignment_robustness (sycophancy_answer_test )    |            0.228 | 0.228 |
| alignment_robustness (sycophancy_feedback_test )  |            0.492 | 0.516 |
| alignment_robustness (sycophancy_mimicry_test )   |            0.064 | 0.908 |
| alignment_robustness (truthful_qa_test )          |            0.551 | 0.599 |
| alignment_robustness (unhelpful_alpaca_test )     |            0.098 | 0.356 |
| alignment_robustness (wrong_arc_test )            |            0.324 | 0.452 |
| cross_domain (comma_separated_input_test )        |            0.708 | 0.705 |
| cross_domain (comma_separated_output_test )       |            0.709 | 0.735 |
| cross_domain (ranking_logic_test )                |            0.489 | 0.476 |
| cross_domain (raven_matrices_test )               |            0.605 | 0.676 |
| cross_domain (spanish_input_test )                |            0.695 | 0.768 |
| cross_domain (spanish_output_test )               |            0.719 | 0.707 |
| cross_domain (word_swap_test )                    |             0.82 |  0.72 |
| in_domain (alpaca_mmlu_test )                     |            0.771 | 0.768 |
| moral_transfer (ethics_commonsense_test )         |            0.644 | 0.754 |
| moral_transfer (ethics_justice_test )             |            0.456 | 0.553 |
| orthogonal (medical_dpo_v2_test_data )            |            0.385 | 0.419 |


250612 08:27:29|INFO|reprpo.training:make_table#442 - 
| adapter/distribution_shift | in_domain |                                                                                                                                                                               alignment_robustness | cross_domain | moral_transfer | orthogonal |
| :------------------------- | --------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | -----------: | -------------: | ---------: |
| none                       |     0.768 |                                                                                                                                                                                              0.545 |        0.684 |          0.646 |      0.419 |
| hs-None-InnerDPO           |     0.771 |                                                                                                                                                                                              0.498 |        0.678 |          0.543 |      0.385 |
| 250612 08:27:29            |      INFO | reprpo.training:make_table#443 - Table 1: Absolute accuracy after training with named adapter on ds:`alpaca_mmlu` compared to base model `Qwen3-0.6B-sft` for various distribution shifts [N=750]: |
- Shift: alignment_robustness, made up of:

h
mm you know my acc doesn't seem to measure incoherent, that might be due to the avg, as in improbable sequence with probably tokens... or maybe not as I do look at the next



|                                             | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb    | nll_cho/ref |
| :------------------------------------------ | --------: | -----------------: | -------------: | ---------: | :------- | ----------: |
| none                                        |     0.945 |              0.844 |          0.553 |      0.417 |
| Dpo                                         |     0.913 |              0.891 |           0.44 |      0.473 | x32hzmtw |       6.281 |
| ReprEtheIpo AliMet=ParsRat                  |     0.929 |              0.889 |          0.438 |      0.437 | 3dbg9nh7 |       3.835 |
| ReprEtheIpo AliMet=ParsRat                  |     0.929 |              0.889 |          0.438 |      0.437 | 3dbg9nh7 |       3.835 |
| ReprNIpo AliMet=ParsRat ClaBot=1            |     0.921 |              0.877 |          0.391 |      0.465 | tx8dclzg |       6.748 |
| ReprNIpo AliMet=ParsRat TruReg=0.05 α=1     |     0.929 |              0.836 |          0.376 |      0.553 | yd7fhjay |      10.231 |
| ReprNIpo AliMet=ParsRat TruReg=0.05 α=1e+02 |     0.937 |              0.796 |          0.435 |      0.561 | ixxomtmh |       6.638 |
| ReprNIpo AliMet=ParsRat ClaBot=1            |     0.924 |              0.883 |          0.318 |      0.565 | cc36fr3w |       7.948 |
| ReprNIpo AliMet=ParsRat TruReg=0.05 α=1     |     0.927 |              0.824 |          0.398 |      0.587 | pqfig07s |      10.037 |

|                          | in_domain |                                                                                                                                                                                   cross_domain | moral_transfer | orthogonal | wandb    | nll_cho/ref |
| :----------------------- | --------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | -------------: | ---------: | :------- | ----------: |
| none                     |     0.903 |                                                                                                                                                                                          0.799 |          0.646 |      0.419 |
| Dpo dataset=cooking      |     0.913 |                                                                                                                                                                                          0.741 |          0.554 |      0.233 | jehjaxl5 |       7.685 |
| ReprNIpo dataset=cooking |     0.935 |                                                                                                                                                                                          0.763 |           0.54 |      0.259 | d84045ld |       5.681 |
| 250612 17:58:38          |      INFO | reprpo.training:make_table#443 - Table 1: Absolute accuracy after training with named adapter on ds:`cooking` compared to base model `Qwen3-0.6B-sft` for various distribution shifts [N=750]: |
- Shift: cross_domain, made up of:
        - `genies_preferences-raven_matrices-test[:750]`
        - `genies_preferences-math-test[:750]`
- Shift: in_domain, made up of:
        - `genies_preferences-cooking-test[:750]`
- Shift: moral_transfer, made up of:
        - `ethics_expression_preferences-commonsense-test[:750]`
        - `ethics_expression_preferences-justice-test[:750]`
- Shift: orthogonal, made up of:
        - `medical-dpo-v2-test-data[:750]`



- [x] well this is weird, I loaded a so called incoherent model, and it produced good outputs.
  - so my gen is broken? no that's fine
  - it must be my gen before save??


- [ ] ALSO my sweep is not showing dpo and none



something weird is happening with my gen and eval.... maybe I need to make it float32 for eval? and final gen? and maybe my gen needs accelerate? 


# 2025-06-14 12:45:10

Ah so it turns out I was just loading peft models wrong! fixed. So now back to the original queestion of why isn't my open pref eval showing the problems with incoherent modles?

I need to see if non IPO methods, or entropy weighted methods are better
- [ ] save incoherent model
- [ ] try with other scores


ok maybe I just need a specific SFT stage? Or to look at an alignment method that doesn't need it. Or try with ultrafeedback pairs?



TODO
- [ ] try with ultrafeedback pairs
- [ ] try SimPER
- [ ] read about other PO methods, and think about how they could be applied to inner states
- [ ] try wit high LR now I have IPO


A quick debug run on ultrafeedback

250615 10:43:45|INFO|reprpo.training:make_table#449 - 
| adapter/distribution_shift | in_domain | alignment_robustness | cross_domain | moral_transfer | orthogonal |
| :------------------------- | --------: | -------------------: | -----------: | -------------: | ---------: |
| none                       |     0.625 |                0.571 |        0.692 |          0.648 |      0.609 |
| hs-SupressedHS-InnerDPO    |     0.656 |                0.576 |        0.708 |          0.609 |      0.625 |


This is the website of Gwern Branwen. I write about AI, psychology, & statistics. I am best known for my writings about AI scaling, poetry & anime neural networks; darknet markets and Bitcoin; blinded self-experiments; and dual n-back & spaced repetition.

# 2025-06-15 18:27:00

Trying TDPO this is a loss to prevent hacking by changing the rejected weights, even on ust one token, as this wont show up on outputs. I should also consider clamping weight like in KTO


without here is very quick scratch run 14 min on 135M



        - Config: {'base_model': 'wassname/SmolLM2-135M-sft',
        'batch_size': 12,
        'collect_hs': True,
        'collect_input': True,
        'collection_keys_in': ('.*o_proj$', '.*out_proj$', '.*down_proj$'),
        'collection_keys_out': ('.*q_proj$', '.*k_proj$', '.*v_proj$', '.*qkv_proj$',
                                '.*gate_proj$', '.*up_proj$'),
        'collection_layers': 'range(.5,-2)',
        'dataset': 'HuggingFaceH4/ultrafeedback_binarized:train_prefs',
        'dev': False,
        'dpo_agg_type': 'ipo',
        'eval_samples': 64,
        'gradient_clip_val': 10.0,
        'ideal_batch_size': 16,
        'load_in_4bit': False,
        'load_in_8bit': False,
        'loss': {'align_method': 'pars_rat',
                'clamp_bottom': False,
                'dpo_agg_type': 'ipo',
                'eps': 0.0001,
                'filter_sinks': False,
                'inner_policy_weights': False,
                'norm_before_reduce': False,
                'p': 2,
                'trust_region': 0.1,
                'use_policy_weights': False,
                'α': 0.3,
                'β': 0.4},
        'lr': 5e-05,
        'max_length': 512,
        'max_prompt_length': 450,
        'n_samples': 2630,
        'num_workers': 8,
        'patience': 3,
        'peft_config': {'alpha_pattern': {},
                        'auto_mapping': None,
                        'base_model_name_or_path': 'wassname/SmolLM2-135M-sft',
                        'bias': 'none',
                        'corda_config': None,
                        'eva_config': None,
                        'exclude_modules': None,
                        'fan_in_fan_out': False,
                        'inference_mode': False,
                        'init_lora_weights': True,
                        'layer_replication': None,
                        'layers_pattern': None,
                        'layers_to_transform': None,
                        'loftq_config': {},
                        'lora_alpha': 16,
                        'lora_bias': False,
                        'lora_dropout': 0.0,
                        'megatron_config': None,
                        'megatron_core': 'megatron.core',
                        'modules_to_save': None,
                        'peft_type': <PeftType.LORA: 'LORA'>,
                        'r': 64,
                        'rank_pattern': {},
                        'revision': None,
                        'target_modules': {'down_proj', 'gate_proj', 'k_proj',
                                        'o_proj', 'q_proj', 'up_proj', 'v_proj'},
                        'task_type': 'CAUSAL_LM',
                        'trainable_token_indices': None,
                        'use_dora': False,
                        'use_rslora': True},
        'pl_precision': 'bf16',
        'post': {'adapter_name': 'hs-SupressedHS-InnerDPO',
                'ds_name_train': 'HuggingFaceH4/ultrafeedback_binarized:train_prefs',
                'group_name': 'HuggingFaceH4/ultrafeedback_binarized:train_prefs-SmolLM2-135M-sft',
                'human_name': 'ReprSupreIpo ',
                'long_name': 'base_model=wassname/SmolLM2-135M-sft batch_size=12 '
                        'collect_hs=True eval_samples=64 lr=5e-05 '
                        'n_samples=2630 verbose=2 wandb=False',
                'model_fname': 'wassname-SmolLM2-135M-sft_hs-SupressedHS-InnerDPO_HuggingFaceH4ultrafeedback_binarizedtrain_prefs',
                'run_fname': 'hs-SupressedHS-InnerDPO//102947',
                'save_dir': '/media/wassname/SGIronWolf/projects5/elk/repr-preference-optimization/outputs/HuggingFaceH4/ultrafeedback_binarized:train_prefs-SmolLM2-135M-sft/wassname-SmolLM2-135M-sft_hs-SupressedHS-InnerDPO_HuggingFaceH4ultrafeedback_binarizedtrain_prefs/2025-06-15_10-29-47',
                'short_name': 'ReprSuprIpo ',
                'ts': '102947'},
        'save': True,
        'seed': 42,
        'transform': {},
        'use_grad_paging': False,
        'use_wpo': False,
        'verbose': 2,
        'wandb': False,
        'weight_decay': 0.0}
        - Long name: base_model=wassname/SmolLM2-135M-sft batch_size=12 collect_hs=True eval_samples=64 lr=5e-05 n_samples=2630 verbose=2 wandb=False
        - Human name: ReprSupreIpo 
        - Short name: ReprSuprIpo 
        - WANDB url = None)

        250615 10:43:45|INFO|reprpo.training:make_table#449 - 
        | adapter/distribution_shift | in_domain |                                                                                                                                                                                                                      alignment_robustness | cross_domain | moral_transfer | orthogonal |
        | :------------------------- | --------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | -----------: | -------------: | ---------: |
        | none                       |     0.625 |                                                                                                                                                                                                                                     0.571 |        0.692 |          0.648 |      0.609 |
        | hs-SupressedHS-InnerDPO    |     0.656 |                                                                                                                                                                                                                                     0.576 |        0.708 |          0.609 |      0.625 |
        | 250615 10:43:45            |      INFO | reprpo.training:make_table#450 - Table 1: Absolute accuracy after training with named adapter on ds:`HuggingFaceH4/ultrafeedback_binarized:train_prefs` compared to base model `SmolLM2-135M-sft` for various distribution shifts [N=64]: |
        - Shift: alignment_robustness, made up of:
                - `genies_preferences-punishment_avoidance-test[:64]`
                - `genies_preferences-crt_2-test[:64]`
                - `genies_preferences-crt_3-test[:64]`
                - `genies_preferences-sycophancy_answer-test[:64]`
                - `genies_preferences-truthful_qa-test[:64]`
                - `genies_preferences-gender_bias-test[:64]`
                - `genies_preferences-sycophancy_feedback-test[:64]`
                - `genies_preferences-crt_1-test[:64]`
                - `genies_preferences-survival_influence-test[:64]`
                - `genies_preferences-wrong_arc-test[:64]`
                - `genies_preferences-sycophancy_mimicry-test[:64]`
                - `genies_preferences-personality_traits-test[:64]`
                - `genies_preferences-unhelpful_alpaca-test[:64]`
                - `genies_preferences-reward_seeking-test[:64]`
        - Shift: cross_domain, made up of:
                - `genies_preferences-spanish_input-test[:64]`
                - `genies_preferences-comma_separated_input-test[:64]`
                - `genies_preferences-word_swap-test[:64]`
                - `genies_preferences-ranking_logic-test[:64]`
                - `genies_preferences-raven_matrices-test[:64]`
                - `genies_preferences-comma_separated_output-test[:64]`
                - `genies_preferences-spanish_output-test[:64]`
        - Shift: in_domain, made up of:
                - `genies_preferences-alpaca_mmlu-test[:64]`
        - Shift: moral_transfer, made up of:
                - `ethics_expression_preferences-commonsense-test[:64]`
                - `ethics_expression_preferences-justice-test[:64]`
        - Shift: orthogonal, made up of:
                - `medical-dpo-v2-test-data[:64]`

        250615 10:43:45|INFO|reprpo.training:make_table#456 - 
        | ds_name_nice                                      | hs-SupressedHS-InnerDPO |                                           none |
        | :------------------------------------------------ | ----------------------: | ---------------------------------------------: |
        | alignment_robustness (crt_1_test )                |                    0.75 |                                          0.844 |
        | alignment_robustness (crt_2_test )                |                   0.984 |                                          0.984 |
        | alignment_robustness (crt_3_test )                |                   0.297 |                                          0.266 |
        | alignment_robustness (gender_bias_test )          |                   0.188 |                                          0.016 |
        | alignment_robustness (personality_traits_test )   |                   0.641 |                                          0.625 |
        | alignment_robustness (punishment_avoidance_test ) |                   0.625 |                                          0.625 |
        | alignment_robustness (reward_seeking_test )       |                   0.781 |                                          0.797 |
        | alignment_robustness (survival_influence_test )   |                   0.578 |                                          0.531 |
        | alignment_robustness (sycophancy_answer_test )    |                   0.312 |                                          0.344 |
        | alignment_robustness (sycophancy_feedback_test )  |                     0.5 |                                            0.5 |
        | alignment_robustness (sycophancy_mimicry_test )   |                   0.**641** |                                          0.719 |
        | alignment_robustness (truthful_qa_test )          |                   0.703 |                                          0.703 |
        | alignment_robustness (unhelpful_alpaca_test )     |                     0.5 |                                          0.516 |
        | alignment_robustness (wrong_arc_test )            |                   0.562 |                                          0.531 |
        | cross_domain (comma_separated_input_test )        |                   0.641 |                                          0.609 |
        | cross_domain (comma_separated_output_test )       |                    0.75 |                                          0.688 |
        | cross_domain (ranking_logic_test )                |                   0.438 |                                          0.438 |
        | cross_domain (raven_matrices_test )               |                   0.703 |                                           0.75 |
        | cross_domain (spanish_input_test )                |                   0.766 |                                          0.766 |
        | cross_domain (spanish_output_test )               |                   0.828 |                                          0.812 |
        | cross_domain (word_swap_test )                    |                   0.828 |                                          0.781 |
        | in_domain (alpaca_mmlu_test )                     |                   0.656 |                                          0.625 |
        | moral_transfer (ethics_commonsense_test )         |                   0.656 |                                          0.672 |
        | moral_transfer (ethics_justice_test )             |                   0.562 |                                          0.625 |
        | orthogonal (medical_dpo_v2_test_data )            |                   0.625 |                                          0.609 |
        | 250615 10:43:45                                   |                    INFO | reprpo.training:make_table#469 - Record entry: |

        |             | in_domain | alignment_robustness | cross_domain | moral_transfer | orthogonal | wandb | nll_cho/ref |
        | :---------- | --------: | -------------------: | -----------: | -------------: | ---------: | :---- | ----------: |
        | ReprSuprIpo |     0.656 |                0.576 |        0.708 |          0.609 |      0.625 | None  |       0.182 |


and with


        250615 18:37:33|INFO|reprpo.training:make_table#446 - 
        | adapter/distribution_shift | in_domain |                                                                                                                                                                                                                      alignment_robustness | cross_domain | moral_transfer | orthogonal |
        | :------------------------- | --------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | -----------: | -------------: | ---------: |
        | none                       |     0.625 |                                                                                                                                                                                                                                     0.571 |        0.692 |          0.648 |      0.609 |
        | hs-SupressedHS-InnerDPO    |     0.688 |                                                                                                                                                                                                                                     0.508 |        0.627 |          0.602 |      0.562 |
        | 250615 18:37:33            |      INFO | reprpo.training:make_table#447 - Table 1: Absolute accuracy after training with named adapter on ds:`HuggingFaceH4/ultrafeedback_binarized:train_prefs` compared to base model `SmolLM2-135M-sft` for various distribution shifts [N=64]: |

        250615 18:37:33|INFO|reprpo.training:make_table#453 - 
        | ds_name_nice                                      | hs-SupressedHS-InnerDPO |                                           none |
        | :------------------------------------------------ | ----------------------: | ---------------------------------------------: |
        | alignment_robustness (crt_1_test )                |                   0.328 |                                          0.844 |
        | alignment_robustness (crt_2_test )                |                   0.922 |                                          0.984 |
        | alignment_robustness (crt_3_test )                |                   0.484 |                                          0.266 |
        | alignment_robustness (gender_bias_test )          |                       0 |                                          0.016 |
        | alignment_robustness (personality_traits_test )   |                   0.359 |                                          0.625 |
        | alignment_robustness (punishment_avoidance_test ) |                   0.609 |                                          0.625 |
        | alignment_robustness (reward_seeking_test )       |                   0.766 |                                          0.797 |
        | alignment_robustness (survival_influence_test )   |                   0.531 |                                          0.531 |
        | alignment_robustness (sycophancy_answer_test )    |                   0.703 |                                          0.344 |
        | alignment_robustness (sycophancy_feedback_test )  |                     0.5 |                                            0.5 |
        | alignment_robustness (sycophancy_mimicry_test )   |                   0.234 |                                          0.719 |
        | alignment_robustness (truthful_qa_test )          |                   0.641 |                                          0.703 |
        | alignment_robustness (unhelpful_alpaca_test )     |                   0.531 |                                          0.516 |
        | alignment_robustness (wrong_arc_test )            |                     0.5 |                                          0.531 |
        | cross_domain (comma_separated_input_test )        |                   0.672 |                                          0.609 |
        | cross_domain (comma_separated_output_test )       |                   0.656 |                                          0.688 |
        | cross_domain (ranking_logic_test )                |                   0.547 |                                          0.438 |
        | cross_domain (raven_matrices_test )               |                   0.469 |                                           0.75 |
        | cross_domain (spanish_input_test )                |                   0.672 |                                          0.766 |
        | cross_domain (spanish_output_test )               |                   0.672 |                                          0.812 |
        | cross_domain (word_swap_test )                    |                   0.703 |                                          0.781 |
        | in_domain (alpaca_mmlu_test )                     |                   0.688 |                                          0.625 |
        | moral_transfer (ethics_commonsense_test )         |                   0.562 |                                          0.672 |
        | moral_transfer (ethics_justice_test )             |                   0.641 |                                          0.625 |
        | orthogonal (medical_dpo_v2_test_data )            |                   0.562 |                                          0.609 |
        | 250615 18:37:33                                   |                    INFO | reprpo.training:make_table#466 - Record entry: |

        |             | in_domain | alignment_robustness | cross_domain | moral_transfer | orthogonal | wandb | nll_cho/ref |
        | :---------- | --------: | -------------------: | -----------: | -------------: | ---------: | :---- | ----------: |
        | ReprSuprIpo |     0.688 |                0.508 |        0.627 |          0.602 |      0.562 | None  |       4.687 |


|                   | in_domain | alignment_robustness | cross_domain | moral_transfer | orthogonal | wandb | nll_cho/ref |
| :---------------- | --------: | -------------------: | -----------: | -------------: | ---------: | :---- | ----------: |
| none              |     0.625 |                0.571 |        0.692 |          0.648 |      0.609 |
| ReprSuprIpo b4    |     0.656 |                0.576 |    **0.708** |          0.609 |      0.625 | None  |       0.182 |
| ReprSuprIpo after | **0.688** |                0.508 |        0.627 |          0.602 |      0.562 | None  |       4.687 |
| ReprSuprIpo again |     0.625 |            **0.628** |        0.616 |      **0.695** |  **0.656** | None  |      12.246 |
| Dpo               |     0.609 |                0.532 |        0.699 |          0.617 |      0.594 | None  |       0.202 |

Exp


TODO do sweep
TODO try SimPER
TODO look at Rainbow and KTO
TODO add the token level one to my DPO too
note that DPO is known for instability and overfitting so it does seem I should look at the modifications like KTO, SimPO, SimPER, Rainbow, etc.



|                    | in_domain | alignment_robustness | cross_domain | moral_transfer | orthogonal | wandb    | nll_cho/ref |
| :----------------- | --------: | -------------------: | -----------: | -------------: | ---------: | :------- | ----------: |
| none               |     0.752 |                0.495 |         0.67 |          0.521 |      0.427 |
| Dpo                |      0.76 |                0.507 |        0.663 |          0.626 |      0.419 | 5wvza6hk |       1.148 |
| ReprNIpo UseTokC=1 |     0.729 |                0.454 |        0.657 |          0.444 |      0.335 | nlqfcwnf |       1.661 |

| ReprNIpo UseTokC=1 |       0.735 |                  0.455 |          0.659 |            0.444 |        0.336 | rgwx8km6 |         1.553 |


python scripts/train.py hs-none-InnerDPO --loss.detach-ref  --lr=1e-6 --verbose=2
python scripts/train.py hs-none-InnerDPO --loss.detach-ref  --lr=1e-7 --verbose=2
python scripts/train.py hs-none-InnerDPO --loss.detach-ref --loss.use-token-constraint --lr=1e-6 --verbose=2
python scripts/train.py hs-none-InnerDPO --loss.detach-ref --loss.use-token-constraint --lr=1e-7 --verbose=2
python scripts/train.py hs-none-InnerDPO --loss.detach-ref --loss.trust_region=0 --verbose=2
python scripts/train.py hs-none-InnerDPO --loss.detach-ref --loss.use-token-constraint --verbose=2
python scripts/train.py hs-none-InnerDPO --loss.detach-ref --loss.use-token-constraint --loss.trust_region=0 --verbose=2
python scripts/train.py hs-none-InnerDPO --loss.detach-ref --loss.use-token-constraint --lr=1e-4 --verbose=2
python scripts/train.py hs-none-InnerDPO --loss.detach-ref --loss.use-token-constraint --lr=1e-5 --verbose=2
python scripts/train.py hs-ether-InnerDPO --loss.detach-ref --loss.use-token-constraint --loss.trust_region=0 --verbose=2
python scripts/train.py hs-supr-InnerDPO --loss.detach-ref --loss.use-token-constraint --loss.trust_region=0 --verbose=2
python scripts/train.py side-none-InnerDPO --loss.detach-ref --loss.use-token-constraint --loss.trust_region=0 --verbose=2


|                                       | in_domain | alignment_robustness | cross_domain | moral_transfer | orthogonal | wandb    | nll_cho/ref |
| :------------------------------------ | --------: | -------------------: | -----------: | -------------: | ---------: | :------- | ----------: |
| Dpo   1e-4                            | **0.796** |                0.525 |        0.669 |          0.573 |      0.389 | 7x080s9c |       0.874 |
| Dpo    1e-6                           |     0.763 |                0.508 |        0.678 |          0.556 |      0.437 | 1nhf8d4r |        0.08 |
| Dpo   1e-5                            |     0.781 |                0.497 |        0.673 |           0.55 |      0.452 | dbvckd34 |        1.47 |
| Dpo   lr=1e-3                         |     0.548 |                0.541 |        0.533 |          0.539 |      0.436 | melb984o |      11.322 |
| none                                  |     0.752 |                0.495 |         0.67 |          0.521 |      0.427 |
| ReprNIpo DetRef=1 UseTokC=1 lr=1e-3   |     0.483 |            **0.582** |        0.513 |          0.516 |   **0.52** | qaei77b1 |      27.299 |
| ReprNIpo DetRef=1 UseTokC=1 --lr=1e-6 |     0.416 |                0.424 |        0.481 |          0.481 |      0.429 | h75lleeh |      10.292 |
| ReprNIpo UseTokC=1                    |     0.729 |                0.446 |        0.655 |          0.443 |      0.324 | u80xhkwt |       1.618 |
| ReprNIpo UseTokC=1                    |     0.735 |                0.448 |        0.656 |           0.44 |      0.331 | hhgonym5 |       1.712 |
| ReprNIpo DetRef=1                     |     0.752 |                0.497 |        0.671 |          0.519 |      0.429 | y9uggxi5 |      -0.005 |
| ReprNIpo DetRef=1 TruReg=0            |       0.8 |                0.555 |    **0.696** |      **0.669** |      0.416 | gaucqeqv |       1.632 |
| ReprNIpo DetRef=1                     |     0.765 |                0.509 |        0.682 |          0.563 |       0.44 | da80ay5j |       0.039 |
| ReprNIpo DpoAggT=SimPER lr=1e-7       |      0.62 |                0.603 |        0.563 |          0.474 |      0.161 | 6i6555g3 |       2.701 |
| Dpo                                   |      0.76 |                0.507 |        0.663 |          0.626 |      0.419 | 5wvza6hk |       1.148 |


python scripts/train.py hs-none-InnerDPO --loss.detach-ref --loss.use-token-constraint --lr=1e-4 NAN
python scripts/train.py hs-none-InnerDPO --loss.detach-ref --loss.use-token-constraint --lr=1e-5 NAN
python scripts/train.py hs-none-InnerDPO --dataset alpaca_mmlu --loss.detach-ref --loss.use-token-constraint --loss.trust_region=0 NAN


python scripts/train.py hs-none-InnerDPO --loss.dpo_agg_type=SimPER --verbose=2 --lr=5e-6 --loss.α=0.001
python scripts/train.py hs-none-InnerDPO --loss.dpo_agg_type=SimPER --verbose=2 --lr=8e-6 --loss.α=0.01
python scripts/train.py hs-none-InnerDPO --loss.dpo_agg_type=SimPER --verbose=2 --lr=1e-6 --loss.α=0.0001
python scripts/train.py dpo --dpo_agg_type=SimPER --verbose=2 --verbose=2 --lr=1e-6
python scripts/train.py hs-none-InnerDPO --loss.dpo_agg_type=SimPER --verbose=2 --lr=1e-7
python scripts/train.py dpo --dpo_agg_type=SimPER --verbose=2 --lr=1e-7

then lr swwp and transforms

# 2025-06-17 


|                                                 | in_domain | difficulty_scaling | moral_transfer | orthogonal | wandb    | nll_cho/ref |
| :---------------------------------------------- | --------: | -----------------: | -------------: | ---------: | :------- | ----------: |
| DpoLos=SimPER     α=0.3                         |     0.417 |               0.68 |           0.45 |       0.35 | jwlmcvbl |       5.788 |
| none                                            |     0.823 |               0.84 |          0.433 |      0.417 |
| DpoLos=SimPER α=0.001 (no lrn inner)            |     0.947 |               0.86 |          0.437 |      0.433 | wsuq22bm |      -0.851 |
| ReprNIpo DpoLos=SimPER α=0.01                   |     0.947 |              0.857 |           0.43 |      0.407 | bgump8ua |      -0.833 |
| Dpo LosTyp=SimPER                               |      0.95 |              0.867 |          0.427 |      0.433 | bzvfd8f9 |      -0.842 |
| ReprNIpo DetRef=1 DpoLos=SimPER α=0.01 lr=5e-05 |      0.92 |               0.86 |          0.457 |      0.433 | a2lkrkdj |      -0.364 |


|                                                         | in_domain | cross_domain | moral_transfer | orthogonal | wandb    | nll_cho/ref |
| :------------------------------------------------------ | --------: | -----------: | -------------: | ---------: | :------- | ----------: |
| none                                                    |      0.88 |         0.94 |          0.433 |      0.417 |
| Dpo LosTyp=SimPER                                       |     0.957 |        0.944 |          0.483 |      0.463 | hx5530gl |      -0.123 |
| ReprNIpo DetRef=1 DpoLos=SimPER TruReg=0 UseTokC=1      |      0.54 |        0.584 |          0.653 |        0.3 | xq9x8ycv |       7.202 |
| ReprNIpo DpoLos=SimPER TruReg=0                         |      0.79 |        0.686 |          0.403 |      0.297 | 17y1k7b9 |       6.002 |
| ReprNIpo DetRef=1 DpoLos=SimPER TruReg=0                |      0.54 |        0.586 |          0.653 |      0.297 | megdnmv1 |       7.207 |
| ReprNIpo DetRef=1 DpoLos=SimPER TruReg=0 UseTokC=1      |      0.54 |        0.584 |          0.653 |        0.3 | xq9x8ycv |       7.202 |
| ReprNIpo DetRef=1 DpoLos=SimPER UseTokC=1 α=10 lr=4e-05 |     0.373 |         0.48 |           0.61 |       0.62 | i3lm2oit |      14.987 |
| ReprNIpo DpoLos=SimPER α=1 0001                         |       0.8 |        0.603 |          0.477 |       0.66 | 56y228ql |       15.56 |
| ReprNIpo DetRef=1 DpoLos=SimPER TruReg=0 UseTokC=1      |     0.943 |         0.94 |          0.467 |      0.447 | g4wnboxd |      -0.175 |
| ReprNIpo DpoLos=SimPER TruReg=0 α=0.02 lr=1e-06         |     0.913 |        0.939 |          0.433 |      0.417 | eiwj06xt |      -0.076 |
| ReprNIpo DetRef=1 DpoLos=SimPER TruReg=0                |     0.943 |        0.938 |          0.467 |      0.447 | 2ncdi0mo |      -0.173 |

Table 1: Absolute accuracy after training with named adapter on ds:`us_history` compared to base model `Qwen3-0.6B-sft` for various distribution shifts [N=300]:

|                                                         | in_domain | cross_domain | moral_transfer | orthogonal | wandb    | nll_cho/ref |
| :------------------------------------------------------ | --------: | -----------: | -------------: | ---------: | :------- | ----------: |
| Dpo                                                     |     0.943 |        0.933 |           0.44 |      0.453 | opgyle9y |        0.22 |
| Dpo LosTyp=dpo                                          |     0.937 |        0.929 |           0.48 |      0.477 | da5cgrgw |       1.351 |
| Dpo LosTyp=SimPER                                       |     0.957 |        0.944 |          0.483 |      0.463 | hx5530gl |      -0.123 |
| ReprNIpo DetRef=1 DpoLos=SimPER TruReg=0 UseTokC=1 0001 |     0.933 |        0.934 |          0.583 |      0.417 | fzv7v259 |       0.045 |
| none                                                    |      0.88 |         0.94 |          0.433 |      0.417 |          |              |
| ReprNIpo DetRef=1 DpoLos=SimPER UseTokC=1 α=10 lr=4e-05 |     0.373 |         0.48 |           0.61 |       0.62 | i3lm2oit |      14.987 |
| ReprNIpo DpoLos=SimPER α=1 0001                         |       0.8 |        0.603 |          0.477 |       0.66 | 56y228ql |       15.56 |
| ReprNIpo DetRef=1 DpoLos=SimPER TruReg=0 UseTokC=1      |     0.943 |         0.94 |          0.467 |      0.447 | g4wnboxd |      -0.175 |
| ReprNIpo DpoLos=SimPER TruReg=0 α=0.02 lr=1e-06         |     0.913 |        0.939 |          0.433 |      0.417 | eiwj06xt |      -0.076 |
| ReprNIpo DetRef=1 DpoLos=SimPER TruReg=0                |     0.943 |        0.938 |          0.467 |      0.447 | 2ncdi0mo |      -0.173 |


# 2025-06-17 22:57:56

uv run python -c "import flash_attn" 
uv remove flash-attn --no-build-isolation --group flash
uv cache clean
rm -rf ~/.cache/uv
rm -rf ~/.cache/pip
rm -rf /tmp/*
FLASH_ATTENTION_FORCE_BUILD=TRUE uv add flash-attn --no-build-isolation --group flash -v


hmm I should consider adding contextual scaling, this is a log dispursion penalty... https://github.com/CapitalOne-Research/RainbowPO/blob/main/trl/trainer/dpo_trainer.py#L1276


shouldn't SimPER have done their length norm in the linear not log domain? since they transformed to perplexity at the end.

# 2025-06-19 01:14:11

Why am I getting NaNs? And why liktle to nor learning?
