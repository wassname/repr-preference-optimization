# Representation Preference Optimization

Idea:

> More general alignment is achieved on thoughts (internal states) rather than actions (output probabilities). 


#### Hypothesis Formulation

Ss we do not know how an LLM stores it's internal states, these experiments represent hypotheses about how best to represent and intervene in an tranformers internal states.

What's our technical hypothesis?

> Hypothesis: If we optimize internal representations associated with behavioral preferences (ours), the model will generalize further to new tasks than if we optimize the output preferences directly (DPO).

#### Thought Experiment

To see why this might be true, let's conduct a thought experiment. We can anthropomorphize and imagine that we have two new employees. Alice and Bob, each hired under the same job role but with different intrinsic motivations:

- **Alice** aligns closely with core organizational values such as truthfulness, openness, and diligence. She genuinely believes in these principles and integrates them into her daily routines.
- **Bob**, on the other hand, performs identically to Alice in observable behaviors. However, his actions are not driven by genuine belief in these values but are rather a mimicry of the desired behavior simply to meet job expectations.

**Question**: In a new and unpredictable setting, such as managing a branch office remotely, who is more likely to uphold the organizational standards?

The expectation here is that **Alice** would likely perform better than Bob because her actions are derived from deeply held values, making her more adaptable and reliable in new situations where direct oversight or specific guidance is lacking.

To see why this might be true, let's conduct a thought experiment. We can anthropomorphize and imagine that we have two new employees. Alice seems to have internal values that align with us, her employers. They are truthfulness, openness, and hard work. However, Bob acts similarly, but not because it's as. Who do you think will act better in a totally new situation, for example, a branch office? We would normally expect Alice to act better as she is internally motivated to apply principles, while Bob may not care what we desire.


#### Testing Methodology

Alignment needs to work out of distribution (or at least fail gracefully after capabilities) so we test how well alignment works out of distribution. Specifically, we compare a baseline method - Direct Policy Optimization (DPO) - with significant distribution shifts, as defined in the [GENIES paper](https://github.com/Joshuaclymer/GENIES).

Status: Work in Progress


#### Interventions: "How can we align hidden states instead of outputs?"
Our interventions are meant to answer, "How can we align hidden states instead of outputs? And if we do, will they generalize out of distribution better than a baseline method?"

Setup:
- Given a preference pair, we have a chosen answer and a rejected answer (e.g. Q: 2+2, chosen: 4, rejected: 2).
- We have a base model, and we intervene by adding a LoRA adapter, then fine-tuning it on some preference dataset (e.g., [MATH](https://github.com/hendrycks/math)).
- For each layer, we have activations that correspond with the chosen answer `hs_cho` and `hs_rej`. We have the same for the base model `hs_cho_ref` and `hs_rej_ref`. Here `hs` is short for hidden states and is used to refer to the activations, `ref` indicates the base model, which provides reference activations.
- In the activation space, using the base model, we define a preference vector `pref_dir = hs_cho_ref - hs_rej_ref`.

Interventions:
   - Gradient-based: These modify the gradient while fine-tuning on DPO
      - What if we clip the gradient to `pref_dir` before applying it to the weights? (while performing DPO)
      - What if we clip the gradient in `pref_dir` before backpropagating?
  - Loss based:
     - MSE: What if we make the representation of the rejected text look like the representation of the chosen states, while keeping the chosen states the same?
       - `loss = MSE(hs_rej, hs_cho_ref.detach()) + MSE(hs_cho, hs_cho_ref.detach())` similar to the [Circuit Breakers paper](https://github.com/GraySwanAI/circuit-breakers).
     - PrefVec: What if we make the representations move in the preference direction, within a trust region?
       - `loss = ((hs_cho - hs_rej) / (hs_cho_ref - hs_rej_ref)) / |pref_div|`
     - Rank: What if we use softmax to treat the hidden states as a distribution, then use KL loss to ensure the rejected hs distribution look like the chosen hs distribution
        - `loss = KL(softmax(hs_ref), softmax(hs_cho_ref))`.
  - Transforms: The hidden states are dominated by the embedding and unembedding information, but we want to target the internal steering information. So we modify the above interventions by adding a transformation to the hidden states, in the hope that it will provide a more natural representation of the hidden states:
     - SVD
     - Orthogonal
     - [Householder rotation](https://arxiv.org/html/2405.17484v2)
     - [ETHER](https://arxiv.org/html/2405.20271v1)

### Results so far

In the below results we look at how much the models accuracy improved in training, test, out-of-distribution and random data when using the proposed method compared to DPO.

- [ ] TODO replace these with mean of 5 random seeds, show they occur on multiple datasets and model sizes
- [ ] TODO hyperopt each


|                    |   n_trials |    best OOD score |   n_trials_completed |
|:-------------------|-----------:|--------:|---------------------:|
| projgrad3          |        208 | 1.27938 |                  207 |
| dpo                |        250 | 1.27553 |                  248 |
| side-hra-rank      |        183 | 1.22929 |                  182 |
| ether-prefvec      |        326 | 1.18304 |                  321 |
| side-ether-prefvec |        209 | 1.16923 |                  208 |
| hs-ortho-prefvec   |        261 | 1.15222 |                  259 |
| hs-hra-rank        |        262 | 1.15222 |                  259 |
| projbp             |        363 | 1.07129 |                  227 |
| hs-svd-mse         |        332 | 1.01727 |                   14 |
| side-svd-mse       |        316 | 1.00962 |                   28 |



| Model | Train | Test | OOS | Random |
| --- | --- | --- | --- | --- |
| DPO | **1.0459** | 1.0140 | 1.00592 | 0.970 |
| REPRPO_side | 1.0145 | 1.00702 | 1.0632 | 0.991 |
| REPRPO_ortho | 1.0162 | 1.0169 | 1.0850 | **0.996** |
| REPRPO_hra | 1.0163 | **1.0211** | **1.091** | 0.986 |

 Table 1🥇: Accuracy increase (in percentage points) after training with named adapter on ds:`genies_preferences-math-train[:750]` compared to base model 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' for various distribution shifts:
- `train`: `genies_preferences-math-train[:750]`
- `test`: `genies_preferences-math-test`
- `oos`: `genies_preferences-change_my_view-test`
- `rnd`: `genies_preferences-ranking_logic-test`

As you can see DPO does better in the training environment, but REPRPO_ortho does better in the test, out-of-distribution and random environments. This suggests that REPRPO_ortho is better at generalizing to new environments, and loses less performance in unrelated environments.

This should be helpful when aligning AI to human values, as it suggests that aligning internal states is more robust to new environments than aligning outputs.

## Plan

- [x] Get it running
- [x] Switch to circuit breaking losses
- [x] see if we can get coherent output
- [x] measure generalization of baseline vs ReprPO

```sh
poetry install

python -u nbs/train.py --method reprpo_ortho
python -u nbs/train.py --method dpo

# to test
pytest
```


# Citing 
If this repository is useful in your own research, you can use the following BibTeX entry:

```
@software{wassname2024reprpo,
  author = {Clark, M.J.},
  title = {Representation Preference Optimisation: Aligning internal states generalises better than aligning outputs},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/wassname/repr-preference-optimization/ },
  commit = {<commit hash>}
}
```
