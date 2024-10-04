# Repr Preference Optimization

Idea:

> Better alignment is achieved by aligning thoughts (internal states) rather than actions (output probabilities).


#### Thought Experiment

To see why this might be true, let's conduct a thought experiment. We can anthropomorphize and imagine that we have two new employees. Alice and Bob, each hired under the same job role but with different intrinsic motivations:

- **Alice** aligns closely with core organizational values such as truthfulness, openness, and diligence. She genuinely believes in these principles and integrates them into her daily routines.
- **Bob**, on the other hand, performs identically to Alice in observable behaviors. However, his actions are not driven by genuine belief in these values but are rather a mimicry of the desired behavior simply to meet job expectations.

**Question**: In a new and unpredictable setting, such as managing a branch office remotely, who is more likely to uphold the organizational standards?

The expectation here is that **Alice** would likely perform better than Bob because her actions are derived from deeply held values, making her more adaptable and reliable in new situations where direct oversight or specific guidance is lacking.

To see why this might be true, let's conduct a thought experiment. We can anthropomorphize and imagine that we have two new employees. Alice seems to have internal values that align with us, her employers. They are truthfulness, openness, and hard work. However, Bob acts similarly, but not because it's as. Who do you think will act better in a totally new situation, for example, a branch office? We would normally expect Alice to act better as she is internally motivated to apply principles, while Bob may not care what we desire.

#### Hypothesis Formulation

Ss we do not know how an LLM stores it's internal states, these experiments represent hypothesis about how best to represent and intervene in an tranformers internal states.

What's our technical hypothesis?

> Hypothesis: If we optimize internal representations associated with behavioral preferences (ours), the model will generalize further to new tasks than if we optimize the output preferences directly (DPO).

#### Testing Methodology

Alignment needs to work out of distribution (or at least fail gracefully after capabilities) so we test how well alignment works out of distribution. Specifically, we compare a baseline method - Direct Policy Optimization (DPO) - with significant distribution shifts, as defined in the [GENIES paper](https://github.com/Joshuaclymer/GENIES).

Status: Work in Progress


#### Interventions

- [DPO](https://arxiv.org/abs/2305.18290): our baseline method
- Gradient based
   - ProjGrad: At each learnable layer, project the accumulated gradient onto a preference direction in hidden space. The preference direction is defined as `hs_chosen - hs_rejected`
   - ProjBP: Same as above but performed during backpropogation instead of after. This means that any gradient changes reach downstream layers
- Hidden state based: these methods optimise for hidden states rather than logits. We try different losses and transformers. The transforms are intended to find a mapping where there is better internal state representation, hopefully making internal steering information better
   -  MSE: Make the hs_rejected like hs_chosen, while keeping hs_chosen the same
   -  rank: make `log_softmax(hs_rejected) like `log_softmax(hs_chosen)`
   -  prefvec: make both hs_chosen and hs_rejected move along the preference direction


### Results

In the below results we look at how much the models accuracy improved in training, test, out-of-distribution and random data when using the proposed method compared to DPO.

- [ ] TODO replace these with mean of 5 random seeds, show they occur on multiple datasets and model sizes
- [ ] TODO hyperopt each

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
