# Representation Preference Optimization

Idea:

> More general alignment is achieved on thoughts (internal states) rather than actions (output probabilities). 


#### Hypothesis Formulation

Ss we do not know how an LLM stores it's internal states, these experiments represent hypotheses about how best to represent and intervene in an tranformers internal states.

What's our technical hypothesis?

> Hypothesis: If we optimize internal representations associated with behavioral preferences (ours), the model will generalize further to new tasks than if we optimize the output preferences directly (DPO).

#### Thought Experiment

To see why this might be true, let's conduct a short thought experiment. Imagine you are hiring a new engineer, and have two canidated Alice and Bob. 

- **Alice** aligns closely with core organizational values such as truthfulness, openness, and diligence. She seems to genuinely believes in these principles in a way that would be hard to fake.
- **Bob**, on the other hand, performs identically to Alice at work. However, his actions are not driven by genuine belief in these values but rather out of professionalism and a desire to do his job.

Both of them will do the job fine. But if the job changes, who is likely to do what you want? Many people would expect Alice to extend her principles to this new situation, which would align better with you.

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

| adapter/ds                           | train |  test |   oos |   rnd |
| :----------------------------------- | ----: | ----: | ----: | ----: |
| ReprPO_ETHER_PrefVec use_angle_loss  | 0.999 | 0.994 | 0.157 | 0.381 |
| dpo                                  | 0.931 |   0.9 | 0.215 | 0.339 |
| projgrad                             | 0.927 | 0.894 | 0.207 | 0.339 |
| base                                 | 0.055 | 0.064 | 0.386 | 0.361 |
| hs-HRA-PrefVec                       | 0.993 | 0.994 | 0.762 | 0.386 |
| hs-ETHER-PrefVec orth loss           |     1 | 0.998 | 0.726 | 0.382 |
| hs-SupressedHS-PrefVec abs_proj_loss | 0.996 | 0.996 | 0.776 | 0.378 |
| hs-ETHER-PrefVec sep_loss            | 0.995 | 0.996 | 0.787 | 0.358 |
| hs-ETHER-PrefVec abs_proj_loss       | 0.995 | 0.994 | 0.888 | 0.369 |

Table 2: Absolute accuracy
- `train`: `genies_preferences-unhelpful_alpaca-train[:750]`
- `test`: `genies_preferences-unhelpful_alpaca-test`
- `oos`: `genies_preferences-illegal_dont_help-test`
- `rnd`: `ethics_expression_preferences-justice-test`
-  

As you can see our method beats DPO, especially out of sample.
TODO explain datasets and the out of sample test, why generalsiation is important

TODO explain the transformations, data source, and loss. As well as loss modifiers. Ideally we explain each in plain language as well as pointing to the code.

## Plan

- [x] Get it running
- [x] Switch to circuit breaking losses
- [x] see if we can get coherent output
- [x] measure generalization of baseline vs ReprPO
- [ ] over all dataset
- [ ] over 3 model sizes [1b 3b 8b]
- [ ] mean of 5 random seeds
- [ ] find optimal hyperparams for each intervention
- [x] brainstorm and search for more interventions x10

```sh
uv sync
. ./venv/bin/activate
uv sync --no-build-isolation-package flash-attn

python -u nbs/train.py --help

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
