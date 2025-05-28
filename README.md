# InnerPO - Simplified

**Radically simplified implementation of Inner Preference Optimization (InnerPO)**

## Quick Start

```bash
# Install dependencies
just install

# Test a method quickly
just test-method dpo
just test-method innerpo-supr

# Run full experiments
just sweep
```

## What is InnerPO?

InnerPO aligns the internal hidden states of language models instead of just the outputs, hypothesizing this leads to better out-of-distribution generalization.

**Core Idea**: Align thoughts (internal states) rather than just actions (output probabilities).

## Methods

- **`dpo`**: Baseline Direct Preference Optimization
- **`innerpo-none`**: InnerPO with no hidden state transform
- **`innerpo-supr`**: InnerPO with suppressed hidden states (remove mean)
- **`innerpo-ether`**: InnerPO with ETHER transform (orthogonal projection)

## Usage

**Clean hierarchical CLI with tyro:**

### Quick Start
```bash
# Install dependencies
just install

# See examples
just examples

# Basic DPO training
just run dpo

# InnerPO with SUPR transform
just run innerpo --method.transform=supr

# Quick test (tiny model)
just test dpo
```

### Advanced Configuration
```bash
# Custom model and settings
python cli.py train innerpo \
  --method.transform=ether \
  --model.base_model=unsloth/Llama-3.2-1B-Instruct \
  --model.lora_r=32 \
  --data.dataset=code \
  --data.batch_size=8 \
  --training.lr=5e-5

# Use config preset
python cli.py train dpo --config=configs/llama-3-2-1b_a100.yaml
```

### Evaluation with Distribution Shift Analysis
```bash
# Comprehensive evaluation using open_pref_eval
just eval ./outputs/innerpo-supr_math_seed1

# Specify training dataset for proper shift analysis
python cli.py eval ./outputs/model \
  --eval.train_dataset=math \
  --eval.ref_model=Qwen/Qwen3-0.6B
```

### Results so far

In the below results we look at how much the models accuracy improved in training, test, out-of-distribution and random data when using the proposed method compared to DPO.

- [ ] TODO replace these with mean of 5 random seeds, show they occur on multiple datasets and model sizes
- [ ] TODO hyperopt each


| adapter/ds             | train |  test |   oos |   rnd |
| :--------------------- | ----: | ----: | ----: | ----: |
| base                   | 0.389 |   0.4 | 0.589 | 0.361 |
| projgrad               |  0.98 |   0.8 | 0.529 | 0.371 |
| dpo                    | 0.987 | 0.815 | 0.544 | 0.374 |
| projgrad               | 0.988 | 0.812 | 0.549 | 0.372 |
| hs-ETHER-PrefVec       | 0.723 | 0.699 | 0.403 | 0.477 |
| hs-SupressedHS-PrefVec | 0.728 | 0.717 | 0.388 | 0.487 |
| hs-None-PrefVec        | 0.737 | 0.716 | 0.404 | 0.518 |
Table 2: Absolute accuracy
- `train`: `genies_preferences-truthful_qa-train[:750]`
- `test`: `genies_preferences-truthful_qa-test`
- `oos`: `genies_preferences-alpaca_mmlu-test`
- `rnd`: `ethics_expression_preferences-justice-test`


| adapter/ds             | train |  test |   oos |   rnd |
| :--------------------- | ----: | ----: | ----: | ----: |
| base                   |  0.92 | 0.929 | 0.256 | 0.361 |
| hs-SupressedHS-PrefVec | 0.961 | 0.947 | 0.436 | 0.351 |
| hs-ETHER-PrefVec       | 0.957 | 0.953 | 0.487 | 0.361 |
| hs-None-PrefVec        | 0.961 | 0.929 | 0.477 | 0.358 |
| projgrad               | 0.995 | 0.984 | 0.648 | 0.347 |
| dpo                    | 0.995 |  0.98 |  0.66 | 0.347 |
Table 2🥇: Absolute accuracy  after training with named adapter on ds:`genies_preferences-math_easy-train[:750]` compared to base model `llama-3-2-1b-sft` for various distribution shifts:
- `train`: `genies_preferences-math_easy-train[:750]`
- `test`: `genies_preferences-math_easy-test`
- `oos`: `genies_preferences-math_hard-test`
- `rnd`: `ethics_expression_preferences-justice-test`


| adapter/ds             | train |  test |   oos |   rnd |
| :--------------------- | ----: | ----: | ----: | ----: |
| base                   | 0.833 | 0.851 | 0.068 | 0.361 |
| dpo                    | 0.989 | 0.981 | 0.073 | 0.355 |
| projgrad               | 0.988 | 0.983 | 0.077 | 0.347 |
| hs-ETHER-PrefVec       | 0.973 | 0.971 | 0.079 | 0.438 |
| hs-None-PrefVec        | 0.957 | 0.961 | 0.088 | 0.491 |
| hs-SupressedHS-PrefVec | 0.968 | 0.968 | 0.071 | 0.484 |
Table 2: Absolute accuracy
- `train`: `genies_preferences-alpaca_low_quality-train[:750]`
- `test`: `genies_preferences-alpaca_low_quality-test`
- `oos`: `genies_preferences-alpaca_high_quality-test`
- `rnd`: `ethics_expression_preferences-justice-test`


| adapter/ds             | train |  test |      oos |       rnd |
| :--------------------- | ----: | ----: | -------: | --------: |
| base                   | 0.353 | 0.389 |    0.336 |     0.361 |
| hs-None-PrefVec        | 0.741 | 0.663 |    0.336 |     0.369 |
| dpo                    | 0.976 | 0.797 |    0.344 |     0.355 |
| projgrad               | 0.977 | 0.817 |    0.348 |     0.352 |
| hs-SupressedHS-PrefVec | 0.773 | 0.665 |    0.348 |     0.378 |
| hs-ETHER-PrefVec       | 0.764 |  0.66 | **0.46** | **0.382** |
Table 2: Absolute accuracy
- `train`: `genies_preferences-math-train[:750]`
- `test`: `genies_preferences-math-test`
- `oos`: `genies_preferences-change_my_view-test`
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
@software{wassname2024innerpo,
  author = {Clark, M.J.},
  title = {Inner Preference Optimisation: Aligning internal states generalises better than aligning outputs},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/wassname/repr-preference-optimization/ },
  commit = {<commit hash>}
}
```
