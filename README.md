# Representation Preference Optimization (ReprPO)

> **Hypothesis**: If instead of optimising behavior we optimise internal representations associated with preferences, the model will generalize further to new tasks

If you allow some anthropomorphism, aligning internal representations is a bit like aligning someone's thoughts, values, habits, or concepts. Changing someone's values is expected to be more robust than changing their behavior in one context. This is supported by recent work such as https://github.com/blackswan-ai/short-circuiting/issues

Status: WIP

## Plan

- [x] Get it running
- [x] add bnb and lora to speed it up
- [x] Switch to circuit breaking losses
- [x] see if we can get coherent output
- [ ] measure generalization of baseline vs ReprPO

```sh
poetry install

python -u train.py model=pythia69 datasets=[hh] loss=sft exp_name=anthropic_dpo_pythia69 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false
```
