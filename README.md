# Repr Preference Optimization

A fork of [DPO](https://github.com/eric-mitchell/direct-preference-optimization) to test the hypothesis:

> If we optimize internal representations associated with behavioral preferences, the model with generalize further to new tasks

If we align thoughts (hidden states) rather than actions (output probs) we should get better alignment? If we anthropomorphize, we would expectg this to be the case in humans. This is the hypothesis of this repo. 

Specifically we test to see if aligning internal representations associated with prefered actions (ours) is better than aligning output preference (DPO).

We test generalization we use the distribution shifts defined in GENIES.

Status: WIP

## Plan

- [x] Get it running
- [x] Switch to circuit breaking losses
- [x] see if we can get coherent output
- [x] measure generalization of baseline vs ReprPO

```sh
poetry install

python -u nbs.train.py
```


# Citing 
If this repository is useful in your own research, you can use the following BibTeX entry:

@software{wassname2024reprpo,
  author = {Clark, M.J.},
  title = {Representation Preference Optimisation: Aligning internal states generalises better than aligning outputs},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/wassname/repr-preference-optimization/ },
  commit = {<commit hash>}
}
