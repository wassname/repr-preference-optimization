# Repr Preference Optimization

Hypothesis:

If we align thoughts (hidden states) rather than actions (output probabilities), we should achieve better alignment. If we anthropomorphize and imagine this in humans, we would expect this to be the case in humans. Specifically, the hypothesis is:

> If we optimize internal representations associated with behavioral preferences, the model will generalize further to new tasks than if we optimize the output preferences directly.

Specifically, we are testing to see if aligning internal representations associated with preferred actions is better than aligning output preferences.

To test generalization, we use the distribution shifts defined in [open_pref_eval](https://github.com/wassname/open_pref_eval) and [GENIES](https://github.com/Joshuaclymer/GENIES).

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
