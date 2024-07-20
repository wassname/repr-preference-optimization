from matplotlib import pyplot as plt
from trl import DPOTrainer
import pandas as pd
import numpy as np

def extract_hist(trainer: DPOTrainer):
    df_hist1 = pd.DataFrame(trainer.state.log_history)
    df_hist1 = df_hist1.groupby('step').mean() # mean over gradient accum?
    df_hist2 = df_hist1.dropna(axis=1, thresh=int(len(df_hist1)*0.9))

    return df_hist1, df_hist2


def plot_hist(trainer: DPOTrainer):

    current_args = trainer.args.to_dict()
    default_args = type(trainer.args)('').to_dict()
    blocklist = ['output_dir', 'per_device_eval_batch_size', 'run_name' 'remove_unused_columns']
    args_diff = {k: v for k, v in current_args.items() if v != default_args[k]}
    args_diff = {k: v for k, v in args_diff.items() if k not in blocklist}


    plt.style.use('ggplot')
    df_hist1, df_hist2 = extract_hist(trainer)
    df_hist1 = pd.DataFrame(trainer.state.log_history)
    df_hist1 = df_hist1.groupby('step').mean() # mean over gradient accum?
    df_hist2 = df_hist1.dropna(axis=1, thresh=int(len(df_hist1)*0.9))

    N = len(df_hist2.columns)
    fig, axes = plt.subplots(int(np.ceil(N/2)), 2, figsize=(12, N), sharex=True)
    axes = axes.flatten()
    for i, c in enumerate(df_hist2.columns):
        x = df_hist2[c].dropna()
        x.plot(ax=axes[i], style='.', ms=7, title=c)
        x.ewm(span=30).mean().plot(ax=axes[i])
    plt.tight_layout()

    return df_hist1, args_diff


def plot_paired_hist(trainer: DPOTrainer):
    fontsize= 10
    plt_kwargs = dict(figsize=(6,1.5), style='.', ms=7)
    df_hist1, df_hist2 = extract_hist(trainer)

    # any ending in /loss and not starting in component
    x = df_hist1[[c for c in df_hist1.columns if c.endswith('/loss') and not c.startswith('component')]].dropna()
    if len(x.columns) > 0:
        x.plot(**plt_kwargs)
        plt.legend(fontsize=fontsize-2)
        plt.title('raw losses before weighting\nuse to work out alpha and check loss is going down in absence of confounders', fontdict={'fontsize': fontsize})
        plt.show()

    # plot any columns starting with component
    x = df_hist1[[c for c in df_hist1.columns if c.startswith('component')]].dropna()
    if len(x.columns) > 0:
        x.plot(**plt_kwargs)
        plt.title('loss components after weighting, before comb\nUse see how each contributes to loss, debug alpha and coeffecient', fontdict={'fontsize': fontsize})
        plt.legend(fontsize=fontsize-2)
        plt.show()

    # any including cosine
    x = df_hist1[[c for c in df_hist1.columns if 'cosine' in c]].dropna()
    if len(x.columns) > 0:
        x.plot(**plt_kwargs)
        plt.title('cosine similarity\nboth should stay high, with some tradeoff\notherwise incoherent/degenerate state', fontdict={'fontsize': fontsize})
        plt.legend(fontsize=fontsize-2)
        plt.show()


