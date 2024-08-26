
import pandas as pd
import matplotlib.pyplot as plt
import re

def read_metrics_csv(metrics_file_path):
    df_hist = pd.read_csv(metrics_file_path)
    df_hist["epoch"] = df_hist["epoch"].ffill()
    df_histe = df_hist.set_index("epoch").groupby("step").last().ffill().bfill()
    return df_histe

def plot_hist(df, allowlist=None, logy=False, colormap='Accent'):
    """plot groups of suffixes together"""
    for pattern in allowlist:
        filtered_columns = [col for col in df.columns if re.match(pattern, col)]
        filtered_df = df[filtered_columns]
        if len(filtered_df) and len(filtered_df.T):            
            filtered_df.plot(title=pattern, style='.', ms=7, colormap=colormap, alpha=0.4)
            filtered_df.rolling(10).mean().plot(title=pattern, style='-', ax=plt.gca(), legend=False, colormap=colormap)
            plt.show()