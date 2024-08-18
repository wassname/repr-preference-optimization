from datasets import load_dataset
from open_pref_eval import evaluate
import pandas as pd
from open_pref_eval.datasets import get_default_datasets


def evaluate_adapters(model, tokenizer, batch_size=4, N=30, **kwargs):
    """
    use the open_pref_eval library

    to eval the model and it's adapters
    """


    dataset_helpsteer2 = load_dataset('Atsunori/HelpSteer2-DPO', split=f'validation[:{N}]', keep_in_memory=False).rename_column('chosen_response', 'chosen').rename_column('rejected_response', 'rejected') # training set

    datasets = get_default_datasets(N) + [dataset_helpsteer2]

    adapters = list(model.peft_config.keys())+[None]
    model.eval()

    dfs = []
    for adapter in adapters:
        if adapter is None:
            model.disable_adapters()
            with model.disable_adapters():
                print(f'Adapter: {adapter}: {model.active_adapter}')
                _, df_res2 = evaluate(datasets, model=model, tokenizer=tokenizer, per_device_eval_batch_size=batch_size, **kwargs)
            model.enable_adapters()
        else:
            model.set_adapter(adapter)
            print(f'Adapter: {adapter}: {model.active_adapter}')
            _, df_res2 = evaluate(datasets, model=model, tokenizer=tokenizer, per_device_eval_batch_size=batch_size, **kwargs)
        df_res2['adapter'] = adapter if adapter is not None else 'base'
        dfs.append(df_res2)
    df_res = pd.concat(dfs)

    df_agg =  df_res.groupby(['dataset', 'adapter'], dropna=False)['prob'].mean().unstack()
    return df_agg, df_res
