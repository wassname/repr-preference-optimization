from datasets import load_dataset
from open_pref_eval.evaluation import evaluate_model
from open_pref_eval.datasets import get_default_datasets


def evaluate_adapters(model, tokenizer, N=30, **kwargs):
    """
    use the open_pref_eval library

    to eval the model and it's adapters
    """

    dataset_helpsteer2 = (
        load_dataset(
            "Atsunori/HelpSteer2-DPO", split=f"validation[:{N}]", keep_in_memory=False
        )
        .rename_column("chosen_response", "chosen")
        .rename_column("rejected_response", "rejected")
    )  # training set

    datasets = get_default_datasets(N) + [dataset_helpsteer2]
    model.eval()

    return evaluate_model(datasets=datasets, model=model, tokenizer=tokenizer, **kwargs)
