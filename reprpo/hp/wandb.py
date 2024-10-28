
"""
modified from optuna_integratoion.wandb.wandb.py

"""
import optuna
import wandb
from typing import Any, Sequence
from optuna.integration.wandb import WeightsAndBiasesCallback

class WeightsAndBiasesCallback2(WeightsAndBiasesCallback):

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Failed and pruned trials have `None` as values.
        metrics = {}
        values: list = trial.values

        if values is not None:
            if isinstance(self._metric_name, str):
                if len(values) > 1:
                    # Broadcast default name for multi-objective optimization.
                    names = ["{}_{}".format(self._metric_name, i) for i in range(len(values))]

                else:
                    names = [self._metric_name]

            else:
                if len(self._metric_name) != len(values):
                    raise ValueError(
                        "Running multi-objective optimization "
                        "with {} objective values, but {} names specified. "
                        "Match objective values and names, or use default broadcasting.".format(
                            len(values), len(self._metric_name)
                        )
                    )

                else:
                    names = [*self._metric_name]

            metrics = {name: value for name, value in zip(names, values)}

        if self._as_multirun:
            metrics["trial_number"] = trial.number

        attributes = {"direction": [d.name for d in study.directions]}

        step = trial.number if wandb.run else None
        run = wandb.run

        # Might create extra runs if a user logs in wandb but doesn't use the decorator.

        if not run:
            run = self._initialize_run()
            run.name = f"trial/{trial.number}/{run.name}"

        run.log({**trial.params, **metrics}, step=step)

        if self._as_multirun:
            run.config.update({**attributes, **trial.params}, allow_val_change=True) # changed
            run.tags = tuple(self._wandb_kwargs.get("tags", ())) + (study.study_name,)
            run.finish()
        else:
            run.config.update(attributes)
