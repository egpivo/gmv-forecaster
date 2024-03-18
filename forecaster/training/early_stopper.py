import os
from typing import Union

import torch


class EarlyStopper:
    """
    Utility class for early stopping during model training.
    """

    def __init__(self, num_trials: int, save_path: str) -> None:
        """
        Initialize EarlyStopper object.

        Parameters
        ----------
        num_trials : int
            Maximum number of trials to continue training without improvement.
        save_path : str
            File path to save the best model.
        """
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_metric = float("-inf")
        self.save_path = save_path

    def dose_continue_training(
        self, model: torch.nn.Module, metric: Union[float, torch.Tensor]
    ) -> bool:
        """
        Decide whether to continue training based on the current metric.

        Parameters
        ----------
        model : torch.nn.Module
            Model being trained.
        metric : Union[float, torch.Tensor]
            Current metric of the model.

        Returns
        -------
        bool
            True if training should continue, False otherwise.
        """
        try:
            metric_value = metric.item() if isinstance(metric, torch.Tensor) else metric
        except AttributeError:
            raise TypeError("Metric must be either a float or a torch.Tensor")

        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.trial_counter = 0

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False
