import os

import torch


class EarlyStopper:
    """
    Utility class for early stopping during model training.
    """

    def __init__(self, num_trials: int, save_path: str):
        """
        Initialize EarlyStopper object.

        Args:
            num_trials (int): Maximum number of trials to continue training without improvement.
            save_path (str): File path to save the best model.
        """
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def continue_training(self, model: torch.nn.Module, accuracy: float) -> bool:
        """
        Decide whether to continue training based on the current accuracy.

        Args:
            model: Model being trained.
            accuracy (float): Current accuracy of the model.

        Returns:
            bool: True if training should continue, False otherwise.
        """
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
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
