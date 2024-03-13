import os

import torch
from torchmetrics.retrieval import RetrievalAUROC, RetrievalRecall
from tqdm import tqdm


class EarlyStopper:
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy) -> bool:
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


def train_model(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test_model(model, data_loader, device, top_k=10):
    model.eval()
    targets, predicts = list(), list()

    with torch.no_grad():
        for fields, target in tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    auc_at_k = RetrievalAUROC(top_k=k)(targets, predicts)
    recall_at_k = RetrievalRecall(top_k=k)(targets, predicts)

    return auc_at_k, recall_at_k
