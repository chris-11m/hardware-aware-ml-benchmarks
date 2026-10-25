import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_mlp(model, X_val, y_val, device):
    model.eval()

    X_tensor = torch.as_tensor(
        X_val,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (probabilities >= 0.5).astype(int)

    y_true = np.asarray(y_val)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "f1": f1_score(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }