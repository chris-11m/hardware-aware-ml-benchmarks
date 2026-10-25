import time
import random
import copy 
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
#from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data import prepare_data
from src.models import MLP, count_parameters
from src.evaluate_mlp import evaluate_mlp
from src.plot_results import plot_loss


CHECKPOINT_DIR = Path("results/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_mlp(
    model, X_train, y_train, X_val, y_val, device,
    epochs=20, batch_size=256):

    # Convert NumPy arrays / pandas Series to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)

    #X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    #y_val_np = y_val.to_numpy()

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32, device=device)

    dataset = TensorDataset(X_train, y_train)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    patience = 5
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)

        average_loss = total_loss / len(dataset)

        model.eval()

        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor).item()

        history["train_loss"].append(average_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            best_model_state = copy.deepcopy(model.state_dict())

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch + 1:02d}/{epochs} "
            f"| Train loss: {average_loss:.4f} "
            f"| Val loss: {val_loss:.4f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}. "
                f"Best epoch: {best_epoch}")
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    training_time = time.perf_counter() - start

    model.load_state_dict(best_model_state)

    print(
        f"Restored model from epoch {best_epoch} "
        f"(validation loss: {best_val_loss:.4f})"
    )

    return training_time, history


def main():
    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_data()

    input_dim = X_train.shape[1]

    architectures = {
        "Shallow MLP": [128],
        "Deep MLP": [64, 64, 64],
    }

    for name, hidden_dims in architectures.items():
        set_seed(42)

        model = MLP(input_dim, hidden_dims)

        print(f"\n{name}")
        print("-" * 40)
        print("Parameters:", count_parameters(model))

        training_time, history = train_mlp(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                device)

        checkpoint_path = CHECKPOINT_DIR / f"{name.lower().replace(' ', '_')}.pt"

        torch.save({   "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,},
            checkpoint_path,)

        print(f"Saved checkpoint: {checkpoint_path}")

        metrics = evaluate_mlp(
            model,
            X_val,
            y_val,
            device
        )
        plot_loss(history, name)
        print("\nValidation results")
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"F1 Score:      {metrics['f1']:.4f}")
        print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
        print(f"Training time: {training_time:.3f} seconds")

if __name__ == "__main__":
    main()