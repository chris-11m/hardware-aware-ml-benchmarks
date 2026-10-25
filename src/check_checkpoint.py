import torch

from src.models import MLP
from src.data import prepare_data
from src.evaluate_mlp import evaluate_mlp


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        "results/checkpoints/shallow_mlp.pt",
        map_location=device,
        weights_only=True,
    )

    model = MLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded model:")
    print(model)

    _, X_val, _, _, y_val, _, _ = prepare_data()

    metrics = evaluate_mlp(
        model,
        X_val,
        y_val,
        device,
    )

    print("Validation metrics:", metrics)


if __name__ == "__main__":
    main()