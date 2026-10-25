import matplotlib.pyplot as plt


def plot_loss(history, model_name):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title(f"{model_name}: Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()