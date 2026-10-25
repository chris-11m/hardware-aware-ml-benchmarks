import time
import statistics
from pathlib import Path

import torch

from src.models import MLP


CHECKPOINT_DIR = Path("results/checkpoints")

BATCH_SIZES = [1, 16, 64, 256, 1024]
WARMUP_RUNS = 20
TIMED_RUNS = 100


def load_model(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = MLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def benchmark_model(model, input_dim, batch_size, device):

    # Synthetic input, already on the target device.
    X = torch.randn(
        batch_size,
        input_dim,
        device=device,
    )

    with torch.inference_mode():

        # Warm up GPU kernels and runtime.
        for _ in range(WARMUP_RUNS):
            model(X)

        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []

        for _ in range(TIMED_RUNS):

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            model(X)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            latencies.append(elapsed_ms)

    median_latency_ms = statistics.median(latencies)

    throughput = (
        batch_size / (median_latency_ms / 1000)
    )

    return median_latency_ms, throughput


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    checkpoints = {
        "Shallow MLP": CHECKPOINT_DIR / "shallow_mlp.pt",
        "Deep MLP": CHECKPOINT_DIR / "deep_mlp.pt",
    }

    for name, path in checkpoints.items():

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        input_dim = checkpoint["input_dim"]

        model = load_model(path, device)

        print(f"\n{name}")

        for batch_size in BATCH_SIZES:

            latency, throughput = benchmark_model(
                model,
                input_dim,
                batch_size,
                device,
            )

            print(
                f"Batch {batch_size:4d} | "
                f"Latency: {latency:.4f} ms | "
                f"Throughput: {throughput:.0f} samples/s"
            )


if __name__ == "__main__":
    main()