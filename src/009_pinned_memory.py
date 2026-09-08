"""A demo of pinned memory and a GPU-resident dataset.

Training on a GPU has two costs: the maths, and moving the data to the card.
For a small model the move is often the slower part. This demo measures three
ways to feed the GPU, from slowest to fastest.

1. A plain DataLoader. Each batch is copied from ordinary CPU memory.
2. A DataLoader with pin_memory=True. Pinned memory is page-locked, so the
   card can copy it by direct memory access, and the copy can overlap work.
3. The whole dataset held on the GPU. Nothing is copied during training.

FashionMNIST is small: 60000 images of 28x28 floats is about 188 MB. A
dataset that fits on the card should live on the card.
"""

import time
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, ReLU, Linear, Module, Flatten, Sequential
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

BATCH_SIZE: int = 64
TRANSFER_MB: int = 256
TRANSFER_REPEATS: int = 20


def heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


class NeuralNetwork(Module):

    def __init__(self) -> None:
        super().__init__()
        self.flatten: Flatten = Flatten()
        self.layers: Sequential = Sequential(
            Linear(28 * 28, 512),
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(self.flatten(x))


def transfer_bandwidth(device: str) -> None:
    """Measure the raw copy speed of ordinary memory against pinned memory."""

    heading("Raw transfer speed: pageable memory against pinned memory")

    count: int = TRANSFER_MB * 1024 * 1024 // 4  # float32 is 4 bytes
    pageable: Tensor = torch.randn(count)
    pinned: Tensor = pageable.pin_memory()

    for name, source, non_blocking in (
            ("pageable", pageable, False),
            ("pinned  ", pinned, True)):

        # Warm up, so the first slow copy does not skew the result.
        source.to(device, non_blocking=non_blocking)
        torch.cuda.synchronize()

        start: float = time.perf_counter()
        for _ in range(TRANSFER_REPEATS):
            source.to(device, non_blocking=non_blocking)
        torch.cuda.synchronize()
        elapsed: float = time.perf_counter() - start

        gb: float = TRANSFER_MB * TRANSFER_REPEATS / 1024.0
        print(f"{name}: {elapsed:6.3f} s for {gb:.1f} GB -> {gb / elapsed:6.2f} GB/s")


def time_dataloader_epoch(
        data_loader: DataLoader,
        model: NeuralNetwork,
        loss_fn: CrossEntropyLoss,
        optimizer: Optimizer,
        device: str,
        non_blocking: bool) -> float:
    """Train for one pass over the data. Return the seconds it took."""

    model.train()
    torch.cuda.synchronize()
    start: float = time.perf_counter()

    for X, y in data_loader:
        X = X.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        loss: Tensor = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    return time.perf_counter() - start


def time_gpu_resident_epoch(
        images: Tensor,
        labels: Tensor,
        model: NeuralNetwork,
        loss_fn: CrossEntropyLoss,
        optimizer: Optimizer) -> float:
    """Train for one pass, with every sample already on the GPU."""

    model.train()
    torch.cuda.synchronize()
    start: float = time.perf_counter()

    # Shuffle by index. This replaces what a DataLoader does, but the index
    # tensor lives on the GPU, so no data crosses the bus.
    order: Tensor = torch.randperm(images.shape[0], device=images.device)

    for i in range(0, images.shape[0], BATCH_SIZE):
        batch: Tensor = order[i:i + BATCH_SIZE]
        loss: Tensor = loss_fn(model(images[batch]), labels[batch])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    return time.perf_counter() - start


def build_parts(device: str) -> Tuple[NeuralNetwork, CrossEntropyLoss, Optimizer]:
    """Make a fresh model, loss function and optimizer for one measurement."""
    torch.manual_seed(0)
    model: NeuralNetwork = NeuralNetwork().to(device)
    loss_fn: CrossEntropyLoss = CrossEntropyLoss()
    # If you move the model to the GPU, you must declare the optimizer *afterwards*. Otherwise, training won't work.
    optimizer: Optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer


def main() -> None:

    if not torch.cuda.is_available():
        print("This demo needs an NVIDIA GPU. No CUDA device was found.")
        return

    device: str = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")

    transfer_bandwidth(device)

    heading("Loading FashionMNIST")

    transform: v2.Compose = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data: FashionMNIST = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    print(f"Samples: {len(training_data)}")

    heading("One training pass, three ways to feed the GPU")

    results = []

    # 1 and 2. A DataLoader, with pinning off and then on.
    for pin_memory in (False, True):
        data_loader: DataLoader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
        model, loss_fn, optimizer = build_parts(device)
        seconds: float = time_dataloader_epoch(data_loader, model, loss_fn, optimizer, device, non_blocking=pin_memory)
        label: str = f"DataLoader, pin_memory={pin_memory}"
        results.append((label, seconds))
        print(f"{label:<34} {seconds:7.2f} s")

    # 3. The whole dataset on the GPU. FashionMNIST stores raw uint8 pixels,
    # so scale them the same way v2.ToDtype(scale=True) does.
    images: Tensor = (training_data.data.float() / 255.0).unsqueeze(1).to(device)
    labels: Tensor = training_data.targets.to(device)
    size_mb: float = images.element_size() * images.nelement() / (1024 * 1024)

    model, loss_fn, optimizer = build_parts(device)
    seconds = time_gpu_resident_epoch(images, labels, model, loss_fn, optimizer)
    label = "Whole dataset on the GPU"
    results.append((label, seconds))
    print(f"{label:<34} {seconds:7.2f} s")
    print(f"\nThe dataset takes {size_mb:.1f} MB of GPU memory.")

    heading("How much faster")

    baseline: float = results[0][1]
    for label, seconds in results:
        print(f"{label:<34} {baseline / seconds:5.1f}x")

    heading("What to take away")
    print("Pinning helps, but it only makes the copy faster.")
    print("Removing the copy helps far more.")
    print("Put the dataset on the GPU whenever it fits.")


if __name__ == "__main__":
    main()
