
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2


def main() -> None:

    training_data: FashionMNIST = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

    test_data: FashionMNIST = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    )

    BATCH_SIZE: int = 64

    _training_data_loader: DataLoader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_data_loader: DataLoader = DataLoader(test_data, batch_size=BATCH_SIZE)

    for X, y in test_data_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    print(torch.cuda.is_available())
    device: str = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
