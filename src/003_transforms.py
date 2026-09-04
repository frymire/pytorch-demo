"""A demo of dataset transforms.

The code follows the official tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html

A dataset gives features and labels. Both often need a change before training.
A transform changes the feature. A target_transform changes the label.
"""

from typing import Callable

import torch
from torch import Tensor
from torch.nn.functional import one_hot
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2


def heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:

    heading("The raw dataset, with no transforms")
    raw_data: FashionMNIST = FashionMNIST(root="data", train=True, download=True)
    raw_image, raw_label = raw_data[0]  # a PIL image and an int label
    print(f"Feature type: {type(raw_image).__name__}")
    print(f"Label: {raw_label}  (type {type(raw_label).__name__})")

    heading("Lambda Transforms")
    label_to_one_hot: Callable = v2.Lambda(lambda y: one_hot(torch.tensor(y), num_classes=10).float())
    print(f"Label {raw_label} becomes {label_to_one_hot(raw_label)}")

    heading("Both transforms together")
    transform: v2.Transform = v2.Compose([
        v2.ToImage(),  # makes a tensor
        v2.ToDtype(torch.float32, scale=True)  # scales 0-255 values to [0.0, 1.0]
    ])
    transformed_data: FashionMNIST = FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
        target_transform=label_to_one_hot
    )

    image: Tensor
    label: Tensor
    image, label = transformed_data[0]

    print(f"Feature shape: {image.shape}")
    print(f"Feature dtype: {image.dtype}")
    print(f"Feature range: {image.min().item():.1f} to {image.max().item():.1f}")
    print(f"Label shape: {label.shape}")
    print(f"Label: {label}")
    print(f"Label argmax: {label.argmax().item()}  (matches the raw label {raw_label})")


if __name__ == "__main__":
    main()
