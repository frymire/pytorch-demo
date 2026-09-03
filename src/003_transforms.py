"""A demo of dataset transforms.

The code follows the official tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html

A dataset gives features and labels. Both often need a change before training.
A transform changes the feature. A target_transform changes the label.
"""

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2


def heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:

    heading("The raw dataset, with no transforms")
    # FashionMNIST gives a PIL image and an int label.
    raw: FashionMNIST = datasets.FashionMNIST(root="data", train=True, download=True)
    raw_image, raw_label = raw[0]
    print(f"Feature type: {type(raw_image).__name__}")
    print(f"Label: {raw_label}  (type {type(raw_label).__name__})")

    heading("Lambda Transforms")
    # A Lambda transform runs any function you give it. This one turns the
    # int label into a one-hot vector: a vector of ten zeros, with a single
    # 1.0 at the position of the correct class.
    target_transform: Callable = v2.Lambda(
        lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
    )
    print(f"Label {raw_label} becomes {target_transform(raw_label)}")

    heading("Both transforms together")
    ds: FashionMNIST = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        # ToImage makes a tensor. ToDtype scales the 0-255 values into 0.0-1.0.
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        target_transform=target_transform,
    )

    image: Tensor
    label: Tensor
    image, label = ds[0]

    print(f"Feature shape: {image.shape}")
    print(f"Feature dtype: {image.dtype}")
    print(f"Feature range: {image.min().item():.1f} to {image.max().item():.1f}")
    print(f"Label shape: {label.shape}")
    print(f"Label: {label}")
    print(f"Label argmax: {label.argmax().item()}  (matches the raw label {raw_label})")


if __name__ == "__main__":
    main()
