"""A demo of Datasets and DataLoaders.

The code follows the official tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

A Dataset holds the samples and their labels. A DataLoader wraps a Dataset
and hands out small batches of it.
"""

import os
import tempfile
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchvision.io import decode_image, write_png
from torchvision.transforms import v2

LABELS_MAP: Dict[int, str] = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


def heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


class CustomImageDataset(Dataset):
    """A Dataset that reads images from a folder and labels from a CSV file.

    A custom Dataset must define three methods: __init__, __len__ and
    __getitem__.
    """

    def __init__(
            self,
            annotations_file: str,
            img_dir: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None) -> None:

        # The CSV holds one row per image: the file name, then the label.
        self.img_labels: DataFrame = pd.read_csv(annotations_file)
        self.img_dir: str = img_dir
        self.transform: Optional[Callable] = transform
        self.target_transform: Optional[Callable] = target_transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Any]:
        """Load and return one sample at the given position."""
        img_path: str = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image: Tensor = decode_image(img_path)
        label: Any = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def load_a_dataset() -> Tuple[FashionMNIST, FashionMNIST]:

    heading("Loading a Dataset")

    transform: v2.Compose = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    training_data: FashionMNIST = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform)

    test_data: FashionMNIST = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform)

    print(f"Training samples: {len(training_data)}")
    print(f"Test samples: {len(test_data)}")

    return training_data, test_data


def visualize_the_dataset(training_data: FashionMNIST) -> None:

    heading("Iterating and Visualizing the Dataset")

    # A Dataset behaves like a list. training_data[i] gives one sample.
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx: int = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(LABELS_MAP[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    print("Showing a 3x3 grid of random samples. Close the window to go on.")
    plt.show()


def custom_dataset() -> None:

    heading("Creating a Custom Dataset for your files")

    # Build a tiny image folder and a labels CSV, so the class has real data.
    with tempfile.TemporaryDirectory() as img_dir:

        rows = []
        for i in range(3):
            file_name: str = f"image_{i}.png"
            pixels: Tensor = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
            write_png(pixels, os.path.join(img_dir, file_name))
            rows.append({"file_name": file_name, "label": i})

        annotations_file: str = os.path.join(img_dir, "labels.csv")
        pd.DataFrame(rows).to_csv(annotations_file, index=False)

        data_set: CustomImageDataset = CustomImageDataset(annotations_file, img_dir)
        print(f"Samples: {len(data_set)}")

        image, label = data_set[0]
        print(f"Sample 0 image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Sample 0 label: {label}")


def use_a_dataloader(training_data: FashionMNIST, test_data: FashionMNIST) -> None:

    heading("Preparing your data for training with DataLoaders")

    # shuffle=True reorders the samples after every pass over the data.
    train_dataloader: DataLoader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader: DataLoader = DataLoader(test_data, batch_size=64, shuffle=True)
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")

    heading("Iterate through the DataLoader")

    # Each step of the DataLoader gives one batch of features and labels.
    train_features: Tensor
    train_labels: Tensor
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    img: Tensor = train_features[0].squeeze()
    label: Tensor = train_labels[0]
    plt.imshow(img, cmap="gray")
    print("Showing the first image of the batch. Close the window to go on.")
    plt.show()
    print(f"Label: {label} ({LABELS_MAP[int(label)]})")


def main() -> None:
    training_data, test_data = load_a_dataset()
    visualize_the_dataset(training_data)
    custom_dataset()
    use_a_dataloader(training_data, test_data)


if __name__ == "__main__":
    main()
