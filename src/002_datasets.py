"""A demo of Datasets and DataLoaders.

The code follows the official tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

A Dataset holds the samples and their labels. A DataLoader wraps a Dataset
and hands out small batches of it.
"""

import os
import tempfile
from typing import Any, Callable, Dict, Optional, Tuple, List

import pandas
import torch
from matplotlib import pyplot
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchvision.io import decode_image, write_png
from torchvision.transforms import v2


def main() -> None:
    training_data, test_data = load_a_dataset()
    visualize_the_dataset(training_data)
    create_custom_dataset()
    use_a_dataloader(training_data, test_data)


label_to_name: Dict[int, str] = {
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


def print_heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def load_a_dataset() -> Tuple[FashionMNIST, FashionMNIST]:
    print_heading("Loading a Dataset")
    transform: v2.Compose = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data: FashionMNIST = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    test_data: FashionMNIST = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    print(f"Training samples: {len(training_data)}")
    print(f"Test samples: {len(test_data)}")
    return training_data, test_data


def visualize_the_dataset(training_data: FashionMNIST) -> None:

    print_heading("Iterating and Visualizing the Dataset")

    figure = pyplot.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_index: int = torch.randint(len(training_data), size=(1,)).item()
        image, label = training_data[sample_index]
        figure.add_subplot(rows, cols, i)
        pyplot.title(label_to_name[label])
        pyplot.axis("off")
        pyplot.imshow(image.squeeze(), cmap="gray")
    print("Showing a 3x3 grid of random samples. Close the window to go on.")
    pyplot.show()


def create_custom_dataset() -> None:

    print_heading("Creating a Custom Dataset for your files")

    # Build a tiny image folder and a labels CSV, so the class has real data.
    with tempfile.TemporaryDirectory() as image_dir:

        rows: List[Dict[str, Any]] = []
        for i in range(3):
            filename: str = f"image_{i}.png"
            pixels: Tensor = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
            write_png(pixels, os.path.join(image_dir, filename))
            rows.append({"filename": filename, "label": i})

        labels_file: str = os.path.join(image_dir, "labels.csv")
        DataFrame(rows).to_csv(labels_file, index=False)

        data_set: CustomImageDataset = CustomImageDataset(labels_file, image_dir)
        print(f"Samples: {len(data_set)}")

        image, label = data_set[0]
        print(f"Sample 0 image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Sample 0 label: {label}")


class CustomImageDataset(Dataset):
    """
    A Dataset that reads images from a folder and labels from a CSV file.
    A custom Dataset must define three methods: __init__, __len__, and __getitem__.
    """

    def __init__(
            self,
            filepath: str,
            image_dir: str,
            image_transform: Optional[Callable] = None,
            label_transform: Optional[Callable] = None) -> None:

        # The CSV holds one row per image: the file name, then the label.
        self.data_frame: DataFrame = pandas.read_csv(filepath)
        self.image_dir: str = image_dir
        self.image_transform: Optional[Callable] = image_transform
        self.label_transform: Optional[Callable] = label_transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data_frame)

    def __getitem__(self, index: int) -> Tuple[Tensor, Any]:
        """Load and return one sample at the given position."""
        image_path: str = os.path.join(self.image_dir, self.data_frame.iloc[index, 0])
        image: Tensor = decode_image(image_path)
        label: Any = self.data_frame.iloc[index, 1]
        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label


def use_a_dataloader(training_data: FashionMNIST, test_data: FashionMNIST) -> None:

    print_heading("Preparing your data for training with DataLoaders")

    # shuffle=True reorders the samples after every pass over the data.
    training_data_loader: DataLoader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_data_loader: DataLoader = DataLoader(test_data, batch_size=64, shuffle=True)
    print(f"Training batches: {len(training_data_loader)}")
    print(f"Test batches: {len(test_data_loader)}")

    print_heading("Iterate through the DataLoader")

    # Each step of the DataLoader gives one batch of features and labels.
    training_features: Tensor
    training_labels: Tensor
    training_features, training_labels = next(iter(training_data_loader))
    print(f"Feature batch shape: {training_features.size()}")
    print(f"Labels batch shape: {training_labels.size()}")

    image: Tensor = training_features[0].squeeze()
    label: Tensor = training_labels[0]
    pyplot.imshow(image, cmap="gray")
    print("Showing the first image of the batch. Close the window to go on.")
    pyplot.show()
    print(f"Label: {label} ({label_to_name[int(label)]})")


if __name__ == "__main__":
    main()
