from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from matplotlib import pyplot
from torch import Tensor
from torch.nn import CrossEntropyLoss, Conv2d, MaxPool2d, Linear, Module
from torch.nn.functional import relu
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize

# Anchor every path to the project root. The folders then stay in one place,
# whatever working directory you start the script from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / 'data'
RUNS_DIR: Path = PROJECT_ROOT / 'runs'

# Class labels
classes: Tuple[str, ...] = (
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle Boot'
)


def matplotlib_imshow(image: Tensor, one_channel: bool = False) -> None:
    if one_channel:
        image = image.mean(dim=0)
    image = image / 2 + 0.5  # unnormalize
    numpy_image: np.ndarray = image.numpy()
    if one_channel:
        pyplot.imshow(numpy_image, cmap="Greys")
    else:
        pyplot.imshow(np.transpose(numpy_image, (1, 2, 0)))


class Net(Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1: Conv2d = Conv2d(1, 6, 5)
        self.pool: MaxPool2d = MaxPool2d(2, 2)
        self.conv2: Conv2d = Conv2d(6, 16, 5)
        self.fc1: Linear = Linear(16 * 4 * 4, 120)
        self.fc2: Linear = Linear(120, 84)
        self.fc3: Linear = Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x


def select_n_random(data_: Tensor, labels_: Tensor, n: int = 100) -> Tuple[Tensor, Tensor]:
    """Select a random subset of data and corresponding labels."""
    assert len(data_) == len(labels_)
    perm: Tensor = torch.randperm(len(data_))
    return data_[perm][:n], labels_[perm][:n]


def build_data_loaders() -> Tuple[FashionMNIST, DataLoader, DataLoader]:
    """Gather datasets and prepare them for consumption.

    This step writes nothing to TensorBoard. It only supplies the images that
    the other steps log.
    """

    transform: Compose = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    # Store separate training and validations splits in DATA_DIR
    training_set: FashionMNIST = FashionMNIST(DATA_DIR, download=True, train=True, transform=transform)
    validation_set: FashionMNIST = FashionMNIST(DATA_DIR, download=True, train=False, transform=transform)

    training_loader: DataLoader = DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)
    validation_loader: DataLoader = DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=2)

    return training_set, training_loader, validation_loader


def create_writer() -> SummaryWriter:
    """Open one log folder for this run.

    Result in TensorBoard: the left panel lists one run per folder. Without the
    timestamp, every run would draw on top of the last one in the same chart.
    """

    # Default log_dir is "runs/", but it's good to be specific.
    run_dir: Path = RUNS_DIR / datetime.now().strftime('%Y%m%d-%H%M%S')
    return SummaryWriter(str(run_dir), flush_secs=10)


def log_sample_images(writer: SummaryWriter, training_loader: DataLoader) -> None:
    """Log one batch of 4 images as a single picture.

    Result in TensorBoard: the IMAGES tab shows a card named
    "Four Fashion-MNIST Images" with the 4 clothing pictures side by side.
    """

    # Extract a batch of 4 images
    images: Tensor
    labels: Tensor
    images, labels = next(iter(training_loader))

    # Create a grid from the images and show them
    image_grid: Tensor = torchvision.utils.make_grid(images)
    matplotlib_imshow(image_grid, one_channel=True)

    writer.add_image('Four Fashion-MNIST Images', image_grid)
    writer.flush()


def train(writer: SummaryWriter, net: Net, training_loader: DataLoader, validation_loader: DataLoader) -> None:
    """Run the training loop and log the loss every 1000 mini-batches.

    Result in TensorBoard: the SCALARS tab shows a chart named
    "Training vs. Validation Loss" with 2 lines, Training and Validation. Both
    lines should fall as the model learns. A validation line that stops falling
    while the training line keeps falling means the model overfits.
    """

    criterion: CrossEntropyLoss = CrossEntropyLoss()
    optimizer: Optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(len(validation_loader))

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss: float = 0.0

        data: List[Tensor]
        for i, data in enumerate(training_loader, 0):

            # basic training loop
            inputs: Tensor
            labels: Tensor
            inputs, labels = data
            optimizer.zero_grad()
            outputs: Tensor = net(inputs)
            loss: Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000 == 999:    # Every 1000 mini-batches...

                print('Batch {}'.format(i + 1))
                # Check against the validation set
                running_validation_loss: float = 0.0

                net.train(False)  # don't track gradients for validation
                vdata: List[Tensor]
                for j, vdata in enumerate(validation_loader, 0):
                    validation_inputs: Tensor
                    validation_labels: Tensor
                    validation_inputs, validation_labels = vdata
                    validation_outputs: Tensor = net(validation_inputs)
                    validation_loss: Tensor = criterion(validation_outputs, validation_labels)
                    running_validation_loss += validation_loss.item()
                net.train(True)  # turn gradients back on for training

                avg_loss: float = running_loss / 1000
                avg_validation_loss: float = running_validation_loss / len(validation_loader)

                # Log the running loss averaged per batch
                writer.add_scalars(
                    'Training vs. Validation Loss',
                    {'Training': avg_loss, 'Validation': avg_validation_loss},
                    epoch * len(training_loader) + i
                )

                running_loss = 0.0

    print('Finished Training')
    writer.flush()


def log_model_graph(writer: SummaryWriter, net: Net, training_loader: DataLoader) -> None:
    """Trace one batch through the model and log the shape of the model.

    Result in TensorBoard: a GRAPHS tab appears. It draws the layers as boxes
    joined by arrows. Double-click a box to open it and see the layers inside.
    """

    # Again, grab a single mini-batch of images
    images: Tensor
    labels: Tensor
    images, labels = next(iter(training_loader))

    # add_graph() will trace the sample input through your model,
    # and render it as a graph.
    writer.add_graph(net, images)
    writer.flush()


def log_embedding(writer: SummaryWriter, training_set: FashionMNIST) -> None:
    """Log 100 random images as points in a space of 784 dimensions.

    Result in TensorBoard: a PROJECTOR tab appears. It draws the 100 images as
    points in 3D. Images of the same class should sit near each other. The tab
    appears only after you restart the server, because the server fixes the tab
    list when it starts.
    """

    # Extract a random subset of data
    images: Tensor
    labels: Tensor
    images, labels = select_n_random(training_set.data, training_set.targets)

    # get the class labels for each image
    class_labels: List[str] = [classes[label] for label in labels]

    # log embeddings
    features: Tensor = images.view(-1, 28 * 28)
    writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))
    writer.flush()


def main() -> None:

    training_set: FashionMNIST
    training_loader: DataLoader
    validation_loader: DataLoader
    training_set, training_loader, validation_loader = build_data_loaders()

    writer: SummaryWriter = create_writer()

    # To view, start TensorBoard on the command line with:
    #   poetry run tensorboard --logdir=runs --reload_interval=5
    # ...and open a browser tab to http://localhost:6006/

    net: Net = Net()

    log_sample_images(writer, training_loader)              # IMAGES tab
    train(writer, net, training_loader, validation_loader)  # SCALARS tab
    log_model_graph(writer, net, training_loader)           # GRAPHS tab
    log_embedding(writer, training_set)                     # PROJECTOR tab, must restart server after running this

    writer.close()


# num_workers > 0 starts worker processes. On Windows each worker re-imports
# this file, so the runtime code must sit behind this guard.
if __name__ == "__main__":
    main()
