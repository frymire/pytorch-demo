"""A demo of the training loop.

The code follows the official tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

Training repeats one cycle: guess, measure the error, and change the weights
to make the error smaller.
"""

from typing import Sized, cast

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

# Hyperparameters. These are the numbers you choose, not the ones the model
# learns.
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 64
EPOCHS: int = 10


def heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


class NeuralNetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.flatten: nn.Flatten = nn.Flatten()
        self.linear_relu_stack: nn.Sequential = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        logits: Tensor = self.linear_relu_stack(x)
        return logits


def train_loop(
        dataloader: DataLoader,
        model: NeuralNetwork,
        loss_fn: CrossEntropyLoss,
        optimizer: Optimizer) -> None:
    """Run one pass over the whole training set."""

    # DataLoader.dataset is declared as Dataset, which has no length. The
    # real dataset here does have one, so tell the type checker.
    size: int = len(cast(Sized, dataloader.dataset))

    # train() turns on training behaviour, such as dropout.
    model.train()

    for batch, (X, y) in enumerate(dataloader):

        # Guess, then measure how wrong the guess was.
        pred: Tensor = model(X)
        loss: Tensor = loss_fn(pred, y)

        # Work out the blame, apply the fix, then clear the blame.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_value: float = loss.item()
            current: int = batch * BATCH_SIZE + len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(
        dataloader: DataLoader,
        model: NeuralNetwork,
        loss_fn: CrossEntropyLoss) -> None:
    """Score the model on data it never trained on."""

    # eval() turns off training behaviour.
    model.eval()

    size: int = len(cast(Sized, dataloader.dataset))
    num_batches: int = len(dataloader)
    test_loss: float = 0.0
    correct: float = 0.0

    # no_grad stops autograd from tracking. Testing changes no weights.
    with torch.no_grad():
        for X, y in dataloader:
            pred: Tensor = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
          f"Avg loss: {test_loss:>8f} \n")


def main() -> None:

    heading("Prerequisite Code")

    transform: v2.Compose = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    training_data: FashionMNIST = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform)

    test_data: FashionMNIST = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform)

    train_dataloader: DataLoader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_dataloader: DataLoader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model: NeuralNetwork = NeuralNetwork()
    print(model)

    heading("Loss Function and Optimizer")

    # CrossEntropyLoss scores how far the guessed class scores are from the
    # right answer.
    loss_fn: CrossEntropyLoss = nn.CrossEntropyLoss()

    # SGD is the rule that changes the weights: subtract the gradient times
    # the learning rate.
    optimizer: Optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    print(f"Loss function: {loss_fn}")
    print(f"Optimizer: {optimizer}")

    heading("Full Implementation")

    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":
    main()
