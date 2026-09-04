from typing import List

import torch
from torch import Tensor
from torch.nn import Flatten, Sequential, Linear, ReLU, CrossEntropyLoss, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2


class NeuralNetwork(Module):

    CLASSES: List[str] = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    def __init__(self) -> None:

        super().__init__()

        # Detect the accelerator, then move every parameter onto it. Do this
        # before the optimizer is built, so the optimizer sees final tensors.
        accelerator: torch.device | None = torch.accelerator.current_accelerator()
        self.detected_device: str = accelerator.type if accelerator is not None else "cpu"
        self.to(self.detected_device)

        self.flatten: Flatten = Flatten()

        self.layers: Sequential = Sequential(
            Linear(28*28, 512),
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, 10)
        )

        self.loss: CrossEntropyLoss = CrossEntropyLoss()
        self.optimizer: Optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

    @property
    def device(self) -> torch.device:
        # The true device of the weights. This stays correct after a later
        # .to() call, which self.detected_device cannot do.
        return next(self.parameters()).device

    @staticmethod
    def from_file(filepath: str) -> 'NeuralNetwork':
        # Reload the weights and use the trained model.
        model: NeuralNetwork = NeuralNetwork()
        model.load_state_dict(torch.load(filepath, weights_only=True, map_location=model.device))
        return model

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(self.flatten(x))  # logits

    def learn(self, training_data: FashionMNIST, test_data: FashionMNIST):

        NUM_EPOCHS: int = 5
        BATCH_SIZE: int = 64

        training_data_loader: DataLoader = DataLoader(training_data, batch_size=BATCH_SIZE)
        test_data_loader: DataLoader = DataLoader(test_data, batch_size=BATCH_SIZE)

        for X, y in test_data_loader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        for t in range(NUM_EPOCHS):
            print(f"Epoch: {t + 1}\n-------------------------------------")
            self.learn_one_epoch(training_data, training_data_loader)
            self.test(test_data_loader)
        print("Done!")

    def learn_one_epoch(self, data_set: FashionMNIST, data_loader: DataLoader) -> None:

        size: int = len(data_set)
        self.train()

        for batch, (X, y) in enumerate(data_loader):

            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            prediction: Tensor = self(X)
            loss: Tensor = self.loss(prediction, y)

            # Run backpropagation
            loss.backward()  # compute gradients
            self.optimizer.step()  # update weights based on the computed gradients
            self.optimizer.zero_grad()  # reset gradients to zero, since backward() *accumulates* with each call

            if batch % 100 == 0:
                num_complete: int = (batch + 1) * len(X)
                print(f"loss: {loss.item():>7f} [{num_complete:>5d} / {size:>5d}]")

    def predict(self, test_data: FashionMNIST):
        self.eval()  # set the dropout and batch normalization layers to evaluation mode for consistent outputs
        x, y = test_data[0][0], test_data[0][1]
        with torch.no_grad():
            x = x.to(self.device)
            pred = self(x)
            predicted, actual = NeuralNetwork.CLASSES[pred[0].argmax(0)], NeuralNetwork.CLASSES[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')

    def test(self, data_loader: DataLoader) -> None:

        size: int = len(data_loader.dataset)
        num_batches: int = len(data_loader)
        self.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():

            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                prediction: Tensor = self(X)
                test_loss += self.loss(prediction, y).item()
                correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n  Accuracy: {(100*correct):>0.1f}%, Average Loss: {test_loss:>8f}\n")

    def print_model(self):
        print(self)
        print(f"Using {self.detected_device} device.")


def main() -> None:

    transform: v2.Transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data: FashionMNIST = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    test_data: FashionMNIST = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

    model: NeuralNetwork = NeuralNetwork()
    model.print_model()
    model.learn(training_data, test_data)

    FILEPATH: str = "model.pt"
    torch.save(model.state_dict(), FILEPATH)
    reloaded_model: NeuralNetwork = NeuralNetwork.from_file(FILEPATH)
    reloaded_model.predict(test_data)


if __name__ == "__main__":
    main()
