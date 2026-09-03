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
        self.flatten: Flatten = Flatten()
        self.linear_relu_stack: Sequential = Sequential(
            Linear(28*28, 512),
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, 10)
        )
        self.loss_fn: CrossEntropyLoss = CrossEntropyLoss()
        self.optimizer: Optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

    @staticmethod
    def from_file(detected_device: str, filepath: str) -> 'NeuralNetwork':
        # Reload the weights and use the trained model.
        model2: NeuralNetwork = NeuralNetwork().to(detected_device)
        model2.load_state_dict(torch.load(filepath, weights_only=True))
        return model2

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.flatten(x)
        logits: Tensor = self.linear_relu_stack(x)
        return logits

    def learn(
            self,
            detected_device: str,
            training_data: FashionMNIST,
            test_data: FashionMNIST):

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
            self.learn_one_epoch(training_data, training_data_loader, detected_device)
            self.test(test_data_loader, detected_device)
        print("Done!")

    def learn_one_epoch(
            self,
            data_set: FashionMNIST,
            data_loader: DataLoader,
            detected_device: str) -> None:

        size: int = len(data_set)
        self.train()

        for batch, (X, y) in enumerate(data_loader):

            X, y = X.to(detected_device), y.to(detected_device)

            # Compute prediction error
            prediction: Tensor = self(X)
            loss = self.loss_fn(prediction, y)

            # Run backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d} / {size:>5d}]")

    def predict(self, detected_device: str, test_data: FashionMNIST):
        self.eval()
        x, y = test_data[0][0], test_data[0][1]
        with torch.no_grad():
            x = x.to(detected_device)
            pred = self(x)
            predicted, actual = NeuralNetwork.CLASSES[pred[0].argmax(0)], NeuralNetwork.CLASSES[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')

    def test(self, data_loader: DataLoader, detected_device: str) -> None:

        size: int = len(data_loader.dataset)
        num_batches: int = len(data_loader)
        self.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():

            for X, y in data_loader:
                X, y = X.to(detected_device), y.to(detected_device)
                prediction: Tensor = self(X)
                test_loss += self.loss_fn(prediction, y).item()
                correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n  Accuracy: {(100*correct):>0.1f}%, Average Loss: {test_loss:>8f}\n")


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

    print(torch.cuda.is_available())
    accelerator: torch.device | None = torch.accelerator.current_accelerator()
    detected_device: str = accelerator.type if accelerator is not None else "cpu"
    print(f"Using {detected_device} device")
    model: NeuralNetwork = NeuralNetwork().to(detected_device)
    print(model)
    model.learn(detected_device, training_data, test_data)

    FILEPATH: str = "model.pt"
    torch.save(model.state_dict(), FILEPATH)
    reloaded_model: NeuralNetwork = NeuralNetwork.from_file(detected_device, FILEPATH)
    reloaded_model.predict(detected_device, test_data)


if __name__ == "__main__":
    main()
