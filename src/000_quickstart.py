import os
from typing import List

import numpy as np
import torch
from numpy import ndarray
from onnxruntime import InferenceSession
from torch import Tensor
from torch.nn import Flatten, Sequential, Linear, ReLU, CrossEntropyLoss, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
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
        """Reload a trained model from a .pt file."""
        model: NeuralNetwork = NeuralNetwork()
        model.load_state_dict(torch.load(filepath, weights_only=True, map_location=model.device))
        return model

    def export_onnx(self, filepath: str) -> None:
        """Write the model as ONNX, so a runtime outside Python can run it.

        ONNX is an open file format for a trained model. It holds the graph
        of operations and the weights, so a C++ or a browser runtime can run
        the model with no PyTorch and no Python.
        """

        # eval() switches off training behavior before the export.
        self.eval()

        # The exporter runs the model once to record the graph, so it needs
        # an example input of the right shape and dtype.
        example: Tensor = torch.zeros(1, 1, 28, 28, device=self.device)

        # dynamic_shapes marks dimension 0 of the input as free, so the file
        # accepts any batch size, not only the batch size of the example.
        # external_data=False keeps the weights inside the one file.
        # verbose=False stops a Unicode print that fails on a cp1252 console.
        torch.onnx.export(
            self,
            (example,),
            filepath,
            input_names=["image"],
            output_names=["logits"],
            dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
            external_data=False,
            verbose=False
        )

        size_mb: float = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Exported {filepath} ({size_mb:.1f} MB)")

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

    def predict(self, test_data: FashionMNIST) -> str:
        """Predict the class of the first test image. Return the class name."""
        self.eval()  # set the dropout and batch normalization layers to evaluation mode for consistent outputs
        x, y = test_data[0][0], test_data[0][1]
        with torch.no_grad():
            x = x.to(self.device)
            pred = self(x)
            predicted, actual = NeuralNetwork.CLASSES[pred[0].argmax(0)], NeuralNetwork.CLASSES[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
            return predicted

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


class OnnxNeuralNetwork(NeuralNetwork):
    """A NeuralNetwork that runs an ONNX file instead of its own weights.

    Only forward changes. Every other method, such as predict and test, is inherited and works without a change.
    """

    def __init__(self, filepath: str) -> None:
        super().__init__()

        # An InferenceSession is the ONNX Runtime object that holds a loaded model.
        # It runs on the CPU here, so the answer does not depend on the graphics card.
        self.session: InferenceSession = InferenceSession(filepath, providers=["CPUExecutionProvider"])

    @staticmethod
    def from_file(filepath: str) -> 'OnnxNeuralNetwork':
        """Reload a trained model from a .onnx file. This mirrors NeuralNetwork.from_file."""
        return OnnxNeuralNetwork(filepath)

    def forward(self, x: Tensor) -> Tensor:
        # ONNX Runtime takes numpy arrays, not tensors. The model wants [batch, channel, height, width],
        # so reshape one image or a whole batch into that form.
        images: ndarray = x.detach().cpu().reshape(-1, 1, 28, 28).numpy()
        logits: ndarray = self.session.run(None, {"image": images})[0]
        return torch.from_numpy(logits).to(x.device)


def compare_model_predictions(model1: NeuralNetwork, model2: NeuralNetwork, test_data: FashionMNIST) -> None:
    """Check that two models pick the same class for every test image."""

    model1.eval()
    model2.eval()

    # The ONNX export marked dimension 0 as dynamic, so one call takes all 10000 images.
    count: int = len(test_data)
    images: Tensor = torch.stack([test_data[i][0] for i in range(count)])

    with torch.no_grad():
        logits1: ndarray = model1(images.to(model1.device)).cpu().numpy()
        logits2: ndarray = model2(images.to(model2.device)).cpu().numpy()

    matches: int = int((logits1.argmax(axis=1) == logits2.argmax(axis=1)).sum())

    # The two runtimes add the numbers in a different order, so the raw
    # scores differ by a tiny amount. The chosen class must still match.
    largest_gap: float = float(np.abs(logits1 - logits2).max())

    print(f"Predictions that match: {matches} of {count}")
    print(f"Largest difference in the raw scores: {largest_gap:.3e}")

    if matches != count:
        raise RuntimeError("The two models disagree.")
    print("The two models agree on every image.")


def main() -> None:

    transform: v2.Transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data: FashionMNIST = FashionMNIST(root="data", train=True, download=True, transform=transform)
    test_data: FashionMNIST = FashionMNIST(root="data", train=False, download=True, transform=transform)

    model: NeuralNetwork = NeuralNetwork()
    model.print_model()
    model.learn(training_data, test_data)

    PT_FILEPATH: str = "model.pt"
    torch.save(model.state_dict(), PT_FILEPATH)
    reloaded_pt_model: NeuralNetwork = NeuralNetwork.from_file(PT_FILEPATH)
    pt_prediction: str = reloaded_pt_model.predict(test_data)

    ONNX_FILEPATH: str = "model.onnx"
    model.export_onnx(ONNX_FILEPATH)
    reloaded_onnx_model: OnnxNeuralNetwork = OnnxNeuralNetwork.from_file(ONNX_FILEPATH)
    onnx_prediction: str = reloaded_onnx_model.predict(test_data)

    print(f'The .pt file says "{pt_prediction}". The .onnx file says "{onnx_prediction}".')
    print("\nComparing predictions for the full dataset...")
    compare_model_predictions(reloaded_pt_model, reloaded_onnx_model, test_data)


if __name__ == "__main__":
    main()
