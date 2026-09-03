"""A demo of how to build a neural network.

The code follows the official tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

A neural network is a stack of layers. Every layer takes a tensor and
returns a tensor. torch.nn holds the layers you need to build one.
"""

from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import Parameter


def heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


class NeuralNetwork(nn.Module):
    """A small network that sorts a 28x28 image into one of ten classes."""

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


def get_device() -> str:

    heading("Get Device for Training")

    accelerator: torch.device | None = torch.accelerator.current_accelerator()
    device: str = accelerator.type if accelerator is not None else "cpu"
    print(f"Using {device} device")
    return device


def define_the_class(device: str) -> NeuralNetwork:

    heading("Define the Class")

    model: NeuralNetwork = NeuralNetwork().to(device)
    print(model)
    return model


def use_the_model(model: NeuralNetwork, device: str) -> None:

    heading("Use the model")

    # Never call model.forward() by hand. Call the model itself.
    x: Tensor = torch.rand(1, 28, 28, device=device)
    logits: Tensor = model(x)

    # The model returns logits: raw scores, one per class. Softmax turns
    # those scores into probabilities that add up to 1.0.
    pred_probab: Tensor = nn.Softmax(dim=1)(logits)
    y_pred: Tensor = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")


def model_layers() -> Tuple[Tensor, Tensor]:

    heading("nn.Flatten")

    # A batch of three 28x28 images.
    input_image: Tensor = torch.rand(3, 28, 28)
    print(input_image.size())

    # Flatten turns each 28x28 image into one row of 784 values. The batch
    # size of 3 stays as it is.
    flatten: nn.Flatten = nn.Flatten()
    flat_image: Tensor = flatten(input_image)
    print(flat_image.size())

    heading("nn.Linear")

    # A Linear layer applies weights and a bias to every input value.
    layer1: nn.Linear = nn.Linear(in_features=28 * 28, out_features=20)
    hidden1: Tensor = layer1(flat_image)
    print(hidden1.size())

    heading("nn.ReLU")

    # ReLU replaces every negative value with 0.0. Without a step like this,
    # a stack of Linear layers could only learn straight lines.
    print(f"Before ReLU: {hidden1}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1}")

    return flat_image, hidden1


def sequential_and_softmax() -> None:

    heading("nn.Sequential")

    # Sequential runs the layers in order, one after the other.
    flatten: nn.Flatten = nn.Flatten()
    layer1: nn.Linear = nn.Linear(in_features=28 * 28, out_features=20)
    seq_modules: nn.Sequential = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    input_image: Tensor = torch.rand(3, 28, 28)
    logits: Tensor = seq_modules(input_image)
    print(f"Logits size: {logits.size()}")

    heading("nn.Softmax")

    softmax: nn.Softmax = nn.Softmax(dim=1)
    pred_probab: Tensor = softmax(logits)
    print(f"Probabilities: {pred_probab}")
    print(f"Each row adds up to: {pred_probab.sum(dim=1)}")


def model_parameters(model: NeuralNetwork) -> None:

    heading("Model Parameters")

    # named_parameters lists every weight and bias the model will learn.
    print(f"Model structure: {model}\n\n")

    name: str
    param: Parameter
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


def main() -> None:
    device: str = get_device()
    model: NeuralNetwork = define_the_class(device)
    use_the_model(model, device)
    model_layers()
    sequential_and_softmax()
    model_parameters(model)


if __name__ == "__main__":
    main()
