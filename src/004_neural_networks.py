"""A demo of how to build a neural network.

The code follows the official tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

A neural network is a stack of layers. Every layer takes a tensor and
returns a tensor. torch.nn holds the layers you need to build one.
"""

from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Parameter, Flatten, Sequential, Linear, ReLU, Module, Softmax


def main() -> None:
    device: str = get_device()
    model: NeuralNetwork = define_the_class(device)
    use_the_model(model, device)
    model_layers()
    sequential_and_softmax()
    model_parameters(model)


def heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


class NeuralNetwork(Module):
    """A small network that sorts a 28x28 image into one of ten classes."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten: Flatten = Flatten()
        self.linear_relu_stack: Sequential = Sequential(
            Linear(28 * 28, 512),
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, 10),
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
    x: Tensor = torch.rand(1, 28, 28, device=device)
    logits: Tensor = model(x)  # Never call model.forward() by hand. Call the model itself.
    predicted_probabilities: Tensor = Softmax(dim=1)(logits)
    print(f"Predicted class: {predicted_probabilities.argmax(1)}")


def model_layers() -> Tuple[Tensor, Tensor]:

    heading("Batch (3 images)")
    input_image: Tensor = torch.rand(3, 28, 28)
    print(input_image.size())

    heading("Flatten")
    flatten: Flatten = Flatten()  # turns each 28x28 image into one row of 784 values, keeping 3 batch instances
    flattened_image: Tensor = flatten(input_image)
    print(flattened_image.size())

    heading("Linear")
    layer1: Linear = Linear(in_features=28 * 28, out_features=20)
    hidden_layer_activations: Tensor = layer1(flattened_image)
    print(hidden_layer_activations.size())

    heading("ReLU")
    print(f"Before ReLU: {hidden_layer_activations}\n\n")
    hidden_layer_activations = ReLU()(hidden_layer_activations)
    print(f"After ReLU: {hidden_layer_activations}")

    return flattened_image, hidden_layer_activations


def sequential_and_softmax() -> None:

    heading("Sequential")
    layers: Sequential = Sequential(
        Flatten(),
        Linear(in_features=28 * 28, out_features=20),
        ReLU(),
        Linear(20, 10)
    )

    input_image: Tensor = torch.rand(3, 28, 28)
    logits: Tensor = layers(input_image)
    print(f"Logits size: {logits.size()}")

    heading("Softmax")
    softmax: Softmax = Softmax(dim=1)
    predicted_probabilities: Tensor = softmax(logits)
    print(f"Probabilities: {predicted_probabilities}")
    print(f"Each row adds up to: {predicted_probabilities.sum(dim=1)}")


def model_parameters(model: NeuralNetwork) -> None:
    heading("Model Parameters")
    print(f"Model structure: {model}\n\n")
    name: str
    parameters: Parameter
    for name, parameters in model.named_parameters():  # lists weights and biases the model will learn for each layer
        print(f"Layer: {name} | Size: {parameters.size()} | Values : {parameters[:2]} \n")


if __name__ == "__main__":
    main()
