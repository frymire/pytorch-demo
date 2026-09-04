"""A demo of the PyTorch tensor basics.

The code follows the official tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
"""

from typing import Any, List, Tuple

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def initialize_a_tensor() -> None:

    heading("Directly from data")
    data: List[List[int]] = [[1, 2], [3, 4]]
    x_data: Tensor = torch.tensor(data)
    print(x_data)

    heading("From a NumPy array")
    np_array: ndarray = np.array(data)
    x_np: Tensor = torch.from_numpy(np_array)
    print(x_np)

    heading("From another tensor")
    # ones_like keeps the shape and the dtype of x_data.
    x_ones: Tensor = torch.ones_like(x_data)
    print(f"Ones Tensor: \n {x_ones} \n")

    # rand_like keeps the shape, but dtype overrides the dtype of x_data.
    x_rand: Tensor = torch.rand_like(x_data, dtype=torch.float)
    print(f"Random Tensor: \n {x_rand} \n")

    heading("With random or constant values")
    shape: Tuple[int, int] = (2, 3)
    rand_tensor: Tensor = torch.rand(shape)
    ones_tensor: Tensor = torch.ones(shape)
    zeros_tensor: Tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")


def attributes_of_a_tensor() -> Tensor:

    heading("Attributes of a Tensor")
    tensor: Tensor = torch.rand(3, 4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    return tensor


def operations_on_tensors(tensor: Tensor) -> None:

    heading("Move the tensor to the accelerator")
    # Move the tensor to the current accelerator if there is one.
    if torch.accelerator.is_available():
        tensor = tensor.to(torch.accelerator.current_accelerator())
    print(f"Device tensor is stored on: {tensor.device}")

    heading("Standard numpy-like indexing and slicing")
    tensor = torch.rand(4, 4)
    print(tensor)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[..., -1]}")
    tensor[:, 1] = 0

    heading("Joining tensors")
    t1: Tensor = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)

    heading("Arithmetic operations")
    # Matrix multiplication. y1, y2 and y3 all hold the same value.
    # tensor.T returns the transpose of the tensor.
    y1: Tensor = tensor @ tensor.T
    y2: Tensor = tensor.matmul(tensor.T)

    y3: Tensor = torch.rand_like(y1)
    torch.matmul(tensor, tensor.T, out=y3)
    print(f"y1: \n {y1} \n y2: \n {y2} \n y3: \n {y3}")

    # Element-wise product. z1, z2 and z3 all hold the same value.
    z1: Tensor = tensor * tensor
    z2: Tensor = tensor.mul(tensor)

    z3: Tensor = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)
    print(f"z1: \n {z1} \n z2: \n {z2} \n z3: \n {z3}")

    heading("Single-element tensors")
    agg: Tensor = tensor.sum()
    agg_item: Any = agg.item()
    print(agg_item, type(agg_item))

    heading("In-place operations")
    print(f"{tensor} \n")
    tensor.add_(5)
    print(tensor)


def bridge_with_numpy() -> None:

    heading("Tensor to NumPy array")
    t: Tensor = torch.ones(5)
    print(f"t: {t}")
    n: ndarray = t.numpy()
    print(f"n: {n}")

    # The tensor and the array share the same memory. A change to one of them
    # shows up in the other one.
    t.add_(1)
    print(f"t: {t}")
    print(f"n: {n}")

    heading("NumPy array to Tensor")
    n2: ndarray = np.ones(5)
    t2: Tensor = torch.from_numpy(n2)

    # The share works in this direction too.
    np.add(n2, 1, out=n2)
    print(f"t: {t2}")
    print(f"n: {n2}")


def main() -> None:
    initialize_a_tensor()
    tensor: Tensor = attributes_of_a_tensor()
    operations_on_tensors(tensor)
    bridge_with_numpy()


if __name__ == "__main__":
    main()
