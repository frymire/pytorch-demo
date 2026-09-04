"""A demo of automatic differentiation with torch.autograd.

The code follows the official tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

Training changes weights. To know how to change them, PyTorch needs the
gradient of the loss for each weight. torch.autograd works those out for you.
"""

import torch
from torch import Tensor


def main() -> None:
    build_the_graph()
    jacobian_products()


def build_the_graph() -> None:

    heading("Tensors, Functions, and Computational graph")

    x: Tensor = torch.ones(5)  # input tensor
    y: Tensor = torch.zeros(3)  # expected output

    # requires_grad=True tells PyTorch to track every operation on w and b, so it can work out their gradients later.
    w: Tensor = torch.randn(5, 3, requires_grad=True)
    b: Tensor = torch.randn(3, requires_grad=True)

    z: Tensor = torch.matmul(x, w) + b
    loss: Tensor = torch.nn.functional.binary_cross_entropy_with_logits(input=z, target=y)

    # Each result holds a grad_fn: the function that made it. Those links form the graph that autograd walks backward.
    print(f"Gradient function for z = {z.grad_fn}")
    print(f"Gradient function for loss = {loss.grad_fn}")

    heading("Computing Gradients")
    loss.backward()  # walk the graph, filling .grad on every tracked tensor
    print(w.grad)
    print(b.grad)

    heading("Disabling Gradient Tracking")

    z = torch.matmul(x, w) + b
    print(z.requires_grad)  # true

    # Use no_grad when you only want an answer, not training. It runs faster and uses less memory.
    with torch.no_grad():
        z = torch.matmul(x, w) + b
    print(z.requires_grad)  # false

    # detach does the same job for a single tensor.
    z = torch.matmul(x, w) + b
    z_detached: Tensor = z.detach()
    print(z_detached.requires_grad)  # False


def jacobian_products() -> None:

    heading("Tensor Gradients and Jacobian Products")

    input: Tensor = torch.eye(4, 5, requires_grad=True)
    out: Tensor = (input + 1).pow(2).t()

    # backward *adds* to .grad. It does not replace it.
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"First call\n{input.grad}")
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nSecond call\n{input.grad}")  # double the numbers of the first call

    # In a training loop, clear the gradients explicitly at every step.
    input.grad.zero_()
    out.backward(torch.ones_like(out), retain_graph=True)
    print(f"\nCall after zeroing gradients\n{input.grad}")


def heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


if __name__ == "__main__":
    main()
