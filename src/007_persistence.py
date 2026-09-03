"""A demo of how to save and load a model.

The code follows the official tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

WARNING: the first run downloads the trained VGG16 weights, about 528 MB.
PyTorch keeps them in its cache, so later runs are fast.
"""

import os
import tempfile
from typing import Any, Dict

import torch
import torchvision.models as models
from torch import Tensor
from torch.nn import Module


def heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def save_and_load_weights(weights_file: str) -> None:

    heading("Saving and Loading Model Weights")

    # VGG16 is a large image model. IMAGENET1K_V1 asks for weights that are
    # already trained on the ImageNet data.
    model: Module = models.vgg16(weights="IMAGENET1K_V1")

    # A state_dict is a plain dictionary of every learned tensor, keyed by
    # layer name. Save this, not the model object.
    state_dict: Dict[str, Any] = model.state_dict()
    print(f"Entries in the state_dict: {len(state_dict)}")

    torch.save(state_dict, weights_file)
    size_mb: float = os.path.getsize(weights_file) / (1024 * 1024)
    print(f"Saved {weights_file} ({size_mb:.1f} MB)")

    # Build the same shape of model, then pour the weights into it. Without
    # weights= the model starts untrained.
    reloaded: Module = models.vgg16()
    reloaded.load_state_dict(torch.load(weights_file, weights_only=True))

    # eval() switches off dropout and batch-norm updates. Always call it
    # before you make predictions, or your answers will vary run to run.
    reloaded.eval()
    print("Reloaded the weights and called eval()")

    # Prove the two models agree.
    model.eval()
    x: Tensor = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        same: bool = torch.allclose(model(x), reloaded(x))
    print(f"Both models give the same answer: {same}")


def save_and_load_whole_model(model_file: str) -> None:

    heading("Saving and Loading Models with Shapes")

    model: Module = models.vgg16()

    # This saves the object, not only the weights. You then need no class
    # definition to load it back.
    torch.save(model, model_file)
    size_mb: float = os.path.getsize(model_file) / (1024 * 1024)
    print(f"Saved {model_file} ({size_mb:.1f} MB)")

    # weights_only=False lets torch.load rebuild the Python object. Use it
    # only on a file you trust, because it can run code from that file.
    reloaded: Module = torch.load(model_file, weights_only=False)
    print(f"Reloaded a {type(reloaded).__name__} with no class definition given")


def main() -> None:

    # Write into a temporary folder, so the demo leaves no large files in
    # the project. The tutorial writes them into the working directory.
    with tempfile.TemporaryDirectory() as work_dir:

        # The tutorial uses the .pth suffix. Prefer .pt in your own code:
        # Python already uses .pth for path configuration files.
        save_and_load_weights(os.path.join(work_dir, "model_weights.pth"))
        save_and_load_whole_model(os.path.join(work_dir, "model.pth"))

    heading("Which one to use")
    print("Save the state_dict. It survives changes to your code.")
    print("A whole-model file stores a link to your class, so a rename breaks it.")


if __name__ == "__main__":
    main()
