# C++ inference with ONNX Runtime

This module runs the trained FashionMNIST model from C++. It needs no
PyTorch and no Python.

## What it does

1. Reads `model.onnx`, the file that `000_quickstart.py` exports.
2. Reads one image from the raw FashionMNIST test files.
3. Draws the image as text.
4. Runs the model and prints the class it predicts.

## Before you build

Export the model from Python. Run this from the project root:

    poetry run python src/000_quickstart.py

That writes `model.onnx` into the project root.

## Build

Run these from the project root. CMake downloads ONNX Runtime for you.

    cmake -S src/cpp -B src/cpp/build -A x64
    cmake --build src/cpp/build --config Release

## Run

Run this from the project root, so the relative paths work:

    src\cpp\build\Release\onnx_demo.exe model.onnx data/FashionMNIST/raw 0

The last number is the image index. Change it to try another image.

The program returns 0 when the prediction is correct, and 1 when it is wrong.

## Requirements

- Windows x64
- Visual Studio 2022, or another MSVC toolset
- CMake 3.24 or newer
