# C++ inference with ONNX Runtime

This module runs the trained FashionMNIST model from C++. It needs no
PyTorch and no Python.

## What it does

1. Reads `model.onnx`, the file that `000_quickstart.py` exports.
2. Reads one image from the FashionMNIST test data.
3. Draws the image as text.
4. Runs the model and prints the class it predicts.

## What CMake downloads for you

You need no setup step. CMake gets all three of these:

| Item | Source |
|---|---|
| ONNX Runtime 1.29.0 | The Microsoft release page |
| zlib 1.3.1 | The zlib release page |
| FashionMNIST test data | The FashionMNIST data site |

The data files stay gzip compressed. zlib reads them in place.

`model.onnx` is in the repository, so you do not have to run Python first.

## Build

Run these from the project root.

    cmake -S src/cpp -B src/cpp/build -A x64
    cmake --build src/cpp/build --config Release

## Run

The program has built-in default paths, so it runs with no arguments from
any working directory.

    src\cpp\build\Release\onnx_demo.exe

To pick another image, give all three arguments. The last one is the image
index, from 0 to 9999.

    src\cpp\build\Release\onnx_demo.exe model.onnx src/cpp/build/data 42

The program returns 0 when the prediction is correct, and 1 when it is wrong.

## Requirements

- Windows x64
- Visual Studio 2022, or another MSVC toolset
- CMake 3.24 or newer
