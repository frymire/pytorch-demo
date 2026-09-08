# Builds a small C++ program that runs the exported model.onnx file.
#
# The program needs no PyTorch and no Python. It uses ONNX Runtime, which
# CMake downloads for you as a prebuilt library.
#
# Tested on Windows x64 with Visual Studio 2022 and CMake 3.31.

cmake_minimum_required(VERSION 3.24)
project(onnx_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT WIN32)
    message(FATAL_ERROR "This demo downloads the Windows x64 build of ONNX Runtime.")
endif()

# Download and unpack the prebuilt ONNX Runtime. The archive holds no
# CMakeLists.txt, so FetchContent only unpacks it. It does not build it.
include(FetchContent)
set(ONNXRUNTIME_VERSION 1.29.0)
FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip
)
FetchContent_MakeAvailable(onnxruntime)

add_executable(onnx_demo main.cpp)

target_include_directories(onnx_demo PRIVATE ${onnxruntime_SOURCE_DIR}/include)
target_link_libraries(onnx_demo PRIVATE ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib)

# The program loads onnxruntime.dll at run time. Put the DLL beside the
# executable, so the program starts from any working directory.
add_custom_command(
    TARGET onnx_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll
            $<TARGET_FILE_DIR:onnx_demo>)
