# Builds a small C++ program that runs the exported model.onnx file.
#
# The program needs no PyTorch and no Python. CMake downloads everything it
# needs: the ONNX Runtime library, the zlib library, and the FashionMNIST
# test data.
#
# Tested on Windows x64 with Visual Studio 2022 and CMake 3.31.

cmake_minimum_required(VERSION 3.24)
project(onnx_demo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT WIN32)
    message(FATAL_ERROR "This demo downloads the Windows x64 build of ONNX Runtime.")
endif()

include(FetchContent)

# 1. ONNX Runtime. The archive holds no CMakeLists.txt, so FetchContent only
# unpacks it. It does not build it.
set(ONNXRUNTIME_VERSION 1.29.0)
FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip
)
FetchContent_MakeAvailable(onnxruntime)

# 2. zlib. The FashionMNIST files come gzip compressed, and the program reads
# them with gzopen and gzread.
set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    zlib
    URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
)
FetchContent_MakeAvailable(zlib)

# 3. The FashionMNIST test data. Download it once, next to the executable.
set(DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/data)
set(DATA_URL http://fashion-mnist.s3-website.eu-central-1.amazonaws.com)
foreach(NAME t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz)
    if(NOT EXISTS ${DATA_DIR}/${NAME})
        message(STATUS "Downloading ${NAME}")
        file(DOWNLOAD ${DATA_URL}/${NAME} ${DATA_DIR}/${NAME}
             SHOW_PROGRESS STATUS DOWNLOAD_STATUS)
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_CODE)
        if(NOT DOWNLOAD_CODE EQUAL 0)
            file(REMOVE ${DATA_DIR}/${NAME})
            list(GET DOWNLOAD_STATUS 1 DOWNLOAD_MESSAGE)
            message(FATAL_ERROR "Cannot download ${NAME}: ${DOWNLOAD_MESSAGE}")
        endif()
    endif()
endforeach()

add_executable(onnx_demo main.cpp)

# DATA_DIRECTORY gives the program a default path, so it runs with no
# arguments from any working directory.
# Give the program default paths, so it runs with no arguments from any
# working directory.
get_filename_component(DEFAULT_MODEL ${CMAKE_CURRENT_SOURCE_DIR}/../../model.onnx REALPATH)
target_compile_definitions(onnx_demo PRIVATE
    DATA_DIRECTORY="${DATA_DIR}"
    DEFAULT_MODEL_PATH="${DEFAULT_MODEL}")

target_include_directories(onnx_demo PRIVATE
    ${onnxruntime_SOURCE_DIR}/include
    ${zlib_SOURCE_DIR}
    ${zlib_BINARY_DIR})

target_link_libraries(onnx_demo PRIVATE
    ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib
    zlibstatic)

# The program loads onnxruntime.dll at run time. Put the DLL beside the
# executable, so the program starts from any working directory.
add_custom_command(
    TARGET onnx_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll
            $<TARGET_FILE_DIR:onnx_demo>)
