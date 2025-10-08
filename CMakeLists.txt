cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    fmt
    URL file://${CMAKE_SOURCE_DIR}/third_party/fmt/fmt-12.0.0.tar.gz
)

FetchContent_Declare(
    pybind11
    URL file://${CMAKE_SOURCE_DIR}/third_party/pybind11/pybind11-3.0.1.tar.gz
)

FetchContent_MakeAvailable(fmt pybind11)

project(ascend_numpy)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

include_directories(/usr/local/Ascend/ascend-toolkit/latest/include)
link_directories(/usr/local/Ascend/ascend-toolkit/latest/lib64)

include_directories(${Python3_NumPy_INCLUDE_DIRS})

include_directories(include)
add_subdirectory(src)
add_subdirectory(python)
