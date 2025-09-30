add_library(array OBJECT basic.cpp)

target_link_libraries(array PUBLIC utils fmt::fmt ascend_sdk pybind11::pybind11)

