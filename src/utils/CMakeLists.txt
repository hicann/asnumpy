add_library(utils OBJECT npu_array.cpp npu_scalar.cpp status_handler.cpp)

target_include_directories(utils PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(utils PUBLIC fmt::fmt ascend_sdk pybind11::pybind11)