add_library(random OBJECT random.cpp distributions.cpp)

target_include_directories(random PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(random PUBLIC fmt::fmt ascend_sdk pybind11::pybind11)