add_library(linalg OBJECT norms.cpp product.cpp decompositions.cpp solving_inverting.cpp)

target_include_directories(linalg PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(linalg PUBLIC fmt::fmt pybind11::pybind11)


