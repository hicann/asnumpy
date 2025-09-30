add_library(math OBJECT miscellaneous.cpp)
# 不编译所有的math.cpp可以正常编译运行
# add_library(math OBJECT arithmetic_operations.cpp exponents_and_logarithms.cpp floating_point_routines.cpp
#             handling_complex_numbers.cpp hyperbolic_functions.cpp other_special_functions.cpp
#             rational_routines.cpp rounding.cpp sums_products_differences.cpp )

target_include_directories(math PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(math PUBLIC fmt::fmt ascend_sdk pybind11::pybind11)