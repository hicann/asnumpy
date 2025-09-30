add_library(dtypes OBJECT 
    reg.cpp
    np_import.cpp
)

target_include_directories(dtypes PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(dtypes PUBLIC 
    fmt::fmt
    ${Python3_LIBRARIES}
)

# 添加必要的编译定义
target_compile_definitions(dtypes PRIVATE
    NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
)

# 确保 C++17 标准
set_target_properties(dtypes PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)