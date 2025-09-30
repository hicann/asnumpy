add_library(random OBJECT random.cpp)

target_include_directories(random PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(random PUBLIC fmt::fmt ascend_sdk)