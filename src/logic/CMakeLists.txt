add_library(logic OBJECT random.cpp)

target_include_directories(logic PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(logic PUBLIC fmt::fmt ascend_sdk)