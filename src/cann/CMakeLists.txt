add_library(cann OBJECT driver.cpp)

target_include_directories(cann PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cann PUBLIC fmt::fmt ascend_sdk)