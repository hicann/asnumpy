# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

add_library(asnumpy INTERFACE)
add_library(ascend_sdk INTERFACE)

target_include_directories(asnumpy INTERFACE ${CMAKE_SOURCE_DIR}/include)

target_include_directories(ascend_sdk INTERFACE /usr/local/Ascend/ascend-toolkit/latest/include)
target_link_directories(ascend_sdk INTERFACE /usr/local/Ascend/ascend-toolkit/latest/lib64)
target_link_libraries(ascend_sdk INTERFACE ascendcl runtime nnopbase opapi)

if (DEFINED ENV{ASCEND_SIM_MODE} AND "$ENV{ASCEND_SIM_MODE}" STREQUAL "1")
	message(STATUS "ASCEND_SIM_MODE is set to 1. Linking libruntime_camodel.so for simulator.")
	set(ASCEND_SIM_LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/simulator/Ascend910B1/lib")
	target_link_directories(ascend_sdk INTERFACE ${ASCEND_SIM_LIB_PATH})
	target_link_libraries(ascend_sdk INTERFACE runtime_camodel)
else()
	message(STATUS "ASCEND_SIM_MODE is not set to 1. Skipping linking libruntime_camodel.so.")
endif()

add_subdirectory(array)
add_subdirectory(cann)
# add_subdirectory(dtypes)
add_subdirectory(linalg)
add_subdirectory(random)
add_subdirectory(math)
add_subdirectory(logic)
add_subdirectory(utils)

target_link_libraries(asnumpy INTERFACE array cann dtypes linalg random math utils)
