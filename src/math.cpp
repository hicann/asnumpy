/******************************************************************************
 * Copyright [2024]-[2025] [HIT1920/asnumpy] Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "math.hpp"
#include "array.hpp"
#include "utils.hpp"


/**
 * @brief Perform element-wise addition of two NPUArrays.
 * 
 * Performs element-wise addition of two NPUArrays using ACL operations.
 * The function creates a new NPUArray with the same shape and data type as the input arrays.
 * 
 * @param a First NPUArray operand
 * @param b Second NPUArray operand
 * @return NPUArray Result of element-wise addition
 */
NPUArray Add(const NPUArray& a, const NPUArray& b) {
    auto shape = a.shape;
    auto dtype = a.dtype;
    auto result = NPUArray(shape, dtype);
    int32_t alpha = 1;
    auto alpha_scalar = aclCreateScalar(&alpha, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnAddGetWorkspaceSize(a.tensorPtr, b.tensorPtr, alpha_scalar, result.tensorPtr, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    error = aclnnAdd(workspaceAddr, workspaceSize, executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}


NPUArray NewAdd(const NPUArray& a, const NPUArray& b) {
    auto shape = a.shape;
    auto dtype = a.dtype;
    auto result = NPUArray(shape, dtype);
    int32_t alpha = 1;
    auto alpha_scalar = aclCreateScalar(&alpha, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnAddGetWorkspaceSize(a.tensorPtr, b.tensorPtr, alpha_scalar, result.tensorPtr, &workspaceSize, &executor);
    Workspace workspace(workspaceSize);
    error = aclnnAdd(workspace.GetWorkspaceAddr(), workspace.GetWorkspaceSize(), executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}


NPUArray Subtract(const NPUArray& a, const NPUArray& b) {
    auto shape = a.shape;
    auto dtype = a.dtype;
    auto result = NPUArray(shape, dtype);
    int32_t alpha = 1;
    auto alpha_scalar = aclCreateScalar(&alpha, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnSubGetWorkspaceSize(a.tensorPtr, b.tensorPtr, alpha_scalar, result.tensorPtr, &workspaceSize, &executor);
    Workspace workspace(workspaceSize);
    error = aclnnSub(workspace.GetWorkspaceAddr(), workspace.GetWorkspaceSize(), executor, nullptr);
    error = aclrtSynchronizeDevice();
    return result;
}

/**
 * @brief Debug function to print NPUArray contents.
 * 
 * Prints the first 10 elements of an NPUArray for debugging purposes.
 * This function copies data from NPU device to host memory for display.
 * 
 * @param a NPUArray to print
 * @throws std::runtime_error If getting tensor data pointer fails or data copy fails
 */
void Print(const NPUArray& a) {
    fmt::println("----------------- print -------------------");
    auto tensorByteSize = a.tensorSize * NPUArray::GetDataTypeSize(a.aclDtype);
    void* rawDataPtr = nullptr;
    auto error = aclGetRawTensorAddr(a.tensorPtr, &rawDataPtr);
    if (error != ACL_SUCCESS || !rawDataPtr) throw std::runtime_error(fmt::format("Failed to get tensor data pointer. error: {}", error));
    py::array result(a.dtype, a.shape);
    py::buffer_info info = result.request();
    if(info.size * info.itemsize != tensorByteSize) throw std::runtime_error("Size mismatch between tensor and NumPy array");
    error = aclrtMemcpy(info.ptr, tensorByteSize, rawDataPtr, tensorByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("Failed to copy tensor data to host. error: {}", error));
    int32_t* arrayData = static_cast<int32_t*>(info.ptr);
    for(int i = 0; i < 10; i++) {
        printf("array data [%i] = %d\n", i, arrayData[i]);
    }
}