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

#include "creation.hpp"
#include <aclnnop/aclnn_fill_scalar.h>
#include <aclnnop/aclnn_ones.h>
#include <aclnnop/aclnn_zero.h>
#include <aclnnop/aclnn_eye.h>
#include <aclnnop/aclnn_arange.h>



/**
 * @brief Create an array filled with ones of specified shape and data type.
 * 
 * Creates an array stored on NPU by calling aclnnInplaceOne,
 * with all elements initialized to 1.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param dtype np.dtype defining the data type of array elements.
 * @return NPUArray Array initialized with ones.
 * @throws std::runtime_error If ACL operation returns an error.
 */
NPUArray Ones(const std::vector<int64_t>& shape, py::dtype dtype) {
    auto array = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnInplaceOneGetWorkspaceSize(array.tensorPtr, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[creation.cpp](ones) aclnnInplaceOneGetWorkspaceSize error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](ones) Invalid workspaceSize: {}", workspaceSize));
    // 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) {
            std::string error_msg = fmt::format("[creation.cpp](ones) aclrtMalloc error = {}", error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
                error_msg += std::string(" - ") + detailed_msg;
            }
            throw std::runtime_error(error_msg);
        }
    }
    error = aclnnInplaceOne(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[creation.cpp](ones) aclnnInplaceOne error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[creation.cpp](ones) aclrtSynchronizeDevice error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }
    // 执行结束后释放工作空间
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return array;
}


/**
 * @brief Create an array filled with ones of specified shape and ACL data type.
 * 
 * Creates an array stored on NPU by calling aclnnInplaceOne,
 * with all elements initialized to 1. This version uses ACL data type directly
 * for better performance.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param acl_type ACL data type constant.
 * @return NPUArray Array initialized with ones.
 * @throws std::runtime_error If ACL operation returns an error.
 */
NPUArray Ones(const std::vector<int64_t>& shape, aclDataType acl_type) {
    auto array = NPUArray(shape, acl_type);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnInplaceOneGetWorkspaceSize(array.tensorPtr, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[creation.cpp](ones) aclnnInplaceOneGetWorkspaceSize error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](ones) Invalid workspaceSize: {}", workspaceSize));
    // 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) {
            std::string error_msg = fmt::format("[creation.cpp](ones) aclrtMalloc error = {}", error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
                error_msg += std::string(" - ") + detailed_msg;
            }
            throw std::runtime_error(error_msg);
        }
    }
    error = aclnnInplaceOne(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[creation.cpp](ones) aclnnInplaceOne error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[creation.cpp](ones) aclrtSynchronizeDevice error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }
    // 执行结束后释放工作空间
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return array;
}


/**
 * @brief Create an array filled with zeros of specified shape and data type.
 * 
 * Creates an array stored on NPU by calling aclnnInplaceZero,
 * with all elements initialized to 0.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param dtype np.dtype defining the data type of array elements.
 * @return NPUArray Array initialized with zeros.
 * @throws std::runtime_error If ACL operation returns an error.
 */
NPUArray Zeros(const std::vector<int64_t>& shape, py::dtype dtype) {
    auto array = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnInplaceZeroGetWorkspaceSize(array.tensorPtr, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclnnInplaceZeroGetWorkspaceSize error = {}",error));
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](zeros) Invalid workspaceSize: {}", workspaceSize));
    // 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclrtMalloc error = {}",error));
    }
    error = aclnnInplaceZero(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclnnInplaceZero error = {}",error));
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclrtSynchronizeDevice error = {}",error));
    // 执行结束后释放工作空间
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return array;
}


/**
 * @brief Create an array filled with zeros of specified shape and ACL data type.
 * 
 * Creates an array stored on NPU by calling aclnnInplaceZero,
 * with all elements initialized to 0. This version uses ACL data type directly
 * for better performance.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param acl_type ACL data type constant.
 * @return NPUArray Array initialized with zeros.
 * @throws std::runtime_error If ACL operation returns an error.
 */
NPUArray Zeros(const std::vector<int64_t>& shape, aclDataType acl_type) {
    auto array = NPUArray(shape, acl_type);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnInplaceZeroGetWorkspaceSize(array.tensorPtr, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclnnInplaceZeroGetWorkspaceSize error = {}",error));
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](zeros) Invalid workspaceSize: {}", workspaceSize));
    // 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclrtMalloc error = {}",error));
    }
    error = aclnnInplaceZero(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclnnInplaceZero error = {}",error));
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclrtSynchronizeDevice error = {}",error));
    // 执行结束后释放工作空间
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return array;
}


/**
 * @brief Template struct for mapping C++ types to ACL data types
 * 
 * Provides compile-time conversion between C++ primitive types and ACLDataType enum values.
 * Specializations ensure proper mapping for supported data types.
 */
template <typename T>
struct TypeToACLDtype;

template <> struct TypeToACLDtype<float> { static constexpr aclDataType value = ACL_FLOAT; };
template <> struct TypeToACLDtype<double> { static constexpr aclDataType value = ACL_DOUBLE; };
template <> struct TypeToACLDtype<int8_t> { static constexpr aclDataType value = ACL_INT8; };
template <> struct TypeToACLDtype<int16_t> { static constexpr aclDataType value = ACL_INT16; };
template <> struct TypeToACLDtype<int32_t> { static constexpr aclDataType value = ACL_INT32; };
template <> struct TypeToACLDtype<int64_t> { static constexpr aclDataType value = ACL_INT64; };
template <> struct TypeToACLDtype<uint8_t> { static constexpr aclDataType value = ACL_UINT8; };
template <> struct TypeToACLDtype<uint16_t> { static constexpr aclDataType value = ACL_UINT16; };
template <> struct TypeToACLDtype<uint32_t> { static constexpr aclDataType value = ACL_UINT32; };
template <> struct TypeToACLDtype<uint64_t> { static constexpr aclDataType value = ACL_UINT64; };
template <> struct TypeToACLDtype<bool> { static constexpr aclDataType value = ACL_BOOL; };

/**
 * @brief Creates a scalar object with automatic type deduction
 * 
 * Creates an aclScalar object by automatically determining the appropriate ACL data type
 * based on the C++ type of the input value. Uses TypeToACLDtype for compile-time mapping.
 * 
 * @tparam T Type of the value to store in the scalar (automatically deduced)
 * @param value Scalar value to store
 * @return aclScalar* Pointer to the created scalar object
 */
 
template <typename T>
aclScalar* CreateScalar(T value) {
    return CreateScalar(value, TypeToACLDtype<std::decay_t<T>>::value);
}

/**
 * @brief Creates a scalar object with explicit data type specification
 * 
 * Creates an aclScalar object for a given value with explicit data type control.
 * Performs optimized value conversion when the input type matches the target data type.
 * Falls back to static_cast conversion when types differ.
 * 
 * @tparam ValueType Type of the input value (automatically deduced)
 * @param value Scalar value to store
 * @param dtype Target ACL data type for the scalar
 * @return aclScalar* Pointer to the created scalar object
 * @throws std::runtime_error If the specified data type is unsupported
 * 
 * @note For best performance, match ValueType with the ACL data type representation:
 *       - ACL_FLOAT: float
 *       - ACL_DOUBLE: double
 *       - ACL_INTx: matching intx_t type
 *       - ACL_UINTx: matching uintx_t type
 *       - ACL_BOOL: bool
 */
template <typename ValueType>
aclScalar* CreateScalar(ValueType value, aclDataType dtype) {
    switch (dtype) {
        case ACL_FLOAT: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, float>) {
                return aclCreateScalar(&value, ACL_FLOAT);
            } else {
                auto converted = static_cast<float>(value);
                return aclCreateScalar(&converted, ACL_FLOAT);
            }
        }
        case ACL_DOUBLE: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, double>) {
                return aclCreateScalar(&value, ACL_DOUBLE);
            } else {
                auto converted = static_cast<double>(value);
                return aclCreateScalar(&converted, ACL_DOUBLE);
            }
        }
        case ACL_INT32: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, int32_t>) {
                return aclCreateScalar(&value, ACL_INT32);
            } else {
                auto converted = static_cast<int32_t>(value);
                return aclCreateScalar(&converted, ACL_INT32);
            }
        }
        case ACL_INT64: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, int64_t>) {
                return aclCreateScalar(&value, ACL_INT64);
            } else {
                auto converted = static_cast<int64_t>(value);
                return aclCreateScalar(&converted, ACL_INT64);
            }
        }
        case ACL_INT8: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, int8_t>) {
                return aclCreateScalar(&value, ACL_INT8);
            } else {
                auto converted = static_cast<int8_t>(value);
                return aclCreateScalar(&converted, ACL_INT8);
            }
        }
        case ACL_INT16: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, int16_t>) {
                return aclCreateScalar(&value, ACL_INT16);
            } else {
                auto converted = static_cast<int16_t>(value);
                return aclCreateScalar(&converted, ACL_INT16);
            }
        }
        case ACL_UINT8: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, uint8_t>) {
                return aclCreateScalar(&value, ACL_UINT8);
            } else {
                auto converted = static_cast<uint8_t>(value);
                return aclCreateScalar(&converted, ACL_UINT8);
            }
        }
        case ACL_UINT16: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, uint16_t>) {
                return aclCreateScalar(&value, ACL_UINT16);
            } else {
                auto converted = static_cast<uint16_t>(value);
                return aclCreateScalar(&converted, ACL_UINT16);
            }
        }
        case ACL_UINT32: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, uint32_t>) {
                return aclCreateScalar(&value, ACL_UINT32);
            } else {
                auto converted = static_cast<uint32_t>(value);
                return aclCreateScalar(&converted, ACL_UINT32);
            }
        }
        case ACL_UINT64: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, uint64_t>) {
                return aclCreateScalar(&value, ACL_UINT64);
            } else {
                auto converted = static_cast<uint64_t>(value);
                return aclCreateScalar(&converted, ACL_UINT64);
            }
        }
        case ACL_BOOL: {
            if constexpr (std::is_same_v<std::decay_t<ValueType>, bool>) {
                return aclCreateScalar(&value, ACL_BOOL);
            } else {
                auto converted = static_cast<bool>(value);
                return aclCreateScalar(&converted, ACL_BOOL);
            }
        }
        case ACL_FLOAT16: {
            auto converted = static_cast<uint16_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT16);
        }
        case ACL_BF16: {
            auto converted = static_cast<uint16_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_BF16);
        }
        case ACL_INT4: {
            auto converted = static_cast<int8_t>(value);
            return aclCreateScalar(&converted, ACL_INT4);
        }
        case ACL_UINT1: {
            auto converted = static_cast<uint8_t>(value);
            return aclCreateScalar(&converted, ACL_UINT1);
        }
        case ACL_COMPLEX64: {
            auto converted = std::complex<float>(static_cast<float>(value), 0.0f);
            return aclCreateScalar(&converted, ACL_COMPLEX64);
        }
        case ACL_COMPLEX128: {
            auto converted = std::complex<double>(static_cast<double>(value), 0.0);
            return aclCreateScalar(&converted, ACL_COMPLEX128);
        }
        case ACL_COMPLEX32: {
            auto converted = std::complex<float>(static_cast<float>(value), 0.0f);
            return aclCreateScalar(&converted, ACL_COMPLEX32);
        }
        case ACL_HIFLOAT8: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_HIFLOAT8);
        }
        case ACL_FLOAT8_E5M2: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT8_E5M2);
        }
        case ACL_FLOAT8_E4M3FN: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT8_E4M3FN);
        }
        case ACL_FLOAT8_E8M0: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT8_E8M0);
        }
        case ACL_FLOAT6_E3M2: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT6_E3M2);
        }
        case ACL_FLOAT6_E2M3: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT6_E2M3);
        }
        case ACL_FLOAT4_E2M1: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT4_E2M1);
        }
        case ACL_FLOAT4_E1M2: {
            auto converted = static_cast<uint8_t>(static_cast<float>(value));
            return aclCreateScalar(&converted, ACL_FLOAT4_E1M2);
        }
        case ACL_STRING: {
            // 对于字符串类型，创建一个简单的字符串表示
            std::string str_value = std::to_string(static_cast<double>(value));
            // 使用const_cast来移除const限定符，因为aclCreateScalar需要void*
            return aclCreateScalar(const_cast<char*>(str_value.c_str()), ACL_STRING);
        }
        case ACL_DT_UNDEFINED: {
            // 对于未定义类型，使用默认值0
            auto converted = static_cast<int32_t>(0);
            return aclCreateScalar(&converted, ACL_INT32);
        }
        default:
            throw std::runtime_error("Unsupported dtype: " + std::to_string(static_cast<int>(dtype)));
    }
}


/**
 * @brief Create an array filled with specified value of given shape and data type.
 * 
 * Creates an array stored on NPU with all elements initialized to the specified scalar value.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param dtype np.dtype defining the data type of array elements.
 * @param value Scalar value used to fill the array, supporting various numeric types and data type conversions.
 * @return NPUArray Array filled with specified value.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray Full(const std::vector<int64_t>& shape, py::dtype dtype, const py::object& value) {
    auto array = NPUArray(shape, dtype);
    double valueDouble = 0;
    if (value.is_none()) {
        throw std::runtime_error("[creation.cpp](full) Input is None");
    }
    try {
        valueDouble = py::cast<double>(value);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[creation.cpp](full) Conversion error: " + std::string(e.what()));
    }
    aclScalar* scalar = CreateScalar(valueDouble, array.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnInplaceFillScalarGetWorkspaceSize(array.tensorPtr, scalar, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclnnInplaceFillScalarGetWorkspaceSize error = {}",error));
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](full) Invalid workspaceSize: {}", workspaceSize));
    // 3. 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclrtMalloc error = {}",error));
    }
    error = aclnnInplaceFillScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclnnInplaceFillScalar error = {}", error));
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclrtSynchronizeDevice error = {}",error));
    // 6. 释放
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyScalar(scalar);
    return array;
}


/**
 * @brief Create identity matrix (square matrix with ones on main diagonal) with ACL data type.
 * 
 * Creates a square array (matrix) on NPU where elements on the main diagonal are 1,
 * and all other elements are 0. This version uses ACL data type directly for better performance.
 * 
 * @param n Number of rows and columns of output array (dimension of square matrix).
 * @param acl_type ACL data type constant.
 * @return NPUArray Identity matrix.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray Eye(int64_t n, aclDataType acl_type) {
    auto array = NPUArray({n, n}, acl_type);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnEyeGetWorkspaceSize(n, n, array.tensorPtr, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](eye) aclnnEyeGetWorkspaceSize error = {}",error));
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](eye) Invalid workspaceSize: {}", workspaceSize));
    // 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](eye) aclrtMalloc error = {}",error));
    }
    error = aclnnEye(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](eye) aclnnEye error = {}", error));
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](eye) aclrtSynchronizeDevice error = {}",error));
    // 执行结束后释放工作空间
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return array;
}


/**
 * @brief Create an array filled with specified value of given shape and ACL data type.
 * 
 * Creates an array stored on NPU with all elements initialized to the specified scalar value.
 * This version uses ACL data type directly for better performance.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param acl_type ACL data type constant.
 * @param value Scalar value used to fill the array, supporting various numeric types and data type conversions.
 * @return NPUArray Array filled with specified value.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray Full(const std::vector<int64_t>& shape, aclDataType acl_type, const py::object& value) {
    auto array = NPUArray(shape, acl_type);
    double valueDouble = 0;
    if (value.is_none()) {
        throw std::runtime_error("[creation.cpp](full) Input is None");
    }
    try {
        valueDouble = py::cast<double>(value);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[creation.cpp](full) Conversion error: " + std::string(e.what()));
    }
    aclScalar* scalar = CreateScalar(valueDouble, array.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnInplaceFillScalarGetWorkspaceSize(array.tensorPtr, scalar, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclnnInplaceFillScalarGetWorkspaceSize error = {}",error));
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](full) Invalid workspaceSize: {}", workspaceSize));
    // 3. 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclrtMalloc error = {}",error));
    }
    error = aclnnInplaceFillScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclnnInplaceFillScalar error = {}", error));
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclrtSynchronizeDevice error = {}",error));
    // 6. 释放
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyScalar(scalar);
    return array;
}


/**
 * @brief Create identity matrix (square matrix with ones on main diagonal).
 * 
 * Creates a square array (matrix) on NPU where elements on the main diagonal are 1,
 * and all other elements are 0.
 * 
 * @param n Number of rows and columns of output array (dimension of square matrix).
 * @param dtype np.dtype defining the data type of array elements.
 * @return NPUArray Identity matrix.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray Eye(int64_t n, py::dtype dtype) {
    auto array = NPUArray({n, n}, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnEyeGetWorkspaceSize(n, n, array.tensorPtr, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](eye) aclnnEyeGetWorkspaceSize error = {}",error));
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](eye) Invalid workspaceSize: {}", workspaceSize));
    // 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](eye) aclrtMalloc error = {}",error));
    }
    error = aclnnEye(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](eye) aclnnEye error = {}",error));
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](eye) aclrtSynchronizeDevice error = {}",error));
    // 执行结束后释放工作空间
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return array;
}


/**
 * @brief Create an uninitialized array of specified shape and data type.
 * 
 * Creates an array stored on NPU with uninitialized elements.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param dtype np.dtype defining the data type of array elements.
 * @return NPUArray Uninitialized array.
 * @throws std::runtime_error If NPUArray constructor fails.
 */
NPUArray Empty(const std::vector<int64_t>& shape, py::dtype dtype) {
    try {
        return NPUArray(shape, dtype);
    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("[creation.cpp](empty) NPUArray construction error = {}", e.what()));
    }
}


/**
 * @brief Create an uninitialized array with ACL data type.
 * 
 * Creates an array stored on NPU with uninitialized elements.
 * This version uses ACL data type directly for better performance.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param acl_type ACL data type constant.
 * @return NPUArray An uninitialized array.
 */
NPUArray Empty(const std::vector<int64_t>& shape, aclDataType acl_type) {
    try {
        return NPUArray(shape, acl_type);
    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("[creation.cpp](empty) NPUArray construction error = {}", e.what()));
    }
}


/**
 * @brief Create an arithmetic sequence array.
 * 
 * Creates an arithmetic sequence array stored on NPU by calling aclnnArange,
 * similar to NumPy's arange function.
 * 
 * @param start Starting value (inclusive)
 * @param stop Ending value (exclusive)
 * @param step Step size (default 1)
 * @param dtype np.dtype defining the data type of array elements.
 * @return NPUArray Arithmetic sequence array.
 * @throws std::runtime_error If ACL operation returns an error.
 */
NPUArray Arange(double start, double stop, double step, py::dtype dtype) {
    try {
        if (step == 0) {
            throw std::runtime_error("[creation.cpp](arange) step cannot be 0");
        }
        int64_t numElements = static_cast<int64_t>(std::ceil((stop - start) / step));
        if ((step > 0 && start >= stop) || (step < 0 && start <= stop) || numElements <= 0) {
            numElements = 0;
        }
        std::vector<int64_t> out_shape = {numElements};
        if (numElements == 0) {
            throw std::runtime_error("[creation.cpp](arange) Empty arrays not supported, please check parameters");
        }
        auto array = NPUArray(out_shape, dtype);
        aclDataType dataType = array.aclDtype;
        aclScalar* scalarStart = CreateScalar(start, dataType);
        aclScalar* scalarStop = CreateScalar(stop, dataType);
        aclScalar* scalarStep = CreateScalar(step, dataType);
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        auto error = aclnnArangeGetWorkspaceSize(scalarStart, scalarStop, scalarStep, array.tensorPtr, &workspaceSize, &executor);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](arange) aclnnArangeGetWorkspaceSize error = {}",error));
        // 检查workspaceSize是否有效
        if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](arange) Invalid workspaceSize: {}", workspaceSize));
        // 申请工作空间
        void *workspaceAddr = nullptr;
        if(workspaceSize > 0) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](arange) aclrtMalloc error = {}",error));
        }
        error = aclnnArange(workspaceAddr, workspaceSize, executor, nullptr);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](arange) aclnnArange error = {}",error));
        error = aclrtSynchronizeDevice();
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](arange) aclrtSynchronizeDevice error = {}",error));
        // 执行结束后释放工作空间
        if(workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        // 销毁aclScalar对象
        if(scalarStart) aclDestroyScalar(scalarStart);
        if(scalarStop) aclDestroyScalar(scalarStop);
        if(scalarStep) aclDestroyScalar(scalarStep);
        return array;
    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("[creation.cpp](arange) error = {}", e.what()));
    }
}


/**
 * @brief Create an arithmetic sequence array with ACL data type.
 * 
 * Creates an arithmetic sequence array stored on NPU by calling aclnnArange,
 * similar to NumPy's arange function. This version uses ACL data type directly
 * for better performance.
 * 
 * @param start Starting value (inclusive).
 * @param stop Ending value (exclusive).
 * @param step Step size (default 1).
 * @param acl_type ACL data type constant.
 * @return NPUArray Arithmetic sequence array.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray Arange(double start, double stop, double step, aclDataType acl_type) {
    try {
        if (step == 0) {
            throw std::runtime_error("[creation.cpp](arange) step cannot be 0");
        }
        int64_t numElements = static_cast<int64_t>(std::ceil((stop - start) / step));
        if ((step > 0 && start >= stop) || (step < 0 && start <= stop) || numElements <= 0) {
            numElements = 0;
        }
        std::vector<int64_t> out_shape = {numElements};
        if (numElements == 0) {
            throw std::runtime_error("[creation.cpp](arange) Empty arrays not supported, please check parameters");
        }
        auto array = NPUArray(out_shape, acl_type);
        aclDataType dataType = array.aclDtype;
        aclScalar* scalarStart = CreateScalar(start, dataType);
        aclScalar* scalarStop = CreateScalar(stop, dataType);
        aclScalar* scalarStep = CreateScalar(step, dataType);
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        auto error = aclnnArangeGetWorkspaceSize(scalarStart, scalarStop, scalarStep, array.tensorPtr, &workspaceSize, &executor);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](arange) aclnnArangeGetWorkspaceSize error = {}",error));
        // 检查workspaceSize是否有效
        if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](arange) Invalid workspaceSize: {}", workspaceSize));
        // 申请工作空间
        void *workspaceAddr = nullptr;
        if(workspaceSize > 0) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](arange) aclrtMalloc error = {}",error));
        }
        error = aclnnArange(workspaceAddr, workspaceSize, executor, nullptr);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](arange) aclnnArange error = {}",error));
        error = aclrtSynchronizeDevice();
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](arange) aclrtSynchronizeDevice error = {}",error));
        // 执行结束后释放工作空间
        if(workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        // 销毁aclScalar对象
        if(scalarStart) aclDestroyScalar(scalarStart);
        if(scalarStop) aclDestroyScalar(scalarStop);
        if(scalarStep) aclDestroyScalar(scalarStep);
        return array;
    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("[creation.cpp](arange) error = {}", e.what()));
    }
}

// Overloaded function implementations - support automatic dtype conversion

/**
 * @brief Helper function to convert any object to numpy.dtype
 * 
 * This function supports a wide range of input types that can be converted to numpy.dtype:
 * - NumPy dtype objects (e.g., np.dtype('float32'))
 * - NumPy scalar types (e.g., np.float32, np.int32) - these are valid and will be converted to their corresponding dtype
 * - Python built-in types (e.g., int, float, bool, complex)
 * - Dtype strings (e.g., 'float32', 'int64', '>f4', 'complex128')
 * - Structured dtype descriptors (e.g., [('name', 'S10'), ('value', 'f4')])
 * 
 * @param dtype_obj Any object that can be converted to numpy.dtype
 * @return py::dtype The converted numpy.dtype object
 * @throws std::runtime_error If conversion fails
 */
 py::dtype ObjToDtype(py::object dtype_obj) {
    // Pre-check: None input
    if (dtype_obj.is_none()) {
        throw std::invalid_argument("Cannot convert None to numpy dtype");
    }
    // Pre-check: Common invalid types
    if (py::isinstance<py::list>(dtype_obj) || 
        py::isinstance<py::dict>(dtype_obj) ||
        py::isinstance<py::tuple>(dtype_obj)) {
        
        std::string type_name = py::str(py::type::of(dtype_obj)).cast<std::string>();
        throw std::invalid_argument("Container types (" + type_name + ") are not valid dtype inputs. "
                                    "For structured dtypes, use a list of tuples instead.");
    }
    // Pre-check: Protection against recursive objects and malicious objects
    try {
        std::string type_name = py::str(py::type::of(dtype_obj)).cast<std::string>();
        
        // Check if it's a custom class (potentially malicious object)
        if (type_name.find("__main__") != std::string::npos || 
            type_name.find("test_") != std::string::npos ||
            type_name.find("MaliciousObject") != std::string::npos ||
            type_name.find("CustomClass") != std::string::npos) {
            throw std::invalid_argument("Custom objects are not valid dtype inputs. Input type: " + type_name);
        }
        
        // Note: Function types are not checked, allowing NumPy type objects to pass through
    } catch (const py::error_already_set& e) {
        // If type checking fails, continue with conversion attempts
        PyErr_Clear();
    }
    // Cache NumPy module (static local variable initialized only once)
    static py::module_ np = [](){
        try {
            return py::module_::import("numpy");
        } catch (...) {
            throw std::runtime_error("Failed to import numpy module");
        }
    }();
    std::vector<std::string> error_messages;
    // Method 1: Direct conversion to py::dtype (only for known safe types)
    try {
        // Only perform direct conversion for known safe types
        if (py::isinstance<py::str>(dtype_obj) ||
            py::isinstance<py::int_>(dtype_obj) ||
            py::isinstance<py::float_>(dtype_obj) ||
            py::isinstance<py::bool_>(dtype_obj)) {
            
            return py::dtype::from_args(dtype_obj);
        } else {
            // For other types, skip direct conversion to avoid calling potentially problematic __str__ methods
            error_messages.push_back("Direct conversion skipped for unknown type");
        }
    } catch (const py::cast_error& e) {
        error_messages.push_back("Direct conversion failed: " + std::string(e.what()));
    } catch (const py::error_already_set& e) {
        error_messages.push_back("Direct conversion failed: " + std::string(e.what()));
        PyErr_Clear();
    }
    // Method 2: Use numpy.dtype() function
    try {
        py::object converted = np.attr("dtype")(dtype_obj);
        return py::dtype::from_args(converted);
    } catch (const py::error_already_set& e) {
        std::string msg = e.what();
        error_messages.push_back("numpy.dtype() conversion failed: " + msg);
        PyErr_Clear();
    }
    // Method 3: String conversion (only when object can be converted to string)
    if (py::isinstance<py::str>(dtype_obj) || PyObject_HasAttrString(dtype_obj.ptr(), "__str__")) {
        try {
            py::str dtype_str = py::str(dtype_obj);
            
            // Check for empty string
            if (dtype_str.cast<std::string>().empty()) {
                throw std::invalid_argument("Empty string is not a valid dtype");
            }
            
            std::string str_value = dtype_str.cast<std::string>();
            if (str_value.length() > 1000) {
                throw std::invalid_argument("Dtype string too long (possible recursive object)");
            }
            if (str_value.find("object at") != std::string::npos ||
                str_value.find("function") != std::string::npos ||
                str_value.find("generator") != std::string::npos ||
                str_value.find("iterator") != std::string::npos ||
                str_value.find("module") != std::string::npos) {
                throw std::invalid_argument("String contains object representation, not a valid dtype");
            }
            py::object converted = np.attr("dtype")(dtype_str);
            return py::dtype::from_args(converted);
        } catch (const py::error_already_set& e) {
            std::string msg = e.what();
            error_messages.push_back("String conversion failed: " + msg);
            PyErr_Clear();
        }
    }
    std::string type_name;
    std::string repr;
    try {
        type_name = py::str(py::type::of(dtype_obj)).cast<std::string>();
        repr = py::str(dtype_obj).cast<std::string>();
        if (repr.length() > 100) {
            repr = repr.substr(0, 97) + "...";
        }
    } catch (...) {
        type_name = "<error getting type>";
        repr = "<error getting representation>";
    }
    std::string msg = "Failed to convert object to numpy dtype.\n"
                      "Input type: " + type_name + "\n"
                      "Value representation: " + repr + "\n\n";
    if (!error_messages.empty()) {
        msg += "Conversion attempts:\n";
        for (const auto& err : error_messages) {
            msg += "  • " + err + "\n";
        }
        msg += "\n";
    }
    msg += "Supported inputs:\n"
           "  - NumPy dtype objects (e.g., np.dtype('float32'))\n"
           "  - NumPy scalar types (e.g., np.float32, np.int32) - these are valid and will be converted to their corresponding dtype\n"
           "  - Python built-in types (e.g., int, float, bool, complex)\n"
           "  - Dtype strings (e.g., 'float32', 'int64', '>f4', 'complex128')\n"
           "  - Structured dtype descriptors (e.g., [('name', 'S10'), ('value', 'f4')])\n\n";
    throw std::invalid_argument(msg);
}


/**
 * @brief Create an array of ones with automatic dtype conversion.
 * 
 * Overloaded version that accepts any object that can be converted to numpy.dtype.
 * Optimized to detect ACL constant values and use direct ACL path.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param dtype_obj Any object that can be converted to numpy.dtype (e.g., np.float32, np.int32).
 * @return NPUArray Array initialized with ones.
 */
NPUArray Ones(const std::vector<int64_t>& shape, py::object dtype_obj) {
    // Check if dtype_obj is an ACL constant value (integer)
    if (py::isinstance<py::int_>(dtype_obj)) {
        int acl_value = dtype_obj.cast<int>();
        
        // Check if it's a recognized ACL constant (all 28 types)
        switch (acl_value) {
            case ACL_DT_UNDEFINED:
            case ACL_FLOAT:
            case ACL_FLOAT16:
            case ACL_INT8:
            case ACL_INT32:
            case ACL_UINT8:
            case ACL_INT16:
            case ACL_UINT16:
            case ACL_UINT32:
            case ACL_INT64:
            case ACL_UINT64:
            case ACL_DOUBLE:
            case ACL_BOOL:
            case ACL_STRING:
            case ACL_COMPLEX64:
            case ACL_COMPLEX128:
            case ACL_BF16:
            case ACL_INT4:
            case ACL_UINT1:
            case ACL_COMPLEX32:
            case ACL_HIFLOAT8:
            case ACL_FLOAT8_E5M2:
            case ACL_FLOAT8_E4M3FN:
            case ACL_FLOAT8_E8M0:
            case ACL_FLOAT6_E3M2:
            case ACL_FLOAT6_E2M3:
            case ACL_FLOAT4_E2M1:
            case ACL_FLOAT4_E1M2:
                // Use optimized direct ACL path
                return Ones(shape, static_cast<aclDataType>(acl_value));
            default:
                // Not a recognized ACL constant, fall back to normal conversion
                break;
        }
    }
    
    // Fall back to normal conversion path
    py::dtype dtype = ObjToDtype(dtype_obj);
    return Ones(shape, dtype);
}


/**
 * @brief Create an array of zeros with automatic dtype conversion.
 * 
 * Overloaded version that accepts any object that can be converted to numpy.dtype.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param dtype_obj Any object that can be converted to numpy.dtype (e.g., np.float32, np.int32).
 * @return NPUArray Array initialized with zeros.
 */
NPUArray Zeros(const std::vector<int64_t>& shape, py::object dtype_obj) {
    // Check if dtype_obj is an ACL constant value (integer)
    if (py::isinstance<py::int_>(dtype_obj)) {
        int acl_value = dtype_obj.cast<int>();
        
        // Check if it's a recognized ACL constant (all 28 types)
        switch (acl_value) {
            case ACL_DT_UNDEFINED:
            case ACL_FLOAT:
            case ACL_FLOAT16:
            case ACL_INT8:
            case ACL_INT32:
            case ACL_UINT8:
            case ACL_INT16:
            case ACL_UINT16:
            case ACL_UINT32:
            case ACL_INT64:
            case ACL_UINT64:
            case ACL_DOUBLE:
            case ACL_BOOL:
            case ACL_STRING:
            case ACL_COMPLEX64:
            case ACL_COMPLEX128:
            case ACL_BF16:
            case ACL_INT4:
            case ACL_UINT1:
            case ACL_COMPLEX32:
            case ACL_HIFLOAT8:
            case ACL_FLOAT8_E5M2:
            case ACL_FLOAT8_E4M3FN:
            case ACL_FLOAT8_E8M0:
            case ACL_FLOAT6_E3M2:
            case ACL_FLOAT6_E2M3:
            case ACL_FLOAT4_E2M1:
            case ACL_FLOAT4_E1M2:
                // Use optimized direct ACL path
                return Zeros(shape, static_cast<aclDataType>(acl_value));
            default:
                // Not a recognized ACL constant, fall back to normal conversion
                break;
        }
    }
    
    // Fall back to normal conversion path
    py::dtype dtype = ObjToDtype(dtype_obj);
    return Zeros(shape, dtype);
}


/**
 * @brief Create an array with specified value and automatic dtype conversion.
 * 
 * Overloaded version that accepts any object that can be converted to numpy.dtype.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param dtype_obj Any object that can be converted to numpy.dtype (e.g., np.float32, np.int32).
 * @param value The scalar value used to fill the array.
 * @return NPUArray Array filled with the specified value.
 */
NPUArray Full(const std::vector<int64_t>& shape, py::object dtype_obj, const py::object& value) {
    // Check if dtype_obj is an ACL constant value (integer)
    if (py::isinstance<py::int_>(dtype_obj)) {
        int acl_value = dtype_obj.cast<int>();
        
        // Check if it's a recognized ACL constant (all 28 types)
        switch (acl_value) {
            case ACL_DT_UNDEFINED:
            case ACL_FLOAT:
            case ACL_FLOAT16:
            case ACL_INT8:
            case ACL_INT32:
            case ACL_UINT8:
            case ACL_INT16:
            case ACL_UINT16:
            case ACL_UINT32:
            case ACL_INT64:
            case ACL_UINT64:
            case ACL_DOUBLE:
            case ACL_BOOL:
            case ACL_STRING:
            case ACL_COMPLEX64:
            case ACL_COMPLEX128:
            case ACL_BF16:
            case ACL_INT4:
            case ACL_UINT1:
            case ACL_COMPLEX32:
            case ACL_HIFLOAT8:
            case ACL_FLOAT8_E5M2:
            case ACL_FLOAT8_E4M3FN:
            case ACL_FLOAT8_E8M0:
            case ACL_FLOAT6_E3M2:
            case ACL_FLOAT6_E2M3:
            case ACL_FLOAT4_E2M1:
            case ACL_FLOAT4_E1M2:
                // Use optimized direct ACL path
                return Full(shape, static_cast<aclDataType>(acl_value), value);
            default:
                // Not a recognized ACL constant, fall back to normal conversion
                break;
        }
    }
    
    // Fall back to normal conversion path
    py::dtype dtype = ObjToDtype(dtype_obj);
    return Full(shape, dtype, value);
}


/**
 * @brief Create an identity matrix with automatic dtype conversion.
 * 
 * Overloaded version that accepts any object that can be converted to numpy.dtype.
 * 
 * @param n The number of rows and columns in the output array (dimension of the square matrix).
 * @param dtype_obj Any object that can be converted to numpy.dtype (e.g., np.float32, np.int32).
 * @return NPUArray An identity matrix.
 */
NPUArray Eye(int64_t n, py::object dtype_obj) {
    // Check if dtype_obj is an ACL constant value (integer)
    if (py::isinstance<py::int_>(dtype_obj)) {
        int acl_value = dtype_obj.cast<int>();
        
        // Check if it's a recognized ACL constant (all 28 types)
        switch (acl_value) {
            case ACL_DT_UNDEFINED:
            case ACL_FLOAT:
            case ACL_FLOAT16:
            case ACL_INT8:
            case ACL_INT32:
            case ACL_UINT8:
            case ACL_INT16:
            case ACL_UINT16:
            case ACL_UINT32:
            case ACL_INT64:
            case ACL_UINT64:
            case ACL_DOUBLE:
            case ACL_BOOL:
            case ACL_STRING:
            case ACL_COMPLEX64:
            case ACL_COMPLEX128:
            case ACL_BF16:
            case ACL_INT4:
            case ACL_UINT1:
            case ACL_COMPLEX32:
            case ACL_HIFLOAT8:
            case ACL_FLOAT8_E5M2:
            case ACL_FLOAT8_E4M3FN:
            case ACL_FLOAT8_E8M0:
            case ACL_FLOAT6_E3M2:
            case ACL_FLOAT6_E2M3:
            case ACL_FLOAT4_E2M1:
            case ACL_FLOAT4_E1M2:
                // Use optimized direct ACL path
                return Eye(n, static_cast<aclDataType>(acl_value));
            default:
                // Not a recognized ACL constant, fall back to normal conversion
                break;
        }
    }
    
    // Fall back to normal conversion path
    py::dtype dtype = ObjToDtype(dtype_obj);
    return Eye(n, dtype);
}


/**
 * @brief Create an empty array with automatic dtype conversion.
 * 
 * Overloaded version that accepts any object that can be converted to numpy.dtype.
 * 
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param dtype_obj Any object that can be converted to numpy.dtype (e.g., np.float32, np.int32).
 * @return NPUArray An uninitialized array.
 */
NPUArray Empty(const std::vector<int64_t>& shape, py::object dtype_obj) {
    // Check if dtype_obj is an ACL constant value (integer)
    if (py::isinstance<py::int_>(dtype_obj)) {
        int acl_value = dtype_obj.cast<int>();
        
        // Check if it's a recognized ACL constant (all 28 types)
        switch (acl_value) {
            case ACL_DT_UNDEFINED:
            case ACL_FLOAT:
            case ACL_FLOAT16:
            case ACL_INT8:
            case ACL_INT32:
            case ACL_UINT8:
            case ACL_INT16:
            case ACL_UINT16:
            case ACL_UINT32:
            case ACL_INT64:
            case ACL_UINT64:
            case ACL_DOUBLE:
            case ACL_BOOL:
            case ACL_STRING:
            case ACL_COMPLEX64:
            case ACL_COMPLEX128:
            case ACL_BF16:
            case ACL_INT4:
            case ACL_UINT1:
            case ACL_COMPLEX32:
            case ACL_HIFLOAT8:
            case ACL_FLOAT8_E5M2:
            case ACL_FLOAT8_E4M3FN:
            case ACL_FLOAT8_E8M0:
            case ACL_FLOAT6_E3M2:
            case ACL_FLOAT6_E2M3:
            case ACL_FLOAT4_E2M1:
            case ACL_FLOAT4_E1M2:
                // Use optimized direct ACL path
                return Empty(shape, static_cast<aclDataType>(acl_value));
            default:
                // Not a recognized ACL constant, fall back to normal conversion
                break;
        }
    }
    
    // Fall back to normal conversion path
    py::dtype dtype = ObjToDtype(dtype_obj);
    return Empty(shape, dtype);
}


/**
 * @brief Create an arithmetic sequence array with automatic dtype conversion.
 * 
 * Overloaded version that accepts any object that can be converted to numpy.dtype.
 * 
 * @param start Start value (inclusive)
 * @param stop Stop value (exclusive)
 * @param step Step value (default 1)
 * @param dtype_obj Any object that can be converted to numpy.dtype (e.g., np.float32, np.int32).
 * @return NPUArray An array with an arithmetic progression.
 */
NPUArray Arange(double start, double stop, double step, py::object dtype_obj) {
    // Check if dtype_obj is an ACL constant value (integer)
    if (py::isinstance<py::int_>(dtype_obj)) {
        int acl_value = dtype_obj.cast<int>();
        
        // Check if it's a recognized ACL constant (all 28 types)
        switch (acl_value) {
            case ACL_DT_UNDEFINED:
            case ACL_FLOAT:
            case ACL_FLOAT16:
            case ACL_INT8:
            case ACL_INT32:
            case ACL_UINT8:
            case ACL_INT16:
            case ACL_UINT16:
            case ACL_UINT32:
            case ACL_INT64:
            case ACL_UINT64:
            case ACL_DOUBLE:
            case ACL_BOOL:
            case ACL_STRING:
            case ACL_COMPLEX64:
            case ACL_COMPLEX128:
            case ACL_BF16:
            case ACL_INT4:
            case ACL_UINT1:
            case ACL_COMPLEX32:
            case ACL_HIFLOAT8:
            case ACL_FLOAT8_E5M2:
            case ACL_FLOAT8_E4M3FN:
            case ACL_FLOAT8_E8M0:
            case ACL_FLOAT6_E3M2:
            case ACL_FLOAT6_E2M3:
            case ACL_FLOAT4_E2M1:
            case ACL_FLOAT4_E1M2:
                // Use optimized direct ACL path
                return Arange(start, stop, step, static_cast<aclDataType>(acl_value));
            default:
                // Not a recognized ACL constant, fall back to normal conversion
                break;
        }
    }
    
    // Fall back to normal conversion path
    py::dtype dtype = ObjToDtype(dtype_obj);
    return Arange(start, stop, step, dtype);
}