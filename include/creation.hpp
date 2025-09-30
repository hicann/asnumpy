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

#pragma once

#include <stdexcept>
#include <fmt/base.h>
#include <fmt/format.h>
#include "array.hpp"


/**
 * @brief Create an array of ones with the specified shape and data type.
 *
 * This function declaration is used to create an array stored on the NPU, with all elements initialized to 1.
 *
 * @param shape A vector containing the dimensions of the array, defining its shape.
 * @param dtype np.dtype, defining the data type of the elements in the array.
 * @return NPUArray An array initialized to ones.
 */
NPUArray Ones(const std::vector<int64_t>& shape, py::dtype dtype);

/**
 * @brief Create an array of ones with the specified shape and ACL data type.
 *
 * This function declaration is used to create an array stored on the NPU, with all elements initialized to 1.
 * This version uses ACL data type directly for better performance.
 *
 * @param shape A vector containing the dimensions of the array, defining its shape.
 * @param acl_type ACL data type constant.
 * @return NPUArray An array initialized to ones.
 */
NPUArray Ones(const std::vector<int64_t>& shape, aclDataType acl_type);

/**
 * @brief Create an array of zeros with the specified shape and data type.
 *
 * This function declaration is used to create an array stored on the NPU, with all elements initialized to 0.
 *
 * @param shape A vector containing the dimensions of the array, defining its shape.
 * @param dtype np.dtype, defining the data type of the elements in the array.
 * @return NPUArray An array initialized to zeros.
 */
NPUArray Zeros(const std::vector<int64_t>& shape, py::dtype dtype);

/**
 * @brief Create an array of zeros with the specified shape and ACL data type.
 *
 * This function declaration is used to create an array stored on the NPU, with all elements initialized to 0.
 * This version uses ACL data type directly for better performance.
 *
 * @param shape A vector containing the dimensions of the array, defining its shape.
 * @param acl_type ACL data type constant.
 * @return NPUArray An array initialized to zeros.
 */
NPUArray Zeros(const std::vector<int64_t>& shape, aclDataType acl_type);

/**
 * @brief Create an array with the specified shape, data type, and fill value.
 *
 * This function declaration is used to create an array stored on the NPU, 
 * with all elements initialized to the specified scalar value.
 *
 * @param shape A vector containing the dimensions of the array, defining its shape.
 * @param dtype np.dtype, defining the data type of the elements in the array.
 * @param value The scalar value used to fill the array, supporting various numeric types and data type conversions.
 * @return NPUArray An array filled with the specified value.
 */
NPUArray Full(const std::vector<int64_t>& shape, py::dtype dtype, const py::object& value);

/**
 * @brief Create an array with the specified shape, ACL data type, and fill value.
 *
 * This function declaration is used to create an array stored on the NPU, 
 * with all elements initialized to the specified scalar value.
 * This version uses ACL data type directly for better performance.
 *
 * @param shape A vector containing the dimensions of the array, defining its shape.
 * @param acl_type ACL data type constant.
 * @param value The scalar value used to fill the array, supporting various numeric types and data type conversions.
 * @return NPUArray An array filled with the specified value.
 */
NPUArray Full(const std::vector<int64_t>& shape, aclDataType acl_type, const py::object& value);


/**
 * @brief Create an identity matrix (square matrix with ones on the main diagonal).
 *
 * This function declaration is used to create a square array on the NPU, 
 * with ones on the main diagonal and zeros elsewhere.
 *
 * @param n The number of rows and columns in the output array (dimension of the square matrix).
 * @param dtype np.dtype, defining the data type of the elements in the array.
 * @return NPUArray An identity matrix.
 */
NPUArray Eye(int64_t n, py::dtype dtype);

/**
 * @brief Create an identity matrix with the specified dimension and ACL data type.
 *
 * This function declaration is used to create a square array on the NPU, 
 * with ones on the main diagonal and zeros elsewhere.
 * This version uses ACL data type directly for better performance.
 *
 * @param n The number of rows and columns in the output array (dimension of the square matrix).
 * @param acl_type ACL data type constant.
 * @return NPUArray An identity matrix.
 */
NPUArray Eye(int64_t n, aclDataType acl_type);

/**
 * @brief Create an empty array with the specified shape and data type.
 *
 * This function declaration is used to create an array stored on the NPU, with elements not initialized.
 *
 * @param shape A vector containing the dimensions of the array, defining its shape.
 * @param dtype np.dtype, defining the data type of the elements in the array.
 * @return NPUArray An uninitialized array.
 */
NPUArray Empty(const std::vector<int64_t>& shape, py::dtype dtype);

/**
 * @brief Create an empty array with the specified shape and ACL data type.
 *
 * This function declaration is used to create an array stored on the NPU, with elements not initialized.
 * This version uses ACL data type directly for better performance.
 *
 * @param shape A vector containing the dimensions of the array, defining its shape.
 * @param acl_type ACL data type constant.
 * @return NPUArray An uninitialized array.
 */
NPUArray Empty(const std::vector<int64_t>& shape, aclDataType acl_type);

/**
 * @brief Create an array with an arithmetic progression.
 *
 * This function declaration is used to create an array stored on the NPU 
 * with an arithmetic progression, similar to NumPy's arange function.
 *
 * @param start Start value (inclusive)
 * @param stop Stop value (exclusive)
 * @param step Step value (default 1)
 * @param dtype np.dtype, defining the data type of the elements in the array.
 * @return NPUArray An array with an arithmetic progression.
 */
NPUArray Arange(double start, double stop, double step, py::dtype dtype);

/**
 * @brief Create an array with an arithmetic progression using ACL data type.
 *
 * This function declaration is used to create an array stored on the NPU 
 * with an arithmetic progression, similar to NumPy's arange function.
 * This version uses ACL data type directly for better performance.
 *
 * @param start Start value (inclusive)
 * @param stop Stop value (exclusive)
 * @param step Step value (default 1)
 * @param acl_type ACL data type constant.
 * @return NPUArray An array with an arithmetic progression.
 */
NPUArray Arange(double start, double stop, double step, aclDataType acl_type);

// 重载函数声明 - 支持自动 dtype 转换
/**
 * @brief Create an array of ones with automatic dtype conversion.
 *
 * Overloaded version that accepts any object that can be converted to numpy.dtype.
 *
 * @param shape A vector containing the dimensions of the array, defining its shape.
 * @param dtype_obj Any object that can be converted to numpy.dtype (e.g., np.float32, np.int32).
 * @return NPUArray An array initialized to ones.
 */
NPUArray Ones(const std::vector<int64_t>& shape, py::object dtype_obj);

/**
 * @brief Create an array of zeros with automatic dtype conversion.
 *
 * Overloaded version that accepts any object that can be converted to numpy.dtype.
 *
 * @param shape A vector containing the dimensions of the array, defining its shape.
 * @param dtype_obj Any object that can be converted to numpy.dtype (e.g., np.float32, np.int32).
 * @return NPUArray An array initialized to zeros.
 */
NPUArray Zeros(const std::vector<int64_t>& shape, py::object dtype_obj);

/**
 * @brief Create an array with specified value and automatic dtype conversion.
 *
 * Overloaded version that accepts any object that can be converted to numpy.dtype.
 *
 * @param shape A vector containing the dimensions of the array, defining its shape.
 * @param dtype_obj Any object that can be converted to numpy.dtype (e.g., np.float32, np.int32).
 * @param value The scalar value used to fill the array.
 * @return NPUArray An array filled with the specified value.
 */
NPUArray Full(const std::vector<int64_t>& shape, py::object dtype_obj, const py::object& value);

/**
 * @brief Create an identity matrix with automatic dtype conversion.
 *
 * Overloaded version that accepts any object that can be converted to numpy.dtype.
 *
 * @param n The number of rows and columns in the output array (dimension of the square matrix).
 * @param dtype_obj Any object that can be converted to numpy.dtype (e.g., np.float32, np.int32).
 * @return NPUArray An identity matrix.
 */
NPUArray Eye(int64_t n, py::object dtype_obj);

/**
 * @brief Create an empty array with automatic dtype conversion.
 *
 * Overloaded version that accepts any object that can be converted to numpy.dtype.
 *
 * @param shape A vector containing the dimensions of the array, defining its shape.
 * @param dtype_obj Any object that can be converted to numpy.dtype (e.g., np.float32, np.int32).
 * @return NPUArray An uninitialized array.
 */
NPUArray Empty(const std::vector<int64_t>& shape, py::object dtype_obj);

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
NPUArray Arange(double start, double stop, double step, py::object dtype_obj);
