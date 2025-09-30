#pragma once

#include <asnumpy/utils/npu_array.hpp>
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

/**
 * @brief Round elements of the array to the given number of decimals.
 *
 * Uses aclnnRoundDecimals to round each element to the specified number of decimal places.
 *
 * @param a Input array.
 * @param decimals Number of decimal places to round to (default 0).
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with elements rounded to the specified decimals.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray Around(const NPUArray& x, int decimals, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Round elements of the array to the given number of decimals.
 *
 * Equivalent to around(). Provided for API compatibility with NumPy.
 *
 * @param a Input array.
 * @param decimals Number of decimal places to round to (default 0).
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with elements rounded to the specified decimals.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray Round_(const NPUArray& x, int decimals, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Round elements of the array to the nearest integer.
 *
 * Calculates element-wise nearest integers on NPU by calling aclnnRound.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with elements rounded to the nearest integer.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray Rint(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Truncate elements of the array towards zero.
 *
 * Applies element-wise truncation on NPU by calling aclnnTrunc.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with truncated values.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray Fix(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute the floor of each element in the input array.
 *
 * Applies element-wise floor operation on NPU by calling aclnnFloor.
 *
 * @param x Input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with floored values.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray Floor(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute the ceiling of each element in the input array.
 * 
 * Equivalent to numpy.ceil(x), returns the smallest integer greater than or equal to each element.
 * 
 * @param x NPUArray, input array (floating-point type)
 * @return NPUArray Element-wise ceiling values of x
 */
NPUArray Ceil(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute element-wise ceiling of the input.
 * 
 * Equivalent to numpy.ceil(x), returns the smallest integer greater than or equal to each element.
 * 
 * @param x NPUArray, input array (floating-point type)
 * @return NPUArray Element-wise ceiling values of x
 */
NPUArray Trunc(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);