#pragma once

#include "../utils/npu_array.hpp"
#include <utility>

namespace asnumpy{

/**
 * @brief Reduce array by logical AND operation over all elements.
 * 
 * Equivalent to numpy.all(x), returns True if all elements are True.
 *
 * @param x NPUArray, input array (boolean type: ACL_BOOL)
 * @return NPUArray Boolean scalar result
 */
NPUArray All(const NPUArray& x);

/**
 * @brief Reduce array by logical AND operation over specified dimensions.
 * 
 * Equivalent to numpy.all(x, axis=dim, keepdims=keepdims).
 *
 * @param x NPUArray, input array (boolean type: ACL_BOOL)
 * @param dim Dimensions to reduce
 * @param keepdims Whether to keep reduced dimensions with size 1
 * @return NPUArray Reduced boolean array
 */
NPUArray All(const NPUArray& x, const std::vector<int64_t>& dim, bool keepdims);

/**
 * @brief Reduce array by logical OR operation over all elements.
 * 
 * Equivalent to numpy.any(x), returns True if any element is True.
 *
 * @param x NPUArray, input array (boolean type: ACL_BOOL)
 * @return NPUArray Boolean scalar result
 */
NPUArray Any(const NPUArray& x);

/**
 * @brief Reduce array by logical OR operation over specified dimensions.
 * 
 * Equivalent to numpy.any(x, axis=dim, keepdims=keepdims).
 *
 * @param x NPUArray, input array (boolean type: ACL_BOOL)
 * @param dim Dimensions to reduce
 * @param keepdims Whether to keep reduced dimensions with size 1
 * @return NPUArray Reduced boolean array
 */
NPUArray Any(const NPUArray& x, const std::vector<int64_t>& dim, bool keepdims);

NPUArray IsFinite(const NPUArray& x);
NPUArray IsInf(const NPUArray& x);
NPUArray IsNegInf(const NPUArray& x);
NPUArray IsPosInf(const NPUArray& x);
NPUArray LogicalAnd(const NPUArray& x, const NPUArray& y);
NPUArray LogicalOr(const NPUArray& x, const NPUArray& y);
NPUArray LogicalNot(const NPUArray& x);
NPUArray LogicalXor(const NPUArray& x, const NPUArray& y);

/**
 * @brief Perform element-wise greater-than comparison between two arrays.
 * 
 * Compares each element of x1 and x2 and returns a boolean array
 * indicating where x1 > x2. Uses aclnnGtTensor internally.
 * 
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype (optional) Target numpy dtype for the output array (default: np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 > x2.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray greater(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Perform element-wise greater-than comparison between an array and a scalar.
 * 
 * Compares each element of x1 with the scalar value and returns a boolean array
 * indicating where x1 > scalar. Uses aclnnGtScalar internally.
 * 
 * @param x1 Input array.
 * @param scalar Scalar value to compare against.
 * @param dtype (optional) Target numpy dtype for the output array (default: np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 > scalar.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray greater(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Perform element-wise greater-than-or-equal comparison between two arrays.
 * 
 * Compares each element of x1 and x2 and returns a boolean array
 * indicating where x1 >= x2. Uses aclnnGeTensor internally.
 * 
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype (optional) Target numpy dtype for the output array (default: np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 >= x2.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray greater_equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Perform element-wise greater-than-or-equal comparison between an array and a scalar.
 * 
 * Compares each element of x1 with the scalar value and returns a boolean array
 * indicating where x1 >= scalar. Uses aclnnGeScalar internally.
 * 
 * @param x1 Input array.
 * @param scalar Scalar value to compare against.
 * @param dtype (optional) Target numpy dtype for the output array (default: np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 >= scalar.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray greater_equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Perform element-wise less-than comparison between two arrays.
 * 
 * Compares each element of x1 and x2 and returns a boolean array
 * indicating where x1 < x2. Uses aclnnLtTensor internally.
 * 
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype (optional) Target numpy dtype for the output array (default: np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 < x2.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray less(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Perform element-wise less-than comparison between an array and a scalar.
 * 
 * Compares each element of x1 with the scalar value and returns a boolean array
 * indicating where x1 < scalar. Uses aclnnLtScalar internally.
 * 
 * @param x1 Input array.
 * @param scalar Scalar value to compare against.
 * @param dtype (optional) Target numpy dtype for the output array (default: np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 < scalar.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray less(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Perform element-wise less-than-or-equal comparison between two arrays.
 * 
 * Compares each element of x1 and x2 and returns a boolean array
 * indicating where x1 <= x2. Uses aclnnLeTensor internally.
 * 
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype (optional) Target numpy dtype for the output array (default: np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 <= x2.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray less_equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Perform element-wise less-than-or-equal comparison between an array and a scalar.
 * 
 * Compares each element of x1 with the scalar value and returns a boolean array
 * indicating where x1 <= scalar. Uses aclnnLeScalar internally.
 * 
 * @param x1 Input array.
 * @param scalar Scalar value to compare against.
 * @param dtype (optional) Target numpy dtype for the output array (default: np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 <= scalar.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray less_equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Perform element-wise equality comparison between two arrays.
 * 
 * Compares each element of x1 and x2 and returns a boolean array
 * indicating where x1 == x2. Uses aclnnEqual internally.
 * 
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype (optional) Target numpy dtype for the output array (default: np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 == x2.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Perform element-wise not-equal comparison between two arrays.
 * 
 * Compares each element of x1 and x2 and returns a boolean array
 * indicating where x1 != x2. Uses aclnnNeTensor internally.
 * 
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype (optional) Target numpy dtype for the output array (default: np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 != x2.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray not_equal(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Perform element-wise not-equal comparison between an array and a scalar.
 * 
 * Compares each element of x1 with a scalar value and returns a boolean array
 * indicating where x1 != scalar. Uses aclnnNeScalar internally.
 * 
 * @param x1 Input array.
 * @param scalar Scalar value to compare with.
 * @param dtype (optional) Target numpy dtype for the output array (default: np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 != scalar.
 * @throws std::runtime_error If ACL operation fails or scalar type is invalid.
 */
NPUArray not_equal(const NPUArray& x1, const py::object& scalar, std::optional<py::dtype> dtype = std::nullopt);

}