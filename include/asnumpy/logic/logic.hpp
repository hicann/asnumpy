#pragma once

#include "../utils/npu_array.hpp"
#include <utility>


NPUArray All(const NPUArray& x);
NPUArray Any(const NPUArray& x);
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
 * Compares each element of x1 and x2 (or scalar) and returns a boolean array
 * indicating where x1 > x2. Uses aclnnGtTensor for array-array comparison
 * and aclnnGtScalar for array-scalar comparison.
 * 
 * @param x1 First input array.
 * @param x2 Second input array (array or scalar).
 * @param dtype Target numpy dtype for the output array (usually np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 > x2.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray greater(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

/**
 * @brief Perform element-wise greater-than comparison between an array and a scalar.
 * 
 * Compares each element of x1 with the scalar value and returns a boolean array
 * indicating where x1 > scalar. Uses aclnnGtScalar internally.
 * 
 * @param x1 Input array.
 * @param scalar Scalar value to compare against.
 * @param dtype Target numpy dtype for the output array (usually np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 > scalar.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray greater(const NPUArray& x1, const py::object& scalar, py::dtype dtype);


/**
 * @brief Perform element-wise greater-than-or-equal comparison between two arrays.
 * 
 * Compares each element of x1 and x2 (or scalar) and returns a boolean array
 * indicating where x1 >= x2. Uses aclnnGeTensor for array-array comparison
 * and aclnnGeScalar for array-scalar comparison.
 * 
 * @param x1 First input array.
 * @param x2 Second input array (array or scalar).
 * @param dtype Target numpy dtype for the output array (usually np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 >= x2.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray greater_equal(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

/**
 * @brief Perform element-wise greater-than-or-equal comparison between an array and a scalar.
 * 
 * Compares each element of x1 with the scalar value and returns a boolean array
 * indicating where x1 >= scalar. Uses aclnnGeScalar internally.
 * 
 * @param x1 Input array.
 * @param scalar Scalar value to compare against.
 * @param dtype Target numpy dtype for the output array (usually np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 >= scalar.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray greater_equal(const NPUArray& x1, const py::object& scalar, py::dtype dtype);

/**
 * @brief Perform element-wise less-than comparison between two arrays.
 * 
 * Compares each element of x1 and x2 (or scalar) and returns a boolean array
 * indicating where x1 < x2.
 * 
 * @param x1 First input array.
 * @param x2 Second input array (array or scalar).
 * @param dtype Target numpy dtype for the output array (usually np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 < x2.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray less(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

/**
 * @brief Perform element-wise less-than comparison between an array and a scalar.
 * 
 * Compares each element of x1 with the scalar value and returns a boolean array
 * indicating where x1 < scalar.
 * 
 * @param x1 Input array.
 * @param scalar Scalar value to compare against.
 * @param dtype Target numpy dtype for the output array (usually np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 < scalar.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray less(const NPUArray& x1, const py::object& scalar, py::dtype dtype);

/**
 * @brief Perform element-wise less-than-or-equal comparison between two arrays.
 * 
 * Compares each element of x1 and x2 (or scalar) and returns a boolean array
 * indicating where x1 <= x2. Uses aclnnLeTensor for array-array comparison
 * and aclnnLeScalar for array-scalar comparison.
 * 
 * @param x1 First input array.
 * @param x2 Second input array (array or scalar).
 * @param dtype Target numpy dtype for the output array (usually np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 <= x2.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray less_equal(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

/**
 * @brief Perform element-wise less-than-or-equal comparison between an array and a scalar.
 * 
 * Compares each element of x1 with the scalar value and returns a boolean array
 * indicating where x1 <= scalar. Uses aclnnLeScalar internally.
 * 
 * @param x1 Input array.
 * @param scalar Scalar value to compare against.
 * @param dtype Target numpy dtype for the output array (usually np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 <= scalar.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray less_equal(const NPUArray& x1, const py::object& scalar, py::dtype dtype);

/**
 * @brief Perform element-wise equality comparison between two arrays.
 * 
 * Compares each element of x1 and x2 and returns a boolean array
 * indicating where x1 == x2. Uses aclnnEqual internally.
 * 
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array (usually np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 == x2.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray equal(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

/**
 * @brief Perform element-wise not-equal comparison between two arrays.
 * 
 * Compares each element of x1 and x2 and returns a boolean array
 * indicating where x1 != x2. Uses aclnnNeTensor internally.
 * 
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array (usually np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 != x2.
 * @throws std::runtime_error If ACL operation fails.
 */
NPUArray not_equal(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

/**
 * @brief Perform element-wise not-equal comparison between an array and a scalar.
 * 
 * Compares each element of x1 with a scalar value and returns a boolean array
 * indicating where x1 != scalar. Uses aclnnNeScalar internally.
 * 
 * @param x1 Input array.
 * @param scalar Scalar value to compare with.
 * @param dtype Target numpy dtype for the output array (usually np.bool_).
 * @return NPUArray Boolean array where each element indicates the result of x1 != scalar.
 * @throws std::runtime_error If ACL operation fails or scalar type is invalid.
 */
NPUArray not_equal(const NPUArray& x1, const py::object& scalar, py::dtype dtype);
