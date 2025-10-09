#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

namespace asnumpy {
/**
 * @brief Compute element-wise least common multiple (LCM).
 * 
 * Equivalent to numpy.lcm(x1, x2), returns the smallest positive integer divisible by both x1 and x2.
 * Implemented using the relationship: LCM(a, b) = |a * b| / GCD(a, b)
 * 
 * @param x1 NPUArray, input array (integer type)
 * @param x2 NPUArray, input array (integer type)
 * @return NPUArray Element-wise LCM of x1 and x2
 */
NPUArray Lcm(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

/**
 * @brief Compute element-wise greatest common divisor (GCD).
 * 
 * Equivalent to numpy.gcd(x1, x2), returns the largest positive integer dividing both x1 and x2.
 * 
 * @param x1 NPUArray, input array (integer type)
 * @param x2 NPUArray, input array (integer type)
 * @return NPUArray Element-wise GCD of x1 and x2
 */
NPUArray Gcd(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

}