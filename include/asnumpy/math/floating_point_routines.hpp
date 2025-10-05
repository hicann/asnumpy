#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

namespace asnumpy {
/**
 * @brief Compute element-wise sign bit check.
 * 
 * Equivalent to numpy.signbit(x), returns a boolean array indicating whether the sign bit is set (negative values).
 * 
 * @param x NPUArray, input array (numeric type)
 * @return NPUArray Boolean array where True indicates negative elements (sign bit set)
 */
NPUArray Signbit(const NPUArray& x);

}