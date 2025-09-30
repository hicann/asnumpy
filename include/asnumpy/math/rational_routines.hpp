#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

NPUArray Lcm(const NPUArray& x1, const NPUArray& x2);

NPUArray Gcd(const NPUArray& x1, const NPUArray& x2);