#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

NPUArray Sinh(const NPUArray& x);

NPUArray Cosh(const NPUArray& x);

NPUArray Tanh(const NPUArray& x);

NPUArray Arcsinh(const NPUArray& x);

NPUArray Arccosh(const NPUArray& x);

NPUArray Arctanh(const NPUArray& x);