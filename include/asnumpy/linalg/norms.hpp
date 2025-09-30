#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

NPUArray Linalg_Norm(const NPUArray& a, double ord, const std::vector<int64_t>& axis, bool keepdims);

NPUArray Linalg_Det(const NPUArray& a);

std::vector<NPUArray> Linalg_Slogdet(const NPUArray& a);