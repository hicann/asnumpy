#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

std::vector<NPUArray> Linalg_Qr(const NPUArray& a, const std::string& mode);