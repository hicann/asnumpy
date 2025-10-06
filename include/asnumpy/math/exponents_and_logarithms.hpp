#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

namespace asnumpy {
    NPUArray Exp(const NPUArray& x);

    NPUArray Expm1(const NPUArray& x);

    NPUArray Exp2(const NPUArray& x);

    NPUArray Log(const NPUArray& x);

    NPUArray Log10(const NPUArray& x);

    NPUArray Log2(const NPUArray& x);

    NPUArray Log1p(const NPUArray& x);

    NPUArray Logaddexp(const NPUArray& x1, const NPUArray& x2);

    NPUArray Logaddexp2(const NPUArray& x1, const NPUArray& x2);
}