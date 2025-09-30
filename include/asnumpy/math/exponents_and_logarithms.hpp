#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

NPUArray Exp(const NPUArray& x, py::dtype dtype);

NPUArray Expm1(const NPUArray& x, py::dtype dtype);

NPUArray Exp2(const NPUArray& x, py::dtype dtype);

NPUArray Log(const NPUArray& x, py::dtype dtype);

NPUArray Log10(const NPUArray& x, py::dtype dtype);

NPUArray Log2(const NPUArray& x, py::dtype dtype);

NPUArray Log1p(const NPUArray& x, py::dtype dtype);

NPUArray Logaddexp(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

NPUArray Logaddexp2(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);