#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

NPUArray Prod(const NPUArray& a, int64_t axis, py::dtype dtype, bool keepdims);
NPUArray Prod(const NPUArray& a, py::dtype dtype);

NPUArray Sum(const NPUArray& a, const std::vector<int64_t>& axis, py::dtype dtype, bool keepdims);
double Sum(const NPUArray& a);
    
NPUArray Nanprod(const NPUArray& a, int64_t axis, py::dtype dtype, bool keepdims);
NPUArray Nanprod(const NPUArray& a, py::dtype dtype);

NPUArray Nansum(const NPUArray& a, const std::vector<int64_t>& axis, py::dtype dtype, bool keepdims);
NPUArray Nansum(const NPUArray& a, py::dtype dtype);

NPUArray Cumprod(const NPUArray& a, int64_t axis, py::dtype dtype);

NPUArray Cumsum(const NPUArray& a, int64_t axis, py::dtype dtype);

NPUArray Nancumprod(const NPUArray& a, int64_t axis, py::dtype dtype);

NPUArray Nancumsum(const NPUArray& a, int64_t axis, py::dtype dtype);

NPUArray Cross(const NPUArray& a, const NPUArray& b, int64_t axisa, int64_t axisb, int64_t axisc, int64_t axis);