#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>


//NPUArray Convolve(const NPUArray& a, const NPUArray& v);

NPUArray Clip(const NPUArray& a, const NPUArray& a_min, const NPUArray& a_max);
NPUArray Clip(const NPUArray& a, const py::object& a_min, const py::object& a_max);
NPUArray Clip(const NPUArray& a, const py::object& a_min, const NPUArray& a_max);
NPUArray Clip(const NPUArray& a, const NPUArray& a_min, const py::object& a_max);

NPUArray Square(const NPUArray& x);

NPUArray Absolute(const NPUArray& x);

NPUArray Fabs(const NPUArray& x);

NPUArray Sign(const NPUArray& x);

NPUArray Heaviside(const NPUArray& x1, const NPUArray& x2);

NPUArray maximum(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

NPUArray minimum(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

NPUArray fmax(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);

NPUArray fmin(const NPUArray& x1, const NPUArray& x2, py::dtype dtype);