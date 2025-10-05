#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

namespace asnumpy {

//NPUArray Convolve(const NPUArray& a, const NPUArray& v);

NPUArray Clip(const NPUArray& a, const NPUArray& a_min, const NPUArray& a_max);
NPUArray Clip(const NPUArray& a, const NPUArray& a_min, float a_max);
NPUArray Clip(const NPUArray& a, float a_min, float a_max);
NPUArray Clip(const NPUArray& a, float a_min, const NPUArray& a_max);

NPUArray Square(const NPUArray& x);

NPUArray Absolute(const NPUArray& x);

NPUArray Fabs(const NPUArray& x);

NPUArray Nan_to_num(const NPUArray& x, float nan, py::object posinf, py::object neginf);

NPUArray Sign(const NPUArray& x);

NPUArray Heaviside(const NPUArray& x1, const NPUArray& x2);

NPUArray Maximum(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Minimum(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Fmax(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Fmin(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype = std::nullopt);

}