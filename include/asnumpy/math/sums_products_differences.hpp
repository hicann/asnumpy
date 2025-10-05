#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <optional>
#include <utility>

namespace asnumpy {
    NPUArray Prod(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype=std::nullopt);
    double Prod(const NPUArray& a);

    NPUArray Sum(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype=std::nullopt);
    double Sum(const NPUArray& a);
        
    NPUArray Nanprod(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype=std::nullopt);
    double Nanprod(const NPUArray& a);

    NPUArray Nansum(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype=std::nullopt);
    double Nansum(const NPUArray& a);

    NPUArray Cumprod(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype=std::nullopt);

    NPUArray Cumsum(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype=std::nullopt);

    NPUArray Nancumprod(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype=std::nullopt);

    NPUArray Nancumsum(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype=std::nullopt);

    NPUArray Cross(const NPUArray& a, const NPUArray& b, int64_t axis);
}