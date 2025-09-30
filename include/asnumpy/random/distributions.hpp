#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

NPUArray Generator_Pareto(float a, const std::vector<int64_t>& size);

NPUArray Generator_Rayleigh(float scale, const std::vector<int64_t>& size);

NPUArray Generator_Normal(float loc, float scale, const std::vector<int64_t>& size);

NPUArray Generator_Uniform(double low, double high, const std::vector<int64_t>& size);

NPUArray Generator_Standard_normal(const std::vector<int64_t>& size);

NPUArray Generator_Standard_cauchy(const std::vector<int64_t>& size);

NPUArray Generator_Weibull(float a, const std::vector<int64_t>& size);

NPUArray Binomial(int n, float p, const std::vector<int64_t>& size = {});

NPUArray Exponential(float scale = 1.0f, const std::vector<int64_t>& size = {});

NPUArray Geometric(float p, const std::vector<int64_t>& size = {});

NPUArray Gumbel(double loc = 0.0, double scale = 1.0, const std::vector<int64_t>& size = {});

NPUArray Laplace(double loc = 0.0, double scale = 1.0, const std::vector<int64_t>& size = {});

NPUArray Logistic(double loc = 0.0, double scale = 1.0, const std::vector<int64_t>& size = {});

NPUArray Lognormal(double mean = 0.0, double sigma = 1.0, const std::vector<int64_t>& size = {});

NPUArray Multinomial(int64_t n, const NPUArray& pvals, const std::vector<int64_t>& size = {});