#include <asnumpy/random/distributions.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_normal_out.h>
#include <aclnnop/aclnn_uniform.h>
#include <aclnnop/aclnn_rsub.h>
#include <aclnnop/aclnn_pow.h>
#include <aclnnop/aclnn_reciprocal.h>
#include <aclnnop/aclnn_log.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_sqrt.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_tan.h>
#include <aclnnop/aclnn_bernoulli.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_floor.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_foreach_sub_scalar.h>
#include <aclnnop/aclnn_foreach_mul_scalar.h>
#include <aclnnop/aclnn_foreach_mul_scalar.h>
#include <aclnnop/aclnn_foreach_div_scalar.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_foreach_add_scalar.h>
#include <aclnnop/aclnn_foreach_mul_scalar.h>
#include <aclnnop/aclnn_exp.h> 
#include <aclnnop/aclnn_sign.h>
#include <aclnnop/aclnn_multinomial.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>
#include <random>

NPUArray Generator_Pareto(float a, const std::vector<int64_t>& size) {
    auto uni_temp = NPUArray(size, ACL_FLOAT);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    uint64_t seed = dist(gen);
    uint64_t offset = 0;
    uint64_t uni_workspaceSize = 0;
    aclOpExecutor* uni_executor;
    auto error = aclnnInplaceUniformGetWorkspaceSize(uni_temp.tensorPtr, 0.0, 1.0, seed, offset, &uni_workspaceSize, &uni_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* uni_workspaceAddr = nullptr;
    if(uni_workspaceSize > 0) {
        error = aclrtMalloc(&uni_workspaceAddr, uni_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceUniform(uni_workspaceAddr, uni_workspaceSize, uni_executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceUniform error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    auto rsubs_temp = NPUArray(size, ACL_FLOAT);
    float scalar1 = 1.0f;
    float scalar2 = 1.0f;
    aclScalar* other = aclCreateScalar(&scalar1, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&scalar2, ACL_FLOAT);
    uint64_t rsubs_workspaceSize = 0;
    aclOpExecutor* rsubs_executor;
    error = aclnnRsubsGetWorkspaceSize(uni_temp.tensorPtr, other, alpha, rsubs_temp.tensorPtr, &rsubs_workspaceSize, &rsubs_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* rsubs_workspaceAddr = nullptr;
    if(rsubs_workspaceSize > 0) {
        error = aclrtMalloc(&rsubs_workspaceAddr, rsubs_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnRsubs(rsubs_workspaceAddr, rsubs_workspaceSize, rsubs_executor, nullptr);
    CheckAclnnStatus(error, "aclnnRsubs error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    auto result = NPUArray(size, ACL_FLOAT);
    float scalar3 = 1.0f / a;
    aclScalar* exponent = aclCreateScalar(&scalar3, ACL_FLOAT);
    uint64_t exp_workspaceSize = 0;
    aclOpExecutor* exp_executor;
    error = aclnnPowTensorScalarGetWorkspaceSize(result.tensorPtr, exponent, result.tensorPtr, &exp_workspaceSize, &exp_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* exp_workspaceAddr = nullptr;
    if(exp_workspaceSize > 0) {
        error = aclrtMalloc(&exp_workspaceAddr, exp_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnPowTensorScalar(exp_workspaceAddr, exp_workspaceSize, exp_executor, nullptr);
    CheckAclnnStatus(error, "aclnnPowTensorScalar error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    uint64_t reci_workspaceSize = 0;
    aclOpExecutor* reci_executor;
    error = aclnnInplaceReciprocalGetWorkspaceSize(result.tensorPtr, &reci_workspaceSize, &reci_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* reci_workspaceAddr = nullptr;
    if(reci_workspaceSize > 0) {
        error = aclrtMalloc(&reci_workspaceAddr, reci_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceReciprocal(reci_workspaceAddr, reci_workspaceSize, reci_executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceReciprocal error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Generator_Rayleigh(float scale, const std::vector<int64_t>& size) {
    auto uni_temp = NPUArray(size, ACL_FLOAT);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    uint64_t seed = dist(gen);
    uint64_t offset = 0;
    uint64_t uni_workspaceSize = 0;
    aclOpExecutor* uni_executor;
    auto error = aclnnInplaceUniformGetWorkspaceSize(uni_temp.tensorPtr, 0.0, 1.0, seed, offset, &uni_workspaceSize, &uni_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* uni_workspaceAddr = nullptr;
    if(uni_workspaceSize > 0) {
        error = aclrtMalloc(&uni_workspaceAddr, uni_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceUniform(uni_workspaceAddr, uni_workspaceSize, uni_executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceUniform error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    auto result = NPUArray(size, ACL_FLOAT);
    float scalar1 = 1.0f;
    float scalar2 = 1.0f;
    aclScalar* other = aclCreateScalar(&scalar1, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&scalar2, ACL_FLOAT);
    uint64_t rsubs_workspaceSize = 0;
    aclOpExecutor* rsubs_executor;
    error = aclnnRsubsGetWorkspaceSize(uni_temp.tensorPtr, other, alpha, result.tensorPtr, &rsubs_workspaceSize, &rsubs_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* rsubs_workspaceAddr = nullptr;
    if(rsubs_workspaceSize > 0) {
        error = aclrtMalloc(&rsubs_workspaceAddr, rsubs_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnRsubs(rsubs_workspaceAddr, rsubs_workspaceSize, rsubs_executor, nullptr);
    CheckAclnnStatus(error, "aclnnRsubs error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    uint64_t log_workspaceSize = 0;
    aclOpExecutor* log_executor;
    error = aclnnInplaceLogGetWorkspaceSize(result.tensorPtr, &log_workspaceSize, &log_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* log_workspaceAddr = nullptr;
    if(log_workspaceSize > 0) {
        error = aclrtMalloc(&log_workspaceAddr, log_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceLog(log_workspaceAddr, log_workspaceSize, log_executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceLog error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    float scalar3 = -2.0f;
    aclScalar* mulnum = aclCreateScalar(&scalar3, ACL_FLOAT);
    uint64_t muls_workspaceSize = 0;
    aclOpExecutor* muls_executor;
    error = aclnnInplaceMulsGetWorkspaceSize(result.tensorPtr, mulnum, &muls_workspaceSize, &muls_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* muls_workspaceAddr = nullptr;
    if(muls_workspaceSize > 0) {
        error = aclrtMalloc(&muls_workspaceAddr, muls_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceMuls(muls_workspaceAddr, muls_workspaceSize, muls_executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceMuls error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    uint64_t sqrt_workspaceSize = 0;
    aclOpExecutor* sqrt_executor;
    error = aclnnInplaceSqrtGetWorkspaceSize(result.tensorPtr, &sqrt_workspaceSize, &sqrt_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* sqrt_workspaceAddr = nullptr;
    if(sqrt_workspaceSize > 0) {
        error = aclrtMalloc(&sqrt_workspaceAddr, sqrt_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceSqrt(sqrt_workspaceAddr, sqrt_workspaceSize, sqrt_executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceSqrt error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    aclScalar* muls = aclCreateScalar(&scale, ACL_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    error = aclnnInplaceMulsGetWorkspaceSize(result.tensorPtr, muls, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceMuls(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceMuls error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Generator_Normal(float loc, float scale, const std::vector<int64_t>& size) {
    auto result = NPUArray(size, ACL_DOUBLE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    int64_t seed = dist(gen);
    int64_t offset = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnNormalFloatFloatGetWorkspaceSize(loc, scale, seed, offset, result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnNormalFloatFloat(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnNormalFloatFloat error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Generator_Uniform(double low, double high, const std::vector<int64_t>& size) {
    auto result = NPUArray(size, ACL_DOUBLE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    uint64_t seed = dist(gen);
    uint64_t offset = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnInplaceUniformGetWorkspaceSize(result.tensorPtr, low, high, seed, offset, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceUniform(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceUniform error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Generator_Standard_normal(const std::vector<int64_t>& size) {
    float loc = 0.0f;
    float scale = 1.0f;
    auto result = NPUArray(size, ACL_DOUBLE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    int64_t seed = dist(gen);
    int64_t offset = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnNormalFloatFloatGetWorkspaceSize(loc, scale, seed, offset, result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnNormalFloatFloat(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnNormalFloatFloat error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Generator_Standard_cauchy(const std::vector<int64_t>& size) {
    auto result = NPUArray(size, ACL_DOUBLE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    uint64_t seed = dist(gen);
    uint64_t offset = 0;
    uint64_t uni_workspaceSize = 0;
    aclOpExecutor* uni_executor;
    auto error = aclnnInplaceUniformGetWorkspaceSize(result.tensorPtr, 0.0, 1.0, seed, offset, &uni_workspaceSize, &uni_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* uni_workspaceAddr = nullptr;
    if(uni_workspaceSize > 0) {
        error = aclrtMalloc(&uni_workspaceAddr, uni_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceUniform(uni_workspaceAddr, uni_workspaceSize, uni_executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceUniform error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    float scalar1 = 0.5f;
    float scalar2 = 1.0f;
    aclScalar* other = aclCreateScalar(&scalar1, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&scalar2, ACL_FLOAT);
    uint64_t subs_workspaceSize = 0;
    aclOpExecutor* subs_executor;
    error = aclnnInplaceSubsGetWorkspaceSize(result.tensorPtr, other, alpha, &subs_workspaceSize, &subs_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* subs_workspaceAddr = nullptr;
    if(subs_workspaceSize > 0) {
        error = aclrtMalloc(&subs_workspaceAddr, subs_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceSubs(subs_workspaceAddr, subs_workspaceSize, subs_executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceSubs error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    double PI = 3.141592653589793238462643383279502884197169399375105820974944;
    aclScalar* pi = aclCreateScalar(&PI, ACL_DOUBLE);
    uint64_t muls_workspaceSize = 0;
    aclOpExecutor* muls_executor;
    error = aclnnInplaceMulsGetWorkspaceSize(result.tensorPtr, pi, &muls_workspaceSize, &muls_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* muls_workspaceAddr = nullptr;
    if(muls_workspaceSize > 0) {
        error = aclrtMalloc(&muls_workspaceAddr, muls_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceMuls(muls_workspaceAddr, muls_workspaceSize, muls_executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceMuls error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    uint64_t tan_workspaceSize = 0;
    aclOpExecutor* tan_executor;
    error = aclnnInplaceTanGetWorkspaceSize(result.tensorPtr, &tan_workspaceSize, &tan_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* tan_workspaceAddr = nullptr;
    if(tan_workspaceSize > 0) {
        error = aclrtMalloc(&tan_workspaceAddr, tan_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceTan(tan_workspaceAddr, tan_workspaceSize, tan_executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceTan error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Generator_Weibull(float a, const std::vector<int64_t>& size) {
    auto uni_temp = NPUArray(size, ACL_FLOAT);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    uint64_t seed = dist(gen);
    uint64_t offset = 0;
    uint64_t uni_workspaceSize = 0;
    aclOpExecutor* uni_executor;
    auto error = aclnnInplaceUniformGetWorkspaceSize(uni_temp.tensorPtr, 0.0, 1.0, seed, offset, &uni_workspaceSize, &uni_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* uni_workspaceAddr = nullptr;
    if(uni_workspaceSize > 0) {
        error = aclrtMalloc(&uni_workspaceAddr, uni_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceUniform(uni_workspaceAddr, uni_workspaceSize, uni_executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceUniform error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    auto rsubs_temp = NPUArray(size, ACL_FLOAT);
    float scalar1 = 1.0f;
    float scalar2 = 1.0f;
    aclScalar* other1 = aclCreateScalar(&scalar1, ACL_FLOAT);
    aclScalar* alpha = aclCreateScalar(&scalar2, ACL_FLOAT);
    uint64_t rsubs_workspaceSize1 = 0;
    aclOpExecutor* rsubs_executor1;
    error = aclnnRsubsGetWorkspaceSize(uni_temp.tensorPtr, other1, alpha, rsubs_temp.tensorPtr, &rsubs_workspaceSize1, &rsubs_executor1);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* rsubs_workspaceAddr1 = nullptr;
    if(rsubs_workspaceSize1 > 0) {
        error = aclrtMalloc(&rsubs_workspaceAddr1, rsubs_workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnRsubs(rsubs_workspaceAddr1, rsubs_workspaceSize1, rsubs_executor1, nullptr);
    CheckAclnnStatus(error, "aclnnRsubs error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    uint64_t log_workspaceSize = 0;
    aclOpExecutor* log_executor;
    error = aclnnInplaceLogGetWorkspaceSize(rsubs_temp.tensorPtr, &log_workspaceSize, &log_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* log_workspaceAddr = nullptr;
    if(log_workspaceSize > 0) {
        error = aclrtMalloc(&log_workspaceAddr, log_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInplaceLog(log_workspaceAddr, log_workspaceSize, log_executor, nullptr);
    CheckAclnnStatus(error, "aclnnInplaceLog error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    auto result = NPUArray(size, ACL_FLOAT);
    float scalar3 = 0.0f;
    aclScalar* other2 = aclCreateScalar(&scalar3, ACL_FLOAT);
    uint64_t rsubs_workspaceSize = 0;
    aclOpExecutor* rsubs_executor;
    error = aclnnRsubsGetWorkspaceSize(rsubs_temp.tensorPtr, other2, alpha, result.tensorPtr, &rsubs_workspaceSize, &rsubs_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* rsubs_workspaceAddr = nullptr;
    if(rsubs_workspaceSize > 0) {
        error = aclrtMalloc(&rsubs_workspaceAddr, rsubs_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnRsubs(rsubs_workspaceAddr, rsubs_workspaceSize, rsubs_executor, nullptr);
    CheckAclnnStatus(error, "aclnnRsubs error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    float scalar4 = 1.0f / a;
    aclScalar* exponent = aclCreateScalar(&scalar4, ACL_FLOAT);
    uint64_t exp_workspaceSize = 0;
    aclOpExecutor* exp_executor;
    error = aclnnPowTensorScalarGetWorkspaceSize(result.tensorPtr, exponent, result.tensorPtr, &exp_workspaceSize, &exp_executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* exp_workspaceAddr = nullptr;
    if(exp_workspaceSize > 0) {
        error = aclrtMalloc(&exp_workspaceAddr, exp_workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnPowTensorScalar(exp_workspaceAddr, exp_workspaceSize, exp_executor, nullptr);
    CheckAclnnStatus(error, "aclnnPowTensorScalar error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Binomial(int n, float p, const std::vector<int64_t>& size) {
    // 1. 参数校验
    if (n < 0) throw std::runtime_error(fmt::format("Binomial: n={} < 0", n));
    if (p < 0.0f || p > 1.0f) throw std::runtime_error(fmt::format("Binomial: p={} ∉ [0,1]", p));
    if (n == 0) return NPUArray(size, ACL_INT32);

    // 2. 构造伯努利张量形状
    std::vector<int64_t> bernoulli_shape = {static_cast<int64_t>(n)};
    bernoulli_shape.insert(bernoulli_shape.end(), size.begin(), size.end());
    NPUArray bernoulli_tensor(bernoulli_shape, ACL_INT32);

    // 3. 构造标量概率张量
    NPUArray prob_tensor({}, ACL_FLOAT);
    void* prob_data_ptr = nullptr;
    auto ret = aclGetRawTensorAddr(prob_tensor.tensorPtr, &prob_data_ptr);
    if (ret != ACL_SUCCESS || !prob_data_ptr) {
        throw std::runtime_error(fmt::format("Binomial: get prob tensor pointer failed, error={}", ret));
    }
    ret = aclrtMemcpy(prob_data_ptr, sizeof(float), &p, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) throw std::runtime_error(fmt::format("Binomial: copy p failed, error={}", ret));

    // 4. 声明资源并创建流
    uint64_t bernoulli_ws = 0;
    aclOpExecutor* bernoulli_exec = nullptr;
    void* bernoulli_ws_addr = nullptr;
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS || !stream) throw std::runtime_error(fmt::format("Binomial: create stream failed, error={}", ret));

    // 5. 生成伯努利张量
    ret = aclnnBernoulliTensorGetWorkspaceSize(
        bernoulli_tensor.tensorPtr, prob_tensor.tensorPtr, 42, 0, 
        bernoulli_tensor.tensorPtr, &bernoulli_ws, &bernoulli_exec
    );
    if (ret != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format("Binomial: bernoulli get ws failed, error={}", ret));
    }
    if (bernoulli_ws > 0) {
        ret = aclrtMalloc(&bernoulli_ws_addr, bernoulli_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclrtDestroyStream(stream);
            throw std::runtime_error(fmt::format("Binomial: bernoulli malloc ws failed, error={}", ret));
        }
    }
    ret = aclnnBernoulliTensor(bernoulli_ws_addr, bernoulli_ws, bernoulli_exec, stream);
    if (ret != ACL_SUCCESS) {
        aclrtFree(bernoulli_ws_addr);
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format("Binomial: bernoulli compute failed, error={}", ret));
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        aclrtFree(bernoulli_ws_addr);
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format("Binomial: bernoulli sync failed, error={}", ret));
    }

    // 6. 归约求和（核心修正部分）
    NPUArray result(size, ACL_INT32);
    uint64_t sum_ws = 0;
    aclOpExecutor* sum_exec = nullptr;
    void* sum_ws_addr = nullptr;
    std::vector<int64_t> reduce_axis = {0};  // 沿第0维求和

    // 6.1 创建aclIntArray类型的归约轴（适配接口要求）
    aclIntArray* dims_array = aclCreateIntArray(reduce_axis.data(), reduce_axis.size());
    if (dims_array == nullptr) {
        aclrtFree(bernoulli_ws_addr);
        aclrtDestroyStream(stream);
        throw std::runtime_error("Binomial: create dims array failed");
    }

    // 6.2 调用修正后的归约求和接口（按文档参数顺序）
    ret = aclnnReduceSumGetWorkspaceSize(
        bernoulli_tensor.tensorPtr,  // 输入张量
        dims_array,                  // 归约轴（aclIntArray类型）
        false,                       // 是否保留归约轴
        ACL_INT32,                   // 输出数据类型（匹配result的类型）
        result.tensorPtr,            // 输出张量
        &sum_ws,
        &sum_exec
    );
    if (ret != ACL_SUCCESS) {
        aclDestroyIntArray(dims_array);
        aclrtFree(bernoulli_ws_addr);
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format("Binomial: sum get ws failed, error={}", ret));
    }

    // 6.3 分配求和工作空间并执行
    if (sum_ws > 0) {
        ret = aclrtMalloc(&sum_ws_addr, sum_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            aclDestroyIntArray(dims_array);
            aclrtFree(bernoulli_ws_addr);
            aclrtDestroyStream(stream);
            throw std::runtime_error(fmt::format("Binomial: sum malloc ws failed, error={}", ret));
        }
    }
    ret = aclnnReduceSum(sum_ws_addr, sum_ws, sum_exec, stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyIntArray(dims_array);
        aclrtFree(sum_ws_addr);
        aclrtFree(bernoulli_ws_addr);
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format("Binomial: sum compute failed, error={}", ret));
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        aclDestroyIntArray(dims_array);
        aclrtFree(sum_ws_addr);
        aclrtFree(bernoulli_ws_addr);
        aclrtDestroyStream(stream);
        throw std::runtime_error(fmt::format("Binomial: sum sync failed, error={}", ret));
    }

    // 7. 释放所有资源
    aclDestroyIntArray(dims_array);  // 销毁归约轴数组
    aclrtFree(sum_ws_addr);
    aclrtFree(bernoulli_ws_addr);
    aclrtDestroyStream(stream);

    return result;
}


NPUArray Exponential(float scale, const std::vector<int64_t>& size) {
    // 1. 参数校验
    if (scale <= 0.0f) {
        throw std::runtime_error(fmt::format("Exponential: scale={} must > 0", scale));
    }

    // 资源声明（在try外声明，确保catch块可访问）
    uint64_t uniform_ws = 0, sub_ws = 0, log_ws = 0, mul_ws = 0;
    aclOpExecutor *uniform_exec = nullptr, *sub_exec = nullptr, *log_exec = nullptr, *mul_exec = nullptr;
    void *uniform_ws_addr = nullptr, *sub_ws_addr = nullptr, *log_ws_addr = nullptr, *mul_ws_addr = nullptr;
    aclrtStream stream = nullptr;
    aclTensorList *x_list = nullptr, *out_list = nullptr;
    aclTensor* x_tensor_array[1] = {nullptr};  // 用于初始化张量列表的数组
    aclTensor* out_tensor_array[1] = {nullptr};

    try {
        // 2. 初始化核心张量
        aclDataType dtype = ACL_DOUBLE;  // 修正：使用ACL_DOUBLE替代ACL_FLOAT64
        NPUArray uniform_tensor(size, dtype);
        NPUArray result(size, dtype);
        NPUArray scalar_one({}, dtype);
        NPUArray scalar_neg_scale({}, dtype);

        // 3. 拷贝标量值到设备（使用aclGetRawTensorAddr获取指针）
        void* one_device_ptr = nullptr;
        auto ret = aclGetRawTensorAddr(scalar_one.tensorPtr, &one_device_ptr);
        if (ret != ACL_SUCCESS || !one_device_ptr) {
            throw std::runtime_error(fmt::format("Exponential: get scalar_one ptr failed, error={}", ret));
        }
        double host_one = 1.0;
        ret = aclrtMemcpy(one_device_ptr, sizeof(double), &host_one, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: copy 1.0 failed, error={}", ret));
        }

        void* neg_scale_device_ptr = nullptr;
        ret = aclGetRawTensorAddr(scalar_neg_scale.tensorPtr, &neg_scale_device_ptr);
        if (ret != ACL_SUCCESS || !neg_scale_device_ptr) {
            throw std::runtime_error(fmt::format("Exponential: get scalar_neg_scale ptr failed, error={}", ret));
        }
        double host_neg_scale = -static_cast<double>(scale);
        ret = aclrtMemcpy(neg_scale_device_ptr, sizeof(double), &host_neg_scale, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: copy -scale failed, error={}", ret));
        }

        // 4. 创建执行流
        ret = aclrtCreateStream(&stream);
        if (ret != ACL_SUCCESS || !stream) {
            throw std::runtime_error(fmt::format("Exponential: create stream failed, error={}", ret));
        }

        // 5. 初始化张量列表（核心修正：直接通过数组创建，无需aclTensorListAdd）
        x_tensor_array[0] = uniform_tensor.tensorPtr;  // 填充输入张量数组
        x_list = aclCreateTensorList(x_tensor_array, 1);  // 直接创建列表
        if (x_list == nullptr) {
            throw std::runtime_error("Exponential: create x_list failed");
        }

        out_tensor_array[0] = result.tensorPtr;  // 填充输出张量数组
        out_list = aclCreateTensorList(out_tensor_array, 1);  // 直接创建列表
        if (out_list == nullptr) {
            throw std::runtime_error("Exponential: create out_list failed");
        }

        // 6. 生成均匀分布U~(0,1)
        ret = aclnnInplaceUniformGetWorkspaceSize(
            uniform_tensor.tensorPtr, 0.0, 1.0,
            static_cast<uint64_t>(time(nullptr)), 0,
            &uniform_ws, &uniform_exec
        );
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: get uniform ws failed, error={}", ret));
        }
        if (uniform_ws > 0) {
            ret = aclrtMalloc(&uniform_ws_addr, uniform_ws, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(fmt::format("Exponential: malloc uniform ws failed, error={}", ret));
            }
        }
        ret = aclnnInplaceUniform(uniform_ws_addr, uniform_ws, uniform_exec, stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: uniform compute failed, error={}", ret));
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: sync uniform failed, error={}", ret));
        }

        // 7. 计算1 - U
        ret = aclnnForeachSubScalarGetWorkspaceSize(
            x_list, scalar_one.tensorPtr, out_list, &sub_ws, &sub_exec
        );
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: get sub ws failed, error={}", ret));
        }
        if (sub_ws > 0) {
            ret = aclrtMalloc(&sub_ws_addr, sub_ws, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(fmt::format("Exponential: malloc sub ws failed, error={}", ret));
            }
        }
        ret = aclnnForeachSubScalar(sub_ws_addr, sub_ws, sub_exec, stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: sub compute failed, error={}", ret));
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: sync sub failed, error={}", ret));
        }

        // 8. 计算log(1 - U)（更新x_list为result）
        aclDestroyTensorList(x_list);  // 销毁旧列表
        x_tensor_array[0] = result.tensorPtr;  // 更新数组内容
        x_list = aclCreateTensorList(x_tensor_array, 1);  // 重建列表
        if (x_list == nullptr) {
            throw std::runtime_error("Exponential: recreate x_list for log failed");
        }

        ret = aclnnLogGetWorkspaceSize(result.tensorPtr, result.tensorPtr, &log_ws, &log_exec);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: get log ws failed, error={}", ret));
        }
        if (log_ws > 0) {
            ret = aclrtMalloc(&log_ws_addr, log_ws, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(fmt::format("Exponential: malloc log ws failed, error={}", ret));
            }
        }
        ret = aclnnLog(log_ws_addr, log_ws, log_exec, stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: log compute failed, error={}", ret));
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: sync log failed, error={}", ret));
        }

        // 9. 计算 -scale * log(1 - U)
        ret = aclnnForeachMulScalarGetWorkspaceSize(
            x_list, scalar_neg_scale.tensorPtr, out_list, &mul_ws, &mul_exec
        );
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: get mul ws failed, error={}", ret));
        }
        if (mul_ws > 0) {
            ret = aclrtMalloc(&mul_ws_addr, mul_ws, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(fmt::format("Exponential: malloc mul ws failed, error={}", ret));
            }
        }
        ret = aclnnForeachMulScalar(mul_ws_addr, mul_ws, mul_exec, stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: mul compute failed, error={}", ret));
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Exponential: sync mul failed, error={}", ret));
        }

        // 10. 正常流程释放资源
        aclrtFree(mul_ws_addr);
        aclrtFree(log_ws_addr);
        aclrtFree(sub_ws_addr);
        aclrtFree(uniform_ws_addr);
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        aclrtDestroyStream(stream);

        return result;
    } catch (const std::exception& e) {
        // 异常流程释放资源
        aclrtFree(mul_ws_addr);
        aclrtFree(log_ws_addr);
        aclrtFree(sub_ws_addr);
        aclrtFree(uniform_ws_addr);
        if (x_list != nullptr) aclDestroyTensorList(x_list);
        if (out_list != nullptr) aclDestroyTensorList(out_list);
        if (stream != nullptr) aclrtDestroyStream(stream);
        throw;  // 重新抛出异常
    }
}


NPUArray Geometric(float p, const std::vector<int64_t>& size) {
    // 业务参数校验
    if (p <= 0.0f || p > 1.0f) {
        throw std::runtime_error("Geometric: p must be in (0, 1]");
    }
    
    // 特殊处理p=1：几何分布样本恒为1
    if (p == 1.0f) {
        NPUArray result(size, ACL_INT32);
        int32_t host_one = 1;
        void* result_ptr = nullptr;
        
        // 获取数据指针（不做错误判断）
        aclGetRawTensorAddr(result.tensorPtr, &result_ptr);
        
        // 计算张量总字节数（调用NPUArray静态方法GetDataTypeSize）
        int64_t tensor_byte_size = result.tensorSize * NPUArray::GetDataTypeSize(result.aclDtype);
        
        // 拷贝数据（不做错误判断）
        aclrtMemcpy(
            result_ptr, tensor_byte_size,
            &host_one, sizeof(int32_t),
            ACL_MEMCPY_HOST_TO_DEVICE
        );
        return result;
    }

    // 初始化核心张量
    aclDataType float_dtype = ACL_DOUBLE;
    aclDataType int_dtype = ACL_INT32;
    NPUArray U(size, float_dtype);                  // 均匀分布样本
    NPUArray one_minus_U(size, float_dtype);        // 1 - U
    NPUArray log1mU(size, float_dtype);             // log(1 - U)
    NPUArray div_result(size, float_dtype);         // log1mU / log(1 - p)
    NPUArray floor_result(size, float_dtype);       // floor(div_result)
    NPUArray result(size, int_dtype);               // 最终整数结果

    // 初始化标量张量
    NPUArray scalar_one({}, float_dtype);           // 标量1.0（算1-U）
    NPUArray scalar_log1mp({}, float_dtype);        // 标量log(1-p)（除法用）
    NPUArray scalar_add_one({}, float_dtype);       // 标量1.0（floor后加1用）

    // 资源声明
    uint64_t uniform_ws = 0, sub_ws = 0, log_ws = 0, div_ws = 0, floor_ws = 0, add_ws = 0, cast_ws = 0;
    aclOpExecutor *uniform_exec = nullptr, *sub_exec = nullptr, *log_exec = nullptr, *div_exec = nullptr;
    aclOpExecutor *floor_exec = nullptr, *add_exec = nullptr, *cast_exec = nullptr;
    void *uniform_ws_addr = nullptr, *sub_ws_addr = nullptr, *log_ws_addr = nullptr, *div_ws_addr = nullptr;
    void *floor_ws_addr = nullptr, *add_ws_addr = nullptr, *cast_ws_addr = nullptr;
    aclrtStream stream = nullptr;
    aclTensorList *x_list = nullptr, *out_list = nullptr;
    aclTensor* x_tensor_array[1] = {nullptr};
    aclTensor* out_tensor_array[1] = {nullptr};
    void* scalar_ptr = nullptr;

    try {
        // 初始化标量1.0
        double host_one = 1.0;
        aclGetRawTensorAddr(scalar_one.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &host_one, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);

        // 初始化标量add_one（同样为1.0）
        aclGetRawTensorAddr(scalar_add_one.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &host_one, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);

        // 初始化log(1-p)标量
        double host_log1mp = log(1.0 - static_cast<double>(p));
        aclGetRawTensorAddr(scalar_log1mp.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &host_log1mp, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);

        // 创建执行流
        aclrtCreateStream(&stream);

        // 生成均匀分布U~(0,1)
        aclnnInplaceUniformGetWorkspaceSize(
            U.tensorPtr, 0.0, 1.0, static_cast<uint64_t>(time(nullptr)), 0, &uniform_ws, &uniform_exec
        );

        if (uniform_ws > 0) {
            aclrtMalloc(&uniform_ws_addr, uniform_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnInplaceUniform(uniform_ws_addr, uniform_ws, uniform_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算1 - U
        x_tensor_array[0] = U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = one_minus_U.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachSubScalarGetWorkspaceSize(x_list, scalar_one.tensorPtr, out_list, &sub_ws, &sub_exec);

        if (sub_ws > 0) {
            aclrtMalloc(&sub_ws_addr, sub_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnForeachSubScalar(sub_ws_addr, sub_ws, sub_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算log(1 - U)
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = one_minus_U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = log1mU.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnLogGetWorkspaceSize(one_minus_U.tensorPtr, log1mU.tensorPtr, &log_ws, &log_exec);

        if (log_ws > 0) {
            aclrtMalloc(&log_ws_addr, log_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnLog(log_ws_addr, log_ws, log_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 log(1-U) / log(1-p)
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = log1mU.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = div_result.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachDivScalarGetWorkspaceSize(x_list, scalar_log1mp.tensorPtr, out_list, &div_ws, &div_exec);

        if (div_ws > 0) {
            aclrtMalloc(&div_ws_addr, div_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnForeachDivScalar(div_ws_addr, div_ws, div_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 floor(div_result)
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = div_result.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = floor_result.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnFloorGetWorkspaceSize(div_result.tensorPtr, floor_result.tensorPtr, &floor_ws, &floor_exec);

        if (floor_ws > 0) {
            aclrtMalloc(&floor_ws_addr, floor_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnFloor(floor_ws_addr, floor_ws, floor_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 floor_result + 1
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = floor_result.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = floor_result.tensorPtr;  // 复用floor_result内存
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachAddScalarGetWorkspaceSize(x_list, scalar_add_one.tensorPtr, out_list, &add_ws, &add_exec);

        if (add_ws > 0) {
            aclrtMalloc(&add_ws_addr, add_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnForeachAddScalar(add_ws_addr, add_ws, add_exec, stream);
        aclrtSynchronizeStream(stream);

        // 修正aclnnCastGetWorkspaceSize调用：移除多余的float_dtype和ACL_RT_RND_NEAR（匹配ACL接口文档定义）
        aclnnCastGetWorkspaceSize(
            floor_result.tensorPtr,  // 输入张量（自带输入dtype）
            int_dtype,               // 目标输出dtype
            result.tensorPtr,        // 输出张量
            &cast_ws,                // 工作空间大小输出
            &cast_exec               // 执行器输出
        );

        if (cast_ws > 0) {
            aclrtMalloc(&cast_ws_addr, cast_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        // 修正aclnnCast调用：与GetWorkspaceSize参数逻辑一致
        aclnnCast(cast_ws_addr, cast_ws, cast_exec, stream);
        aclrtSynchronizeStream(stream);

        // 释放资源
        if (uniform_ws_addr != nullptr) aclrtFree(uniform_ws_addr);
        if (sub_ws_addr != nullptr) aclrtFree(sub_ws_addr);
        if (log_ws_addr != nullptr) aclrtFree(log_ws_addr);
        if (div_ws_addr != nullptr) aclrtFree(div_ws_addr);
        if (floor_ws_addr != nullptr) aclrtFree(floor_ws_addr);
        if (add_ws_addr != nullptr) aclrtFree(add_ws_addr);
        if (cast_ws_addr != nullptr) aclrtFree(cast_ws_addr);
        
        if (x_list != nullptr) aclDestroyTensorList(x_list);
        if (out_list != nullptr) aclDestroyTensorList(out_list);
        
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
    }   
    catch (const std::exception& e) {
        // 释放资源
        if (uniform_ws_addr != nullptr) aclrtFree(uniform_ws_addr);
        if (sub_ws_addr != nullptr) aclrtFree(sub_ws_addr);
        if (log_ws_addr != nullptr) aclrtFree(log_ws_addr);
        if (div_ws_addr != nullptr) aclrtFree(div_ws_addr);
        if (floor_ws_addr != nullptr) aclrtFree(floor_ws_addr);
        if (add_ws_addr != nullptr) aclrtFree(add_ws_addr);
        if (cast_ws_addr != nullptr) aclrtFree(cast_ws_addr);
        
        if (x_list != nullptr) aclDestroyTensorList(x_list);
        if (out_list != nullptr) aclDestroyTensorList(out_list);
        
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        throw;
    }

    return result;
}
    

NPUArray Gumbel(double loc, double scale, const std::vector<int64_t>& size) {
    // 1. 业务参数校验：scale必须为正数（Gumbel分布尺度参数要求）
    if (scale <= 0.0) {
        throw std::runtime_error("Gumbel: scale must be greater than 0.0");
    }

    // 2. 初始化核心张量（统一使用ACL_DOUBLE避免ACL_FLOAT64未定义问题）
    const aclDataType dtype = ACL_DOUBLE;
    NPUArray U(size, dtype);                  // 均匀分布样本 U ~ (0,1)
    NPUArray log_U(size, dtype);              // log(U)
    NPUArray neg_log_U(size, dtype);          // -log(U)
    NPUArray log_neg_log_U(size, dtype);      // log(-log(U))
    NPUArray scale_mul(size, dtype);          // scale * log(-log(U))
    NPUArray scale_mul_neg(size, dtype);      // -scale * log(-log(U))
    NPUArray result(size, dtype);             // 最终结果：loc - scale*log(-log(U)) = loc + (-scale*log(-log(U)))

    // 3. 初始化标量张量（存储loc、scale及辅助计算的-1.0）
    NPUArray scalar_loc({}, dtype);           // 标量：loc
    NPUArray scalar_scale({}, dtype);         // 标量：scale
    NPUArray scalar_neg_one({}, dtype);       // 标量：-1.0（用于符号反转）

    // 4. 资源声明（工作空间、执行器、流、张量列表等）
    uint64_t uniform_ws = 0, log1_ws = 0, mul1_ws = 0, log2_ws = 0, mul2_ws = 0, mul3_ws = 0, add_ws = 0;
    aclOpExecutor *uniform_exec = nullptr, *log1_exec = nullptr, *mul1_exec = nullptr, *log2_exec = nullptr;
    aclOpExecutor *mul2_exec = nullptr, *mul3_exec = nullptr, *add_exec = nullptr;
    void* uniform_ws_addr = nullptr, *log1_ws_addr = nullptr, *mul1_ws_addr = nullptr, *log2_ws_addr = nullptr;
    void* mul2_ws_addr = nullptr, *mul3_ws_addr = nullptr, *add_ws_addr = nullptr;
    aclrtStream stream = nullptr;
    aclTensorList *x_list = nullptr, *out_list = nullptr;
    aclTensor* x_tensor_array[1] = {nullptr};
    aclTensor* out_tensor_array[1] = {nullptr};
    void* scalar_ptr = nullptr;

    try {
        // 5. 初始化设备端标量（从主机内存拷贝到设备内存）
        // 5.1 初始化标量：loc
        aclGetRawTensorAddr(scalar_loc.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &loc, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
        // 5.2 初始化标量：scale
        aclGetRawTensorAddr(scalar_scale.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &scale, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
        // 5.3 初始化标量：-1.0
        const double host_neg_one = -1.0;
        aclGetRawTensorAddr(scalar_neg_one.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &host_neg_one, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);

        // 6. 创建执行流（用于异步执行NPU操作）
        const aclError stream_err = aclrtCreateStream(&stream);
        if (stream_err != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Gumbel: Failed to create stream. error: {}", stream_err));
        }

        // 7. 生成均匀分布 U ~ (0, 1)（核心分布生成步骤）
        // 7.1 计算均匀分布工作空间大小
        aclnnInplaceUniformGetWorkspaceSize(
            U.tensorPtr,          // 输出张量（原地生成）
            0.0,                  // 分布下界
            1.0,                  // 分布上界
            static_cast<uint64_t>(time(nullptr)),  // 随机种子（时间戳保证随机性）
            0,                    // 保留参数
            &uniform_ws,          // 输出工作空间大小
            &uniform_exec         // 输出执行器
        );
        // 7.2 分配工作空间（非零才分配）
        if (uniform_ws > 0) {
            const aclError malloc_err = aclrtMalloc(&uniform_ws_addr, uniform_ws, ACL_MEM_MALLOC_HUGE_FIRST);
            if (malloc_err != ACL_SUCCESS) {
                throw std::runtime_error(fmt::format("Gumbel: Malloc uniform workspace failed. error: {}", malloc_err));
            }
        }
        // 7.3 执行均匀分布生成
        aclnnInplaceUniform(uniform_ws_addr, uniform_ws, uniform_exec, stream);
        aclrtSynchronizeStream(stream);  // 同步流确保操作完成

        // 8. 计算 log(U)（Gumbel公式第一步）
        // 8.1 销毁上一轮张量列表（避免内存泄漏）
        if (x_list != nullptr) { aclDestroyTensorList(x_list); x_list = nullptr; }
        if (out_list != nullptr) { aclDestroyTensorList(out_list); out_list = nullptr; }
        // 8.2 绑定输入输出张量
        x_tensor_array[0] = U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        out_tensor_array[0] = log_U.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);
        // 8.3 计算log工作空间大小
        aclnnLogGetWorkspaceSize(U.tensorPtr, log_U.tensorPtr, &log1_ws, &log1_exec);
        // 8.4 分配工作空间
        if (log1_ws > 0) {
            const aclError malloc_err = aclrtMalloc(&log1_ws_addr, log1_ws, ACL_MEM_MALLOC_HUGE_FIRST);
            if (malloc_err != ACL_SUCCESS) {
                throw std::runtime_error(fmt::format("Gumbel: Malloc log(U) workspace failed. error: {}", malloc_err));
            }
        }
        // 8.5 执行log计算
        aclnnLog(log1_ws_addr, log1_ws, log1_exec, stream);
        aclrtSynchronizeStream(stream);

        // 9. 计算 -log(U)（Gumbel公式第二步：符号反转）
        // 9.1 销毁上一轮张量列表
        if (x_list != nullptr) { aclDestroyTensorList(x_list); x_list = nullptr; }
        if (out_list != nullptr) { aclDestroyTensorList(out_list); out_list = nullptr; }
        // 9.2 绑定输入输出张量
        x_tensor_array[0] = log_U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        out_tensor_array[0] = neg_log_U.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);
        // 9.3 计算乘法工作空间大小（log_U * (-1)）
        aclnnForeachMulScalarGetWorkspaceSize(x_list, scalar_neg_one.tensorPtr, out_list, &mul1_ws, &mul1_exec);
        // 9.4 分配工作空间
        if (mul1_ws > 0) {
            const aclError malloc_err = aclrtMalloc(&mul1_ws_addr, mul1_ws, ACL_MEM_MALLOC_HUGE_FIRST);
            if (malloc_err != ACL_SUCCESS) {
                throw std::runtime_error(fmt::format("Gumbel: Malloc -log(U) workspace failed. error: {}", malloc_err));
            }
        }
        // 9.5 执行乘法计算
        aclnnForeachMulScalar(mul1_ws_addr, mul1_ws, mul1_exec, stream);
        aclrtSynchronizeStream(stream);

        // 10. 计算 log(-log(U))（Gumbel公式第三步）
        // 10.1 销毁上一轮张量列表
        if (x_list != nullptr) { aclDestroyTensorList(x_list); x_list = nullptr; }
        if (out_list != nullptr) { aclDestroyTensorList(out_list); out_list = nullptr; }
        // 10.2 绑定输入输出张量
        x_tensor_array[0] = neg_log_U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        out_tensor_array[0] = log_neg_log_U.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);
        // 10.3 计算log工作空间大小
        aclnnLogGetWorkspaceSize(neg_log_U.tensorPtr, log_neg_log_U.tensorPtr, &log2_ws, &log2_exec);
        // 10.4 分配工作空间
        if (log2_ws > 0) {
            const aclError malloc_err = aclrtMalloc(&log2_ws_addr, log2_ws, ACL_MEM_MALLOC_HUGE_FIRST);
            if (malloc_err != ACL_SUCCESS) {
                throw std::runtime_error(fmt::format("Gumbel: Malloc log(-log(U)) workspace failed. error: {}", malloc_err));
            }
        }
        // 10.5 执行log计算
        aclnnLog(log2_ws_addr, log2_ws, log2_exec, stream);
        aclrtSynchronizeStream(stream);

        // 11. 计算 scale * log(-log(U))（Gumbel公式第四步）
        // 11.1 销毁上一轮张量列表
        if (x_list != nullptr) { aclDestroyTensorList(x_list); x_list = nullptr; }
        if (out_list != nullptr) { aclDestroyTensorList(out_list); out_list = nullptr; }
        // 11.2 绑定输入输出张量
        x_tensor_array[0] = log_neg_log_U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        out_tensor_array[0] = scale_mul.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);
        // 11.3 计算乘法工作空间大小（log_neg_log_U * scale）
        aclnnForeachMulScalarGetWorkspaceSize(x_list, scalar_scale.tensorPtr, out_list, &mul2_ws, &mul2_exec);
        // 11.4 分配工作空间
        if (mul2_ws > 0) {
            const aclError malloc_err = aclrtMalloc(&mul2_ws_addr, mul2_ws, ACL_MEM_MALLOC_HUGE_FIRST);
            if (malloc_err != ACL_SUCCESS) {
                throw std::runtime_error(fmt::format("Gumbel: Malloc scale*log(-log(U)) workspace failed. error: {}", malloc_err));
            }
        }
        // 11.5 执行乘法计算
        aclnnForeachMulScalar(mul2_ws_addr, mul2_ws, mul2_exec, stream);
        aclrtSynchronizeStream(stream);

        // 12. 计算 -scale * log(-log(U))（Gumbel公式第五步：符号反转）
        // 12.1 销毁上一轮张量列表
        if (x_list != nullptr) { aclDestroyTensorList(x_list); x_list = nullptr; }
        if (out_list != nullptr) { aclDestroyTensorList(out_list); out_list = nullptr; }
        // 12.2 绑定输入输出张量
        x_tensor_array[0] = scale_mul.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        out_tensor_array[0] = scale_mul_neg.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);
        // 12.3 计算乘法工作空间大小（scale_mul * (-1)）
        aclnnForeachMulScalarGetWorkspaceSize(x_list, scalar_neg_one.tensorPtr, out_list, &mul3_ws, &mul3_exec);
        // 12.4 分配工作空间
        if (mul3_ws > 0) {
            const aclError malloc_err = aclrtMalloc(&mul3_ws_addr, mul3_ws, ACL_MEM_MALLOC_HUGE_FIRST);
            if (malloc_err != ACL_SUCCESS) {
                throw std::runtime_error(fmt::format("Gumbel: Malloc -scale*log(-log(U)) workspace failed. error: {}", malloc_err));
            }
        }
        // 12.5 执行乘法计算
        aclnnForeachMulScalar(mul3_ws_addr, mul3_ws, mul3_exec, stream);
        aclrtSynchronizeStream(stream);

        // 13. 计算最终结果：loc + (-scale*log(-log(U)))（Gumbel公式最终步）
        // 13.1 销毁上一轮张量列表
        if (x_list != nullptr) { aclDestroyTensorList(x_list); x_list = nullptr; }
        if (out_list != nullptr) { aclDestroyTensorList(out_list); out_list = nullptr; }
        // 13.2 绑定输入输出张量
        x_tensor_array[0] = scale_mul_neg.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        out_tensor_array[0] = result.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);
        // 13.3 计算加法工作空间大小（scale_mul_neg + loc）
        aclnnForeachAddScalarGetWorkspaceSize(x_list, scalar_loc.tensorPtr, out_list, &add_ws, &add_exec);
        // 13.4 分配工作空间
        if (add_ws > 0) {
            const aclError malloc_err = aclrtMalloc(&add_ws_addr, add_ws, ACL_MEM_MALLOC_HUGE_FIRST);
            if (malloc_err != ACL_SUCCESS) {
                throw std::runtime_error(fmt::format("Gumbel: Malloc final result workspace failed. error: {}", malloc_err));
            }
        }
        // 13.5 执行加法计算
        aclnnForeachAddScalar(add_ws_addr, add_ws, add_exec, stream);
        aclrtSynchronizeStream(stream);

        // 14. 释放资源（正常流程）
        // 14.1 释放工作空间
        if (uniform_ws_addr != nullptr) { aclrtFree(uniform_ws_addr); }
        if (log1_ws_addr != nullptr) { aclrtFree(log1_ws_addr); }
        if (mul1_ws_addr != nullptr) { aclrtFree(mul1_ws_addr); }
        if (log2_ws_addr != nullptr) { aclrtFree(log2_ws_addr); }
        if (mul2_ws_addr != nullptr) { aclrtFree(mul2_ws_addr); }
        if (mul3_ws_addr != nullptr) { aclrtFree(mul3_ws_addr); }
        if (add_ws_addr != nullptr) { aclrtFree(add_ws_addr); }
        // 14.2 释放张量列表
        if (x_list != nullptr) { aclDestroyTensorList(x_list); }
        if (out_list != nullptr) { aclDestroyTensorList(out_list); }
        // 14.3 释放流
        if (stream != nullptr) { aclrtDestroyStream(stream); }
    }
    catch (const std::exception& e) {
        // 15. 异常流程：释放已分配资源并重新抛出异常
        if (uniform_ws_addr != nullptr) { aclrtFree(uniform_ws_addr); }
        if (log1_ws_addr != nullptr) { aclrtFree(log1_ws_addr); }
        if (mul1_ws_addr != nullptr) { aclrtFree(mul1_ws_addr); }
        if (log2_ws_addr != nullptr) { aclrtFree(log2_ws_addr); }
        if (mul2_ws_addr != nullptr) { aclrtFree(mul2_ws_addr); }
        if (mul3_ws_addr != nullptr) { aclrtFree(mul3_ws_addr); }
        if (add_ws_addr != nullptr) { aclrtFree(add_ws_addr); }

        if (x_list != nullptr) { aclDestroyTensorList(x_list); }
        if (out_list != nullptr) { aclDestroyTensorList(out_list); }

        if (stream != nullptr) { aclrtDestroyStream(stream); }

        throw std::runtime_error(fmt::format("Gumbel: Exception occurred - {}", e.what()));
    }

    return result;
}


NPUArray Laplace(double loc, double scale, const std::vector<int64_t>& size) {
    // 业务参数校验：拉普拉斯分布尺度参数必须为正数
    if (scale <= 0.0) {
        throw std::runtime_error("Laplace: scale must be greater than 0.0");
    }

    // 初始化核心张量（使用ACL_DOUBLE类型）
    const aclDataType dtype = ACL_DOUBLE;
    NPUArray U(size, dtype);                  // 均匀分布样本 U ~ [-0.5, 0.5)
    NPUArray abs_U(size, dtype);              // abs_U = |U|
    NPUArray two_abs_U(size, dtype);          // two_abs_U = 2 * abs_U
    NPUArray one_minus_two_abs(size, dtype);  // one_minus_two_abs = 1 - two_abs_U
    NPUArray log_term(size, dtype);           // log_term = log(one_minus_two_abs)
    NPUArray sign_U(size, dtype);             // sign_U = sign(U)
    NPUArray sign_log(size, dtype);           // sign_log = sign_U * log_term
    NPUArray scale_sign_log(size, dtype);     // scale_sign_log = scale * sign_log
    NPUArray neg_scale_sign_log(size, dtype); // neg_scale_sign_log = -scale_sign_log
    NPUArray result(size, dtype);             // 最终结果：loc - scale_sign_log

    // 初始化标量张量
    NPUArray scalar_loc({}, dtype);           // 位置参数 loc
    NPUArray scalar_scale({}, dtype);         // 尺度参数 scale
    NPUArray scalar_2({}, dtype);             // 标量 2.0
    NPUArray scalar_1({}, dtype);             // 标量 1.0
    NPUArray scalar_neg_1({}, dtype);         // 标量 -1.0

    // 资源声明
    uint64_t uniform_ws = 0, abs_ws = 0, mul1_ws = 0, sub_ws = 0, log_ws = 0;
    uint64_t sign_ws = 0, mul2_ws = 0, mul3_ws = 0, add_ws = 0;
    aclOpExecutor *uniform_exec = nullptr, *abs_exec = nullptr, *mul1_exec = nullptr;
    aclOpExecutor *sub_exec = nullptr, *log_exec = nullptr, *sign_exec = nullptr;
    aclOpExecutor *mul2_exec = nullptr, *mul3_exec = nullptr, *add_exec = nullptr;
    void* uniform_ws_addr = nullptr, *abs_ws_addr = nullptr, *mul1_ws_addr = nullptr;
    void* sub_ws_addr = nullptr, *log_ws_addr = nullptr, *sign_ws_addr = nullptr;
    void* mul2_ws_addr = nullptr, *mul3_ws_addr = nullptr, *add_ws_addr = nullptr;
    aclrtStream stream = nullptr;
    aclTensorList *x_list = nullptr, *out_list = nullptr;
    aclTensor* x_tensor_array[1] = {nullptr};
    aclTensor* out_tensor_array[1] = {nullptr};
    void* scalar_ptr = nullptr;

    try {
        // 初始化设备端标量
        // 标量：loc
        aclGetRawTensorAddr(scalar_loc.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &loc, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
        
        // 标量：scale
        aclGetRawTensorAddr(scalar_scale.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &scale, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
        
        // 标量：2.0
        const double host_2 = 2.0;
        aclGetRawTensorAddr(scalar_2.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &host_2, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
        
        // 标量：1.0
        const double host_1 = 1.0;
        aclGetRawTensorAddr(scalar_1.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &host_1, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
        
        // 标量：-1.0
        const double host_neg_1 = -1.0;
        aclGetRawTensorAddr(scalar_neg_1.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &host_neg_1, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);

        // 创建执行流
        aclrtCreateStream(&stream);

        // 生成均匀分布 U ~ [-0.5, 0.5)
        aclnnInplaceUniformGetWorkspaceSize(
            U.tensorPtr,
            -0.5,                 // 分布下界
            0.5,                  // 分布上界
            static_cast<uint64_t>(time(nullptr)),
            0,
            &uniform_ws,
            &uniform_exec
        );

        if (uniform_ws > 0) {
            aclrtMalloc(&uniform_ws_addr, uniform_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnInplaceUniform(uniform_ws_addr, uniform_ws, uniform_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 abs_U = |U|
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = abs_U.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnAbsGetWorkspaceSize(U.tensorPtr, abs_U.tensorPtr, &abs_ws, &abs_exec);

        if (abs_ws > 0) {
            aclrtMalloc(&abs_ws_addr, abs_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnAbs(abs_ws_addr, abs_ws, abs_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 two_abs_U = 2 * abs_U
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = abs_U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = two_abs_U.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachMulScalarGetWorkspaceSize(x_list, scalar_2.tensorPtr, out_list, &mul1_ws, &mul1_exec);

        if (mul1_ws > 0) {
            aclrtMalloc(&mul1_ws_addr, mul1_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnForeachMulScalar(mul1_ws_addr, mul1_ws, mul1_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 one_minus_two_abs = 1 - two_abs_U
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = two_abs_U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = one_minus_two_abs.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachSubScalarGetWorkspaceSize(x_list, scalar_1.tensorPtr, out_list, &sub_ws, &sub_exec);

        if (sub_ws > 0) {
            aclrtMalloc(&sub_ws_addr, sub_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnForeachSubScalar(sub_ws_addr, sub_ws, sub_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 log_term = log(one_minus_two_abs)
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = one_minus_two_abs.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = log_term.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnLogGetWorkspaceSize(one_minus_two_abs.tensorPtr, log_term.tensorPtr, &log_ws, &log_exec);

        if (log_ws > 0) {
            aclrtMalloc(&log_ws_addr, log_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnLog(log_ws_addr, log_ws, log_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 sign_U = sign(U)
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = sign_U.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnSignGetWorkspaceSize(U.tensorPtr, sign_U.tensorPtr, &sign_ws, &sign_exec);

        if (sign_ws > 0) {
            aclrtMalloc(&sign_ws_addr, sign_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnSign(sign_ws_addr, sign_ws, sign_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 sign_log = sign_U * log_term
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = sign_U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = sign_log.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachMulScalarGetWorkspaceSize(x_list, log_term.tensorPtr, out_list, &mul2_ws, &mul2_exec);

        if (mul2_ws > 0) {
            aclrtMalloc(&mul2_ws_addr, mul2_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnForeachMulScalar(mul2_ws_addr, mul2_ws, mul2_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 scale_sign_log = scale * sign_log
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = sign_log.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = scale_sign_log.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachMulScalarGetWorkspaceSize(x_list, scalar_scale.tensorPtr, out_list, &mul3_ws, &mul3_exec);

        if (mul3_ws > 0) {
            aclrtMalloc(&mul3_ws_addr, mul3_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnForeachMulScalar(mul3_ws_addr, mul3_ws, mul3_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 neg_scale_sign_log = -scale_sign_log
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = scale_sign_log.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = neg_scale_sign_log.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachMulScalarGetWorkspaceSize(x_list, scalar_neg_1.tensorPtr, out_list, &mul3_ws, &mul3_exec);

        aclnnForeachMulScalar(mul3_ws_addr, mul3_ws, mul3_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算最终结果 result = loc + neg_scale_sign_log
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = neg_scale_sign_log.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = result.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachAddScalarGetWorkspaceSize(x_list, scalar_loc.tensorPtr, out_list, &add_ws, &add_exec);

        if (add_ws > 0) {
            aclrtMalloc(&add_ws_addr, add_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnForeachAddScalar(add_ws_addr, add_ws, add_exec, stream);
        aclrtSynchronizeStream(stream);

        // 释放资源
        if (uniform_ws_addr != nullptr) aclrtFree(uniform_ws_addr);
        if (abs_ws_addr != nullptr) aclrtFree(abs_ws_addr);
        if (mul1_ws_addr != nullptr) aclrtFree(mul1_ws_addr);
        if (sub_ws_addr != nullptr) aclrtFree(sub_ws_addr);
        if (log_ws_addr != nullptr) aclrtFree(log_ws_addr);
        if (sign_ws_addr != nullptr) aclrtFree(sign_ws_addr);
        if (mul2_ws_addr != nullptr) aclrtFree(mul2_ws_addr);
        if (mul3_ws_addr != nullptr) aclrtFree(mul3_ws_addr);
        if (add_ws_addr != nullptr) aclrtFree(add_ws_addr);
        
        if (x_list != nullptr) aclDestroyTensorList(x_list);
        if (out_list != nullptr) aclDestroyTensorList(out_list);
        
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
    }   
    catch (const std::exception& e) {
        // 释放资源
        if (uniform_ws_addr != nullptr) aclrtFree(uniform_ws_addr);
        if (abs_ws_addr != nullptr) aclrtFree(abs_ws_addr);
        if (mul1_ws_addr != nullptr) aclrtFree(mul1_ws_addr);
        if (sub_ws_addr != nullptr) aclrtFree(sub_ws_addr);
        if (log_ws_addr != nullptr) aclrtFree(log_ws_addr);
        if (sign_ws_addr != nullptr) aclrtFree(sign_ws_addr);
        if (mul2_ws_addr != nullptr) aclrtFree(mul2_ws_addr);
        if (mul3_ws_addr != nullptr) aclrtFree(mul3_ws_addr);
        if (add_ws_addr != nullptr) aclrtFree(add_ws_addr);
        
        if (x_list != nullptr) aclDestroyTensorList(x_list);
        if (out_list != nullptr) aclDestroyTensorList(out_list);
        
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        throw;
    }

    return result;
}


NPUArray Logistic(double loc, double scale, const std::vector<int64_t>& size) {
    // 业务参数校验：Logistic分布尺度参数必须为正数
    if (scale <= 0.0) {
        throw std::runtime_error("Logistic: scale must be greater than 0.0");
    }

    // 初始化核心张量（使用ACL_DOUBLE类型）
    const aclDataType dtype = ACL_DOUBLE;
    NPUArray U(size, dtype);                  // 均匀分布样本 U ~ (0, 1)
    NPUArray one_minus_U(size, dtype);        // 1 - U
    NPUArray ratio(size, dtype);              // U / (1 - U)
    NPUArray log_ratio(size, dtype);          // log(ratio)
    NPUArray scale_log(size, dtype);          // scale * log_ratio
    NPUArray result(size, dtype);             // 最终结果：loc + scale_log

    // 初始化标量张量
    NPUArray scalar_loc({}, dtype);           // 位置参数 loc
    NPUArray scalar_scale({}, dtype);         // 尺度参数 scale
    NPUArray scalar_one({}, dtype);           // 标量 1.0

    // 资源声明
    uint64_t uniform_ws = 0, sub_ws = 0, div_ws = 0, log_ws = 0, mul_ws = 0, add_ws = 0;
    aclOpExecutor *uniform_exec = nullptr, *sub_exec = nullptr, *div_exec = nullptr;
    aclOpExecutor *log_exec = nullptr, *mul_exec = nullptr, *add_exec = nullptr;
    void* uniform_ws_addr = nullptr, *sub_ws_addr = nullptr, *div_ws_addr = nullptr;
    void* log_ws_addr = nullptr, *mul_ws_addr = nullptr, *add_ws_addr = nullptr;
    aclrtStream stream = nullptr;
    aclTensorList *x_list = nullptr, *out_list = nullptr;
    aclTensor* x_tensor_array[1] = {nullptr};
    aclTensor* out_tensor_array[1] = {nullptr};
    void* scalar_ptr = nullptr;

    try {
        // 初始化设备端标量
        // 标量：loc
        aclGetRawTensorAddr(scalar_loc.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &loc, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
        
        // 标量：scale
        aclGetRawTensorAddr(scalar_scale.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &scale, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
        
        // 标量：1.0
        const double host_one = 1.0;
        aclGetRawTensorAddr(scalar_one.tensorPtr, &scalar_ptr);
        aclrtMemcpy(scalar_ptr, sizeof(double), &host_one, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);

        // 创建执行流
        aclrtCreateStream(&stream);

        // 生成均匀分布 U ~ (0, 1)
        aclnnInplaceUniformGetWorkspaceSize(
            U.tensorPtr,
            0.0,                  // 分布下界
            1.0,                  // 分布上界
            static_cast<uint64_t>(time(nullptr)),
            0,
            &uniform_ws,
            &uniform_exec
        );

        if (uniform_ws > 0) {
            aclrtMalloc(&uniform_ws_addr, uniform_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnInplaceUniform(uniform_ws_addr, uniform_ws, uniform_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 one_minus_U = 1 - U
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = one_minus_U.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachSubScalarGetWorkspaceSize(x_list, scalar_one.tensorPtr, out_list, &sub_ws, &sub_exec);

        if (sub_ws > 0) {
            aclrtMalloc(&sub_ws_addr, sub_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnForeachSubScalar(sub_ws_addr, sub_ws, sub_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 ratio = U / (1 - U)
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = U.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = ratio.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachDivScalarGetWorkspaceSize(x_list, one_minus_U.tensorPtr, out_list, &div_ws, &div_exec);

        if (div_ws > 0) {
            aclrtMalloc(&div_ws_addr, div_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnForeachDivScalar(div_ws_addr, div_ws, div_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 log_ratio = log(ratio)
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = ratio.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = log_ratio.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnLogGetWorkspaceSize(ratio.tensorPtr, log_ratio.tensorPtr, &log_ws, &log_exec);

        if (log_ws > 0) {
            aclrtMalloc(&log_ws_addr, log_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnLog(log_ws_addr, log_ws, log_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算 scale_log = scale * log_ratio
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = log_ratio.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = scale_log.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachMulScalarGetWorkspaceSize(x_list, scalar_scale.tensorPtr, out_list, &mul_ws, &mul_exec);

        if (mul_ws > 0) {
            aclrtMalloc(&mul_ws_addr, mul_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnForeachMulScalar(mul_ws_addr, mul_ws, mul_exec, stream);
        aclrtSynchronizeStream(stream);

        // 计算最终结果 result = loc + scale_log
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;
        
        x_tensor_array[0] = scale_log.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        
        out_tensor_array[0] = result.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        aclnnForeachAddScalarGetWorkspaceSize(x_list, scalar_loc.tensorPtr, out_list, &add_ws, &add_exec);

        if (add_ws > 0) {
            aclrtMalloc(&add_ws_addr, add_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        aclnnForeachAddScalar(add_ws_addr, add_ws, add_exec, stream);
        aclrtSynchronizeStream(stream);

        // 释放资源
        if (uniform_ws_addr != nullptr) aclrtFree(uniform_ws_addr);
        if (sub_ws_addr != nullptr) aclrtFree(sub_ws_addr);
        if (div_ws_addr != nullptr) aclrtFree(div_ws_addr);
        if (log_ws_addr != nullptr) aclrtFree(log_ws_addr);
        if (mul_ws_addr != nullptr) aclrtFree(mul_ws_addr);
        if (add_ws_addr != nullptr) aclrtFree(add_ws_addr);
        
        if (x_list != nullptr) aclDestroyTensorList(x_list);
        if (out_list != nullptr) aclDestroyTensorList(out_list);
        
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
    }   
    catch (const std::exception& e) {
        // 释放资源
        if (uniform_ws_addr != nullptr) aclrtFree(uniform_ws_addr);
        if (sub_ws_addr != nullptr) aclrtFree(sub_ws_addr);
        if (div_ws_addr != nullptr) aclrtFree(div_ws_addr);
        if (log_ws_addr != nullptr) aclrtFree(log_ws_addr);
        if (mul_ws_addr != nullptr) aclrtFree(mul_ws_addr);
        if (add_ws_addr != nullptr) aclrtFree(add_ws_addr);
        
        if (x_list != nullptr) aclDestroyTensorList(x_list);
        if (out_list != nullptr) aclDestroyTensorList(out_list);
        
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        throw;
    }

    return result;
}


NPUArray Lognormal(float mean, float sigma, const std::vector<int64_t>& size) {
    // 业务参数校验：对数正态分布的标准差sigma必须为正数
    if (sigma <= 0.0) {
        throw std::runtime_error("Lognormal: sigma must be greater than 0.0");
    }

    // 初始化核心张量（使用ACL_DOUBLE类型，保证计算精度）
    const aclDataType dtype = ACL_DOUBLE;
    NPUArray normal_sample(size, dtype);  // 正态分布样本 N(mean, sigma)
    NPUArray exp_sample(size, dtype);     // 最终结果：exp(normal_sample)（对数正态样本）

    // 资源声明（工作空间、执行器、流、张量列表等）
    uint64_t normal_ws = 0, exp_ws = 0;
    aclOpExecutor *normal_exec = nullptr, *exp_exec = nullptr;
    void* normal_ws_addr = nullptr, *exp_ws_addr = nullptr;
    aclrtStream stream = nullptr;
    aclTensorList *x_list = nullptr, *out_list = nullptr;
    aclTensor* x_tensor_array[1] = {nullptr};
    aclTensor* out_tensor_array[1] = {nullptr};

    try {
        // 创建执行流（用于NPU异步操作）
        aclrtCreateStream(&stream);

        // 步骤1：生成正态分布样本 N(mean, sigma)
        aclnnNormalFloatFloatGetWorkspaceSize(
            mean,                     // 正态分布均值
            sigma,                    // 正态分布标准差
            static_cast<uint64_t>(time(nullptr)),  // 随机种子（时间戳保证随机性）
            0,                        // 保留参数
            normal_sample.tensorPtr,  // 输出张量（原地生成正态样本）
            &normal_ws,               // 正态分布计算所需工作空间大小
            &normal_exec              // 正态分布执行器
        );

        // 分配正态分布工作空间（非零才分配，避免无效内存申请）
        if (normal_ws > 0) {
            aclrtMalloc(&normal_ws_addr, normal_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        // 执行正态分布样本生成，同步流确保操作完成
        aclnnNormalFloatFloat(normal_ws_addr, normal_ws, normal_exec, stream);
        aclrtSynchronizeStream(stream);

        // 步骤2：计算 exp(normal_sample)，得到对数正态样本
        // 销毁上一步张量列表（避免内存泄漏，保持资源整洁）
        aclDestroyTensorList(x_list);
        aclDestroyTensorList(out_list);
        x_list = nullptr;
        out_list = nullptr;

        // 绑定输入（正态样本）和输出（exp结果）张量
        x_tensor_array[0] = normal_sample.tensorPtr;
        x_list = aclCreateTensorList(x_tensor_array, 1);
        out_tensor_array[0] = exp_sample.tensorPtr;
        out_list = aclCreateTensorList(out_tensor_array, 1);

        // 计算指数运算所需工作空间大小
        aclnnExpGetWorkspaceSize(normal_sample.tensorPtr, exp_sample.tensorPtr, &exp_ws, &exp_exec);

        // 分配指数运算工作空间
        if (exp_ws > 0) {
            aclrtMalloc(&exp_ws_addr, exp_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        // 执行指数运算，同步流确保结果有效
        aclnnExp(exp_ws_addr, exp_ws, exp_exec, stream);
        aclrtSynchronizeStream(stream);

        // 释放资源（正常流程）
        if (normal_ws_addr != nullptr) aclrtFree(normal_ws_addr);
        if (exp_ws_addr != nullptr) aclrtFree(exp_ws_addr);
        
        if (x_list != nullptr) aclDestroyTensorList(x_list);
        if (out_list != nullptr) aclDestroyTensorList(out_list);
        
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
    }   
    catch (const std::exception& e) {
        // 异常流程：释放已分配资源，避免内存泄漏
        if (normal_ws_addr != nullptr) aclrtFree(normal_ws_addr);
        if (exp_ws_addr != nullptr) aclrtFree(exp_ws_addr);
        
        if (x_list != nullptr) aclDestroyTensorList(x_list);
        if (out_list != nullptr) aclDestroyTensorList(out_list);
        
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        throw;  // 重新抛出异常，保留原始错误信息
    }

    return exp_sample;
}


NPUArray Multinomial(int64_t n, const NPUArray& pvals, bool replacement, const std::vector<int64_t>& size, int64_t offset = 0) {
    // 业务参数校验：多项式分布核心参数合法性检查
    if (n <= 0) {
        throw std::runtime_error("Multinomial: number of samples 'n' must be greater than 0");
    }
    if (pvals.shape.empty()) {
        throw std::runtime_error("Multinomial: probability array 'pvals' cannot be empty");
    }
    // 校验pvals数据类型（需为浮点型，适配aclnnMultinomial输入要求）
    const aclDataType pvals_dtype = pvals.aclDtype;
    if (pvals_dtype != ACL_FLOAT && pvals_dtype != ACL_DOUBLE) {
        throw std::runtime_error("Multinomial: 'pvals' must be of float or double type");
    }

    // 初始化结果张量（多项式分布输出为计数，用整数型ACL_INT32）
    const aclDataType result_dtype = ACL_INT32;
    NPUArray result(size, result_dtype);

    // 资源声明：工作空间、执行器、流
    uint64_t multinom_ws = 0;
    aclOpExecutor *multinom_exec = nullptr;
    void* multinom_ws_addr = nullptr;
    aclrtStream stream = nullptr;

    try {
        // 1. 创建NPU执行流
        aclrtCreateStream(&stream);

        // 2. 获取多项式分布计算所需工作空间大小（使用文档定义的正确参数顺序）
        aclnnMultinomialGetWorkspaceSize(
            pvals.tensorPtr,       // 输入概率分布张量
            n,                     // 样本数量
            replacement,           // 是否有放回抽样
            static_cast<uint64_t>(time(nullptr)),  // 随机种子
            offset,                // 偏移量参数
            result.tensorPtr,      // 输出结果张量
            &multinom_ws,          // 工作空间大小输出
            &multinom_exec         // 执行器输出
        );

        // 3. 分配工作空间（非零才分配，避免无效内存申请）
        if (multinom_ws > 0) {
            aclrtMalloc(&multinom_ws_addr, multinom_ws, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        // 4. 执行多项式分布抽样计算，同步流确保结果有效
        aclnnMultinomial(
            multinom_ws_addr,      // 工作空间地址
            multinom_ws,           // 工作空间大小
            multinom_exec,         // 执行器
            stream                 // 执行流
        );
        aclrtSynchronizeStream(stream);

        // 5. 正常流程：释放所有资源
        if (multinom_ws_addr != nullptr) aclrtFree(multinom_ws_addr);
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
    }
    catch (const std::exception& e) {
        // 6. 异常流程：释放已分配资源，避免内存泄漏
        if (multinom_ws_addr != nullptr) aclrtFree(multinom_ws_addr);
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        throw;  // 重新抛出异常，保留原始错误信息
    }

    return result;
}
