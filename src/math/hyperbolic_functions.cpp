#include <asnumpy/math/hyperbolic_functions.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_sinh.h>
#include <aclnnop/aclnn_cosh.h>
#include <aclnnop/aclnn_tanh.h>
#include <aclnnop/aclnn_asinh.h>
#include <aclnnop/aclnn_acosh.h>
#include <aclnnop/aclnn_atanh.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

/**
 * @brief Compute element-wise hyperbolic sine.
 * 
 * Equivalent to numpy.sinh(x), calculates sinh(x) = (e^x - e^(-x))/2 for each element.
 * 
 * @param x NPUArray, input array
 * @return NPUArray Element-wise hyperbolic sine of x
 */
NPUArray Sinh(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnSinhGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Sinh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Sinh: malloc workspace failed, error={}", error));
        }
    }

    // 执行双曲正弦计算
    error = aclnnSinh(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Sinh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise hyperbolic cosine.
 * 
 * Equivalent to numpy.cosh(x), calculates cosh(x) = (e^x + e^(-x))/2 for each element.
 * 
 * @param x NPUArray, input array
 * @return NPUArray Element-wise hyperbolic cosine of x
 */
NPUArray Cosh(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnCoshGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Cosh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Cosh: malloc workspace failed, error={}", error));
        }
    }

    // 执行双曲余弦计算
    error = aclnnCosh(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Cosh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise hyperbolic tangent.
 * 
 * Equivalent to numpy.tanh(x), calculates tanh(x) = sinh(x)/cosh(x) for each element.
 * 
 * @param x NPUArray, input array
 * @return NPUArray Element-wise hyperbolic tangent of x
 */
NPUArray Tanh(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnTanhGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Tanh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Tanh: malloc workspace failed, error={}", error));
        }
    }

    // 执行双曲正切计算
    error = aclnnTanh(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Tanh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise inverse hyperbolic sine.
 * 
 * Equivalent to numpy.arcsinh(x), calculates arcsinh(x) = ln(x + √(x² + 1)) for each element.
 * 
 * @param x NPUArray, input array
 * @return NPUArray Element-wise inverse hyperbolic sine of x
 */
NPUArray Arcsinh(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAsinhGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arcsinh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Arcsinh: malloc workspace failed, error={}", error));
        }
    }

    // 执行反双曲正弦计算
    error = aclnnAsinh(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arcsinh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise inverse hyperbolic cosine.
 * 
 * Equivalent to numpy.arccosh(x), calculates arccosh(x) = ln(x + √(x² - 1)) for x ≥ 1.
 * 
 * @param x NPUArray, input array (must contain values ≥ 1)
 * @return NPUArray Element-wise inverse hyperbolic cosine of x
 */
NPUArray Arccosh(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAcoshGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arccosh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Arccosh: malloc workspace failed, error={}", error));
        }
    }

    // 执行反双曲余弦计算
    error = aclnnAcosh(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arccosh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}


/**
 * @brief Compute element-wise inverse hyperbolic tangent.
 * 
 * Equivalent to numpy.arctanh(x), calculates arctanh(x) = 0.5*ln((1+x)/(1-x)) for |x| < 1.
 * 
 * @param x NPUArray, input array (must contain values with absolute value < 1)
 * @return NPUArray Element-wise inverse hyperbolic tangent of x
 */
NPUArray Arctanh(const NPUArray& x) {
    // 初始化结果数组（形状和数据类型与输入一致）
    auto shape = x.shape;
    auto dtype = x.dtype;
    NPUArray result(shape, dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAtanhGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arctanh: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Arctanh: malloc workspace failed, error={}", error));
        }
    }

    // 执行反双曲正切计算
    error = aclnnAtanh(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arctanh: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}