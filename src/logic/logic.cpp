#include <asnumpy/logic/logic.hpp>
#include <fmt/base.h>

#include <aclnnop/aclnn_all.h>
#include <aclnnop/aclnn_any.h>
#include <aclnnop/aclnn_isfinite.h>
#include <aclnnop/aclnn_is_inf.h>
#include <aclnnop/aclnn_isneginf.h>
#include <aclnnop/aclnn_isposinf.h>
#include <aclnnop/aclnn_logical_and.h>
#include <aclnnop/aclnn_logical_or.h>
#include <aclnnop/aclnn_logical_not.h>
#include <aclnnop/aclnn_logical_xor.h>
#include <aclnnop/aclnn_gt_tensor.h>
#include <aclnnop/aclnn_gt_scalar.h>
#include <aclnnop/aclnn_ge_tensor.h>
#include <aclnnop/aclnn_ge_scalar.h>
#include <aclnnop/aclnn_lt_scalar.h>
#include <aclnnop/aclnn_lt_tensor.h>
#include <aclnnop/aclnn_le_tensor.h>
#include <aclnnop/aclnn_le_scalar.h>
#include <aclnnop/aclnn_equal.h>
#include <aclnnop/aclnn_ne_scalar.h>
#include <aclnnop/aclnn_ne_tensor.h>


/**
 * @brief Reduce array by logical AND operation over specified dimensions.
 * 
 * Equivalent to numpy.all(x, axis=dim, keepdims=keepdims), returns True if all elements along the specified dimensions are True.
 * 
 * @param x NPUArray, input array (boolean type: ACL_BOOL)
 * @param dim std::vector<int64_t>, dimensions to reduce
 * @param keepdims bool, whether to keep reduced dimensions with size 1
 * @return NPUArray Reduced boolean array
 */
NPUArray All(const NPUArray& x, const std::vector<int64_t>& dim, bool keepdims) {
    auto result = NPUArray({}, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;
    auto error = ACL_SUCCESS;

    aclIntArray* aclDim = aclCreateIntArray(dim.data(), dim.size());
    if (aclDim == nullptr) {
        throw std::runtime_error("All: failed to create aclIntArray");
    }

    error = aclnnAllGetWorkspaceSize(
        x.tensorPtr,
        aclDim,
        keepdims,
        result.tensorPtr,
        &workspaceSize,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("All: get workspace size failed, error={}", error));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("All: malloc workspace failed, error={}", error));
        }
    }

    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr); 
        throw std::runtime_error(fmt::format("All: create stream failed, error={}", error));
    }

    error = aclnnAll(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("All: computation failed, error={}", error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("All: sync device failed, error={}", error));
    }

    aclrtDestroyStream(stream);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);

    return result;
}


/**
 * @brief Reduce array by logical OR operation over specified dimensions.
 * 
 * Equivalent to numpy.any(x, axis=dim, keepdims=keepdims), returns True if any element along the specified dimensions is True.
 * 
 * @param x NPUArray, input array (boolean type: ACL_BOOL)
 * @param dim std::vector<int64_t>, dimensions to reduce
 * @param keepdims bool, whether to keep reduced dimensions with size 1
 * @return NPUArray Reduced boolean array
 */
NPUArray Any(const NPUArray& x, const std::vector<int64_t>& dim, bool keepdims) {
    auto result = NPUArray({}, ACL_BOOL);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;
    auto error = ACL_SUCCESS;

    aclIntArray* aclDim = aclCreateIntArray(dim.data(), dim.size());
    if (aclDim == nullptr) {
        throw std::runtime_error("Any: failed to create aclIntArray");
    }

    error = aclnnAnyGetWorkspaceSize(
        x.tensorPtr,
        aclDim,
        keepdims,
        result.tensorPtr,
        &workspaceSize,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Any: get workspace size failed, error={}", error));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Any: malloc workspace failed, error={}", error));
        }
    }

    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr); 
        throw std::runtime_error(fmt::format("Any: create stream failed, error={}", error));
    }

    error = aclnnAny(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("Any: computation failed, error={}", error));
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("Any: sync device failed, error={}", error));
    }

    aclrtDestroyStream(stream);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);

    return result;
}


/**
 * @brief Test element-wise for finite values (not inf, -inf, or NaN).
 * 
 * Equivalent to numpy.isfinite(x), returns True for elements that are finite (not inf, -inf, or NaN).
 * Only supports floating-point input arrays.
 * 
 * @param x NPUArray, input array (floating-point type: ACL_FLOAT/ACL_DOUBLE)
 * @return NPUArray Boolean array (same shape as x), True if element is finite
 */
NPUArray IsFinite(const NPUArray& x) {
    // 输入类型校验
    if (x.aclDtype != ACL_FLOAT && x.aclDtype != ACL_DOUBLE) {
        throw std::invalid_argument(fmt::format(
            "IsFinite: unsupported dtype={}, only float/double allowed (finite check requires floating-point)", x.dtype));
    }

    // 初始化布尔型结果
    auto result = NPUArray(x.shape, ACL_BOOL);
    
    // 变量声明
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;
    auto error = ACL_SUCCESS;

    // 1. 获取工作空间大小与执行器
    error = aclnnIsFiniteGetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("IsFinite: get workspace size failed, error={}", error));
    }

    // 2. 分配工作空间
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyAclOpExecutor(executor);  // 释放已创建执行器，避免资源泄漏
            throw std::runtime_error(fmt::format("IsFinite: malloc workspace failed, error={}", error));
        }
    }

    // 3. 创建自定义执行流
    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);  // 释放已分配空间
        aclDestroyAclOpExecutor(executor);                          // 释放执行器
        throw std::runtime_error(fmt::format("IsFinite: create stream failed, error={}", error));
    }

    // 4. 执行有限值判断计算（排除inf、-inf、NaN，标记有限值为True）
    error = aclnnIsFinite(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);       // 释放流
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);  // 释放空间
        aclDestroyAclOpExecutor(executor);                          // 释放执行器
        throw std::runtime_error(fmt::format("IsFinite: computation failed, error={}", error));
    }

    // 5. 同步设备（确保计算完全完成，避免结果未就绪，与已有函数同步逻辑一致）
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);       // 释放流
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);  // 释放空间
        aclDestroyAclOpExecutor(executor);                          // 释放执行器
        throw std::runtime_error(fmt::format("IsFinite: sync device failed, error={}", error));
    }

    // 6. 释放所有资源（按“流→工作空间→执行器”顺序，避免资源依赖导致的释放异常）
    aclrtDestroyStream(stream);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    aclDestroyAclOpExecutor(executor);

    return result;
}


/**
 * @brief Test element-wise for infinity (positive or negative).
 * 
 * Equivalent to numpy.isinf(x), returns True for elements that are infinite (inf or -inf).
 * Only supports floating-point input arrays.
 * 
 * @param x NPUArray, input array (floating-point type: ACL_FLOAT/ACL_DOUBLE)
 * @return NPUArray Boolean array (same shape as x), True if element is infinite
 */
NPUArray IsInf(const NPUArray& x) {
    // 初始化布尔型结果
    auto result = NPUArray(x.shape, ACL_BOOL);
    
    // 变量声明
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;
    auto error = ACL_SUCCESS;

    // 1. 获取工作空间大小与执行器（调用ACL无穷大判断接口）
    error = aclnnIsInfGetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("IsInf: get workspace size failed, error={}", error));
    }

    // 2. 分配工作空间
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyAclOpExecutor(executor);  // 释放已创建执行器
            throw std::runtime_error(fmt::format("IsInf: malloc workspace failed, error={}", error));
        }
    }

    // 3. 创建自定义执行流
    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);  // 释放已分配空间
        aclDestroyAclOpExecutor(executor);                          // 释放执行器
        throw std::runtime_error(fmt::format("IsInf: create stream failed, error={}", error));
    }

    // 4. 执行无穷大判断计算
    error = aclnnIsInf(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);       // 释放流
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);  // 释放空间
        aclDestroyAclOpExecutor(executor);                          // 释放执行器
        throw std::runtime_error(fmt::format("IsInf: computation failed, error={}", error));
    }

    // 5. 同步设备（与Cumsum同步同步逻辑一致）
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);       // 释放流
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);  // 释放空间
        aclDestroyAclOpExecutor(executor);                          // 释放执行器
        throw std::runtime_error(fmt::format("IsInf: sync device failed, error={}", error));
    }

    // 6. 释放所有资源（顺序：流→工作空间→执行器）
    aclrtDestroyStream(stream);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    aclDestroyAclOpExecutor(executor);

    return result;
}


/**
 * @brief Test element-wise for negative infinity (-inf).
 * 
 * Equivalent to numpy.isneginf(x), returns True for elements that are negative infinite (-inf).
 * Only supports floating-point input arrays.
 * 
 * @param x NPUArray, input array (floating-point type: ACL_FLOAT/ACL_DOUBLE)
 * @return NPUArray Boolean array (same shape as x), True if element is negative infinite
 */
NPUArray IsNegInf(const NPUArray& x) {
    // 初始化布尔型结果
    auto result = NPUArray(x.shape, ACL_BOOL);
    
    // 变量声明
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;
    auto error = ACL_SUCCESS;

    // 1. 获取工作空间大小与执行器（调用ACL负无穷判断接口）
    error = aclnnIsNegInfGetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("IsNegInf: get workspace size failed, error={}", error));
    }

    // 2. 分配工作空间
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyAclOpExecutor(executor);
            throw std::runtime_error(fmt::format("IsNegInf: malloc workspace failed, error={}", error));
        }
    }

    // 3. 创建自定义执行流
    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        aclDestroyAclOpExecutor(executor);
        throw std::runtime_error(fmt::format("IsNegInf: create stream failed, error={}", error));
    }

    // 4. 执行负无穷判断计算
    error = aclnnIsNegInf(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        aclDestroyAclOpExecutor(executor);
        throw std::runtime_error(fmt::format("IsNegInf: computation failed, error={}", error));
    }

    // 5. 同步设备
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        aclDestroyAclOpExecutor(executor);
        throw std::runtime_error(fmt::format("IsNegInf: sync device failed, error={}", error));
    }

    // 6. 释放所有资源
    aclrtDestroyStream(stream);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    aclDestroyAclOpExecutor(executor);

    return result;
}


/**
 * @brief Test element-wise for positive infinity (inf).
 * 
 * Equivalent to numpy.isposinf(x), returns True for elements that are positive infinite (inf).
 * Only supports floating-point input arrays.
 * 
 * @param x NPUArray, input array (floating-point type: ACL_FLOAT/ACL_DOUBLE)
 * @return NPUArray Boolean array (same shape as x), True if element is positive infinite
 */
NPUArray IsPosInf(const NPUArray& x) {
    auto result = NPUArray(x.shape, ACL_BOOL);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;
    auto error = ACL_SUCCESS;

    error = aclnnIsPosInfGetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("IsPosInf: get workspace size failed, error={}", error));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("IsPosInf: malloc workspace failed, error={}", error));
        }
    }
    
    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("IsPosInf: create stream failed, error={}", error));
    }

    error = aclnnIsPosInf(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("IsPosInf: computation failed, error={}", error));
    }

    error = aclrtSynchronizeStream(stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("IsPosInf: sync device failed, error={}", error));
    }

    aclrtDestroyStream(stream);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);

    return result;
}


/**
 * @brief Element-wise logical AND operation between two boolean arrays.
 * 
 * Equivalent to numpy.logical_and(x, y), performs element-wise logical AND operation.
 * Input arrays x and y must have the same shape.
 * 
 * @param x NPUArray, first input boolean array (ACL_BOOL)
 * @param y NPUArray, second input boolean array (ACL_BOOL)
 */
NPUArray LogicalAnd(const NPUArray& x, const NPUArray& y) {
    auto result = NPUArray(x.shape, ACL_BOOL);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;
    auto error = ACL_SUCCESS;

    error = aclnnLogicalAndGetWorkspaceSize(x.tensorPtr, y.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("LogicalAnd: get workspace size failed, error={}", error));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("LogicalAnd: malloc workspace failed, error={}", error));
        }
    }
    
    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("LogicalAnd: create stream failed, error={}", error));
    }

    error = aclnnLogicalAnd(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("LogicalAnd: computation failed, error={}", error));
    }

    error = aclrtSynchronizeStream(stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("LogicalAnd: sync device failed, error={}", error));
    }

    aclrtDestroyStream(stream);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);

    return result;
}


/**
 * @brief Element-wise logical OR operation between two boolean arrays.
 * 
 * Equivalent to numpy.logical_or(x, y), performs element-wise logical OR operation.
 * Input arrays x and y must have the same shape.
 * 
 * @param x NPUArray, first input boolean array (ACL_BOOL)
 * @param y NPUArray, second input boolean array (ACL_BOOL)
 */
NPUArray LogicalOr(const NPUArray& x, const NPUArray& y) {
    auto result = NPUArray(x.shape, ACL_BOOL);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;
    auto error = ACL_SUCCESS;

    error = aclnnLogicalOrGetWorkspaceSize(x.tensorPtr, y.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("LogicalOr: get workspace size failed, error={}", error));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("LogicalOr: malloc workspace failed, error={}", error));
        }
    }
    
    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("LogicalOr: create stream failed, error={}", error));
    }

    error = aclnnLogicalOr(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("LogicalOr: computation failed, error={}", error));
    }

    error = aclrtSynchronizeStream(stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("LogicalOr: sync device failed, error={}", error));
    }

    aclrtDestroyStream(stream);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);

    return result;
}


/**
 * @brief Element-wise logical NOT operation on a boolean array.
 * 
 * Equivalent to numpy.logical_not(x), performs element-wise logical NOT operation.
 * 
 * @param x NPUArray, input boolean array (ACL_BOOL)
 * @return NPUArray Resulting boolean array after NOT operation
 */
NPUArray LogicalNot(const NPUArray& x) {
    auto result = NPUArray(x.shape, ACL_BOOL);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;
    auto error = ACL_SUCCESS;

    error = aclnnLogicalNotGetWorkspaceSize(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("LogicalNot: get workspace size failed, error={}", error));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("LogicalNot: malloc workspace failed, error={}", error));
        }
    }
    
    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("LogicalNot: create stream failed, error={}", error));
    }

    error = aclnnLogicalNot(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("LogicalNot: computation failed, error={}", error));
    }

    error = aclrtSynchronizeStream(stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("LogicalNot: sync device failed, error={}", error));
    }

    aclrtDestroyStream(stream);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);

    return result;
}


/**
 * @brief Element-wise logical XOR operation between two boolean arrays.
 * 
 * Equivalent to numpy.logical_xor(x, y), performs element-wise logical XOR operation.
 * Input arrays x and y must have the same shape.
 * 
 * @param x NPUArray, first input boolean array (ACL_BOOL)
 * @param y NPUArray, second input boolean array (ACL_BOOL)
 */
NPUArray LogicalXor(const NPUArray& x, const NPUArray& y) {
    auto result = NPUArray(x.shape, ACL_BOOL);
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;
    auto error = ACL_SUCCESS;

    error = aclnnLogicalXorGetWorkspaceSize(x.tensorPtr, y.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("LogicalXor: get workspace size failed, error={}", error));
    }

    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("LogicalXor: malloc workspace failed, error={}", error));
        }
    }
    
    error = aclrtCreateStream(&stream);
    if (error != ACL_SUCCESS || stream == nullptr) {
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("LogicalXor: create stream failed, error={}", error));
    }

    error = aclnnLogicalXor(workspaceAddr, workspaceSize, executor, stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("LogicalXor: computation failed, error={}", error));
    }

    error = aclrtSynchronizeStream(stream);
    if (error != ACL_SUCCESS) {
        aclrtDestroyStream(stream);
        if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
        throw std::runtime_error(fmt::format("LogicalXor: sync device failed, error={}", error));
    }

    aclrtDestroyStream(stream);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);

    return result;
}

/**
 * @brief Perform element-wise greater-than comparison between two arrays.
 */
NPUArray greater(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    auto out = NPUArray(GetBroadcastShape(x1, x2), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGtTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](greater) aclnnGtTensorGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) throw std::runtime_error("[logic.cpp](greater) Invalid workspaceSize: " + std::to_string(workspaceSize));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[logic.cpp](greater) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnGtTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](greater) aclnnGtTensor error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](greater) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}

/**
 * @brief Perform element-wise greater-than comparison between an array and a scalar.
 */
NPUArray greater(const NPUArray& x1, const py::object& scalar, py::dtype dtype) {
    auto out = NPUArray(x1.shape, dtype);

    double scalar_val = 0;
    try {
        scalar_val = py::cast<double>(scalar);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[logic.cpp](greater) Invalid scalar type: " + std::string(e.what()));
    }
    aclScalar* acl_scalar = aclCreateScalar(&scalar_val, x1.aclDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGtScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](greater) aclnnGtScalarGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        aclDestroyScalar(acl_scalar);
        throw std::runtime_error("[logic.cpp](greater) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(acl_scalar);
            std::string error_msg = "[logic.cpp](greater) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnGtScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](greater) aclnnGtScalar error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](greater) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(acl_scalar);

    return out;
}

/**
 * @brief Perform element-wise greater-than-or-equal comparison between two arrays.
 */
NPUArray greater_equal(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    auto out = NPUArray(GetBroadcastShape(x1, x2), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGeTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](greater_equal) aclnnGeTensorGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) throw std::runtime_error("[logic.cpp](greater_equal) Invalid workspaceSize: " + std::to_string(workspaceSize));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[logic.cpp](greater_equal) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnGeTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string error_msg = "[logic.cpp](greater_equal) aclnnGeTensor error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string error_msg = "[logic.cpp](greater_equal) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}

/**
 * @brief Perform element-wise greater-than-or-equal comparison between an array and a scalar.
 */
NPUArray greater_equal(const NPUArray& x1, const py::object& scalar, py::dtype dtype) {
    auto out = NPUArray(x1.shape, dtype);

    double scalar_val = 0;
    try {
        scalar_val = py::cast<double>(scalar);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[logic.cpp](greater_equal) Invalid scalar type: " + std::string(e.what()));
    }
    aclScalar* acl_scalar = aclCreateScalar(&scalar_val, x1.aclDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGeScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](greater_equal) aclnnGeScalarGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        aclDestroyScalar(acl_scalar);
        throw std::runtime_error("[logic.cpp](greater_equal) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(acl_scalar);
            std::string error_msg = "[logic.cpp](greater_equal) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnGeScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](greater_equal) aclnnGeScalar error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](greater_equal) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(acl_scalar);

    return out;
}

/**
 * @brief Perform element-wise less-than comparison between two arrays.
 */
NPUArray less(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    auto out = NPUArray(GetBroadcastShape(x1, x2), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnLtTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](less) aclnnLtTensorGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0)
        throw std::runtime_error("[logic.cpp](less) Invalid workspaceSize: " + std::to_string(workspaceSize));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[logic.cpp](less) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnLtTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string error_msg = "[logic.cpp](less) aclnnLtTensor error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string error_msg = "[logic.cpp](less) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}

/**
 * @brief Perform element-wise less-than comparison between an array and a scalar.
 */
NPUArray less(const NPUArray& x1, const py::object& scalar, py::dtype dtype) {
    auto out = NPUArray(x1.shape, dtype);

    double scalar_val = 0;
    try {
        scalar_val = py::cast<double>(scalar);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[logic.cpp](less) Invalid scalar type: " + std::string(e.what()));
    }
    aclScalar* acl_scalar = aclCreateScalar(&scalar_val, x1.aclDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnLtScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](less) aclnnLtScalarGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        aclDestroyScalar(acl_scalar);
        throw std::runtime_error("[logic.cpp](less) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(acl_scalar);
            std::string error_msg = "[logic.cpp](less) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnLtScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](less) aclnnLtScalar error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](less) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(acl_scalar);

    return out;
}

/**
 * @brief Perform element-wise less-than-or-equal comparison between two arrays.
 */
NPUArray less_equal(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    auto out = NPUArray(GetBroadcastShape(x1, x2), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnLeTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](less_equal) aclnnLeTensorGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0)
        throw std::runtime_error("[logic.cpp](less_equal) Invalid workspaceSize: " + std::to_string(workspaceSize));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[logic.cpp](less_equal) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnLeTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string error_msg = "[logic.cpp](less_equal) aclnnLeTensor error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string error_msg = "[logic.cpp](less_equal) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}

/**
 * @brief Perform element-wise less-than-or-equal comparison between an array and a scalar.
 */
NPUArray less_equal(const NPUArray& x1, const py::object& scalar, py::dtype dtype) {
    auto out = NPUArray(x1.shape, dtype);

    double scalar_val = 0;
    try {
        scalar_val = py::cast<double>(scalar);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[logic.cpp](less_equal) Invalid scalar type: " + std::string(e.what()));
    }
    aclScalar* acl_scalar = aclCreateScalar(&scalar_val, x1.aclDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnLeScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](less_equal) aclnnLeScalarGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        aclDestroyScalar(acl_scalar);
        throw std::runtime_error("[logic.cpp](less_equal) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(acl_scalar);
            std::string error_msg = "[logic.cpp](less_equal) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnLeScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](less_equal) aclnnLeScalar error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](less_equal) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(acl_scalar);

    return out;
}

/**
 * @brief Perform element-wise equality comparison between two arrays.
 */
NPUArray equal(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    auto out = NPUArray(GetBroadcastShape(x1, x2), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnEqualGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](equal) aclnnEqualGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0)
        throw std::runtime_error("[logic.cpp](equal) Invalid workspaceSize: " + std::to_string(workspaceSize));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[logic.cpp](equal) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnEqual(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string error_msg = "[logic.cpp](equal) aclnnEqual error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string error_msg = "[logic.cpp](equal) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}

/**
 * @brief Perform element-wise not-equal comparison between two arrays.
 */
NPUArray not_equal(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    auto out = NPUArray(GetBroadcastShape(x1, x2), dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnNeTensorGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[logic.cpp](not_equal) aclnnNeTensorGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0)
        throw std::runtime_error("[logic.cpp](not_equal) Invalid workspaceSize: " + std::to_string(workspaceSize));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[logic.cpp](not_equal) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnNeTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string error_msg = "[logic.cpp](not_equal) aclnnNeTensor error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string error_msg = "[logic.cpp](not_equal) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}

/**
 * @brief Perform element-wise not-equal comparison between an array and a scalar.
 */
NPUArray not_equal(const NPUArray& x1, const py::object& scalar, py::dtype dtype) {
    auto out = NPUArray(x1.shape, dtype);

    double scalar_val = 0;
    try {
        scalar_val = py::cast<double>(scalar);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[logic.cpp](not_equal) Invalid scalar type: " + std::string(e.what()));
    }
    aclScalar* acl_scalar = aclCreateScalar(&scalar_val, x1.aclDtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnNeScalarGetWorkspaceSize(
        x1.tensorPtr, acl_scalar, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](not_equal) aclnnNeScalarGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        aclDestroyScalar(acl_scalar);
        throw std::runtime_error("[logic.cpp](not_equal) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(acl_scalar);
            std::string error_msg = "[logic.cpp](not_equal) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnNeScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](not_equal) aclnnNeScalar error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(acl_scalar);
        std::string error_msg = "[logic.cpp](not_equal) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(acl_scalar);

    return out;
}
