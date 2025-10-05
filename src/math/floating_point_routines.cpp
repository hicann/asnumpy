#include <asnumpy/math/floating_point_routines.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_signbit.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

/**
 * @brief Compute element-wise sign bit check.
 * 
 * Equivalent to numpy.signbit(x), returns a boolean array indicating whether the sign bit is set (negative values).
 * 
 * @param x NPUArray, input array (numeric type)
 * @return NPUArray Boolean array where True indicates negative elements (sign bit set)
 */
NPUArray Signbit(const NPUArray& x) {
    // 初始化结果数组（形状与输入一致，数据类型为布尔型）
    auto shape = x.shape;
    NPUArray result(shape, ACL_BOOL);  // 布尔型输出（True表示负数）

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnSignbitGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Signbit: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Signbit: malloc workspace failed, error={}", error));
        }
    }

    // 执行符号位检查（检测是否为负数）
    error = aclnnSignbit(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Signbit: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}

}