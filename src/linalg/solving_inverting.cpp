#include <asnumpy/linalg/solving_inverting.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_inverse.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

NPUArray Linalg_Inv(const NPUArray& a) {
    auto shape = a.shape;
    auto result = NPUArray(shape, ACL_DOUBLE);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnInverseGetWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnInverse(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnInverse error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}