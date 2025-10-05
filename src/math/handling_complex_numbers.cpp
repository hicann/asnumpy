#include <asnumpy/math/handling_complex_numbers.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_real.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy{
    NPUArray Real(const NPUArray& val) {
        auto shape = val.shape;
        auto aclType = val.aclDtype;
        if (val.aclDtype == ACL_COMPLEX64 || val.aclDtype == ACL_COMPLEX128){
            aclType = ACL_FLOAT;
        }
        auto result = NPUArray(shape, aclType);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnRealGetWorkspaceSize(val.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnReal(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnReal error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }
}