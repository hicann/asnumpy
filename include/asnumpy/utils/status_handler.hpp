#include <fmt/format.h>
#include <stdexcept>
#include <string>
#include "acl/acl.h"
#include "aclnn/acl_meta.h" // aclnnStatus definition

/*
 * How to use:
 * 1   auto error = AclnnGetWorkspaceSizeFunc(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
 * 2   CheckAclnnStatus(error, "Failed to allocate workspace.");
 * or:
 * 1   auto error = AclnnGetWorkspaceSizeFunc(x.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
 * 2   CheckGetWorkspaceSizeAclnnStatus(error);
 *
 * output:
 *       RuntimeError: Failed to allocate workspace. error code 161001. Details: xxx
 *
 *       (if GetRecentErrMsg is not nullptr, show Details: xxx)
 */

/**
 * @brief Check aclnnStatus, and throw an exception with detailed error information if it fails
 *
 * @param status ACLNN status code
 * @param context Optional context information, used to describe the operation content
 * @throw std::runtime_error Thrown when status is not ACL_SUCCESS
 */
void CheckAclnnStatus(aclnnStatus status, const std::string& context);


// Directly call the corresponding error function below for commonly used APIs in the operator
// to avoid writing error operation information yourself

// aclnnXXXGetWorkspaceSize
void CheckGetWorkspaceSizeAclnnStatus(aclnnStatus status);
// aclrtMalloc
void CheckMallocAclnnStatus(aclnnStatus status);
// aclrtSynchronizeDevice
void CheckSynchronizeDeviceAclnnStatus(aclnnStatus status);
