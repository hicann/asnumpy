#include <asnumpy/utils/status_handler.hpp>




void CheckAclnnStatus(aclnnStatus status, const std::string& context)
{
    if (status != ACL_SUCCESS) {

        const char* error_msg = aclGetRecentErrMsg();
        std::string full_error_msg;
        
        if (!context.empty()){
            full_error_msg = fmt::format("{} error code: {}. ",context,status);
        }
        else{
            full_error_msg = fmt::format("error code: {}. ",status);
        }

        if (error_msg != nullptr) {
            full_error_msg += fmt::format("Details: {}", error_msg);
        }
        
        throw std::runtime_error(full_error_msg);
    }
}


void CheckGetWorkspaceSizeAclnnStatus(aclnnStatus status){
    CheckAclnnStatus(status, "Failed to allocate workspace.");
}

void CheckMallocAclnnStatus(aclnnStatus status){
    CheckAclnnStatus(status, "Failed to allocate workspace memory.");
}


void CheckSynchronizeDeviceAclnnStatus(aclnnStatus status){
    CheckAclnnStatus(status, "Failed to synchronize device.");
}