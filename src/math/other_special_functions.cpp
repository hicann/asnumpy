#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_sinc.h>
#include <asnumpy/math/other_special_functions.hpp>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

/**
 * @brief Element-wise sinc function using aclnnSinc.
 */
NPUArray sinc(const NPUArray& x, std::optional<py::dtype> dtype) {
    auto shape = x.shape;
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    // 再把 out_dtype 转回 py::dtype，传给 NPUArray 构造函数
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    NPUArray out(shape, out_py_dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 获取 workspace 和执行器
    auto error = aclnnSincGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string msg = "[other_special_functions.cpp](sinc) aclnnSincGetWorkspaceSize error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        throw std::runtime_error(msg);
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string msg = "[other_special_functions.cpp](sinc) aclrtMalloc error = "
                              + std::to_string(error);
            throw std::runtime_error(msg);
        }
    }

    // 执行算子
    error = aclnnSinc(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string msg = "[other_special_functions.cpp](sinc) aclnnSinc error = "
                          + std::to_string(error);
        throw std::runtime_error(msg);
    }

    // 同步设备
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string msg = "[other_special_functions.cpp](sinc) aclrtSynchronizeDevice error = "
                          + std::to_string(error);
        throw std::runtime_error(msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

}