#include <asnumpy/math/sums_products_differences.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_prod.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_sum.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_cumsum.h>
#include <aclnnop/aclnn_cumprod.h>
#include <aclnnop/aclnn_nan_to_num.h>
#include <aclnnop/aclnn_linalg_cross.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

NPUArray Prod(const NPUArray& a, int64_t axis, py::dtype dtype, bool keepdims) {
    auto shape = a.shape;
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnProdDimGetWorkspaceSize(a.tensorPtr, axis, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnProdDim(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnProdDim error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Prod(const NPUArray& a, py::dtype dtype) {
    std::vector<int64_t> shape = {1};
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnProdGetWorkspaceSize(a.tensorPtr, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnProd(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnProd error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Sum(const NPUArray& a, const std::vector<int64_t>& axis, py::dtype dtype, bool keepdims) {
    auto shape = a.shape;
    aclIntArray* axis_array = aclCreateIntArray(axis.data(), axis.size());
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnReduceSumGetWorkspaceSize(a.tensorPtr, axis_array, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnReduceSum(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnReduceSum error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

double Sum(const NPUArray& a) {
    std::vector<int64_t> shape = {1};
    std::vector<aclTensor *> tmp{a.tensorPtr};
    auto input = aclCreateTensorList(tmp.data(), tmp.size());
    auto temp = NPUArray(shape, ACL_DOUBLE);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnSumGetWorkspaceSize(input, temp.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error1);
    }
    error1 = aclnnSum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnSum error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);
    
    py::array results = temp.ToNumpy();
    void* data_ptr = results.mutable_data();
    double result = static_cast<double*>(data_ptr)[0];
    return result;
}

NPUArray Nanprod(const NPUArray& a, int64_t axis, py::dtype dtype, bool keepdims) {
    auto shape = a.shape;
    float scalar = 1.0;
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, scalar, scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error1);
    }
    error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnNanToNum error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);
    
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnProdDimGetWorkspaceSize(temp.tensorPtr, axis, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    CheckGetWorkspaceSizeAclnnStatus(error2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error2);
    }
    error2 = aclnnProdDim(workspaceAddr2, workspaceSize2, executor2, nullptr);
    CheckAclnnStatus(error2, "aclnnProdDim error");
    error2 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error2);
    return result;
}

NPUArray Nanprod(const NPUArray& a, py::dtype dtype) {
    std::vector<int64_t> shape = {1};
    float scalar = 1.0;
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, scalar, scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error1);
    }
    error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnNanToNum error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);
    
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnProdGetWorkspaceSize(temp.tensorPtr, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    CheckGetWorkspaceSizeAclnnStatus(error2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error2);
    }
    error2 = aclnnProd(workspaceAddr2, workspaceSize2, executor2, nullptr);
    CheckAclnnStatus(error2, "aclnnProd error");
    error2 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error2);
    return result;
}

NPUArray Nansum(const NPUArray& a, const std::vector<int64_t>& axis, py::dtype dtype, bool keepdims) {
    auto shape = a.shape;
    float scalar = 0.0;
    aclIntArray* axis_array = aclCreateIntArray(axis.data(), axis.size());
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, scalar, scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error1);
    }
    error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnNanToNum error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);
    
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnReduceSumGetWorkspaceSize(temp.tensorPtr, axis_array, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    CheckGetWorkspaceSizeAclnnStatus(error2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error2);
    }
    error2 = aclnnReduceSum(workspaceAddr2, workspaceSize2, executor2, nullptr);
    CheckAclnnStatus(error2, "aclnnReduceSum error");
    error2 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error2);
    return result;
}

NPUArray Nansum(const NPUArray& a, py::dtype dtype) {
    auto shape = a.shape;
    float scalar = 0.0;
    auto temp1 = NPUArray(shape, a.aclDtype);
    auto temp2 = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, scalar, scalar, temp1.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnNanToNum(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnNanToNum error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    
    std::vector<aclTensor *> tmp{temp1.tensorPtr};
    auto input = aclCreateTensorList(tmp.data(), tmp.size());
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnSumGetWorkspaceSize(input, temp2.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error1);
    }
    error1 = aclnnSum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnSum error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);
    
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnCastGetWorkspaceSize(temp2.tensorPtr, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    CheckGetWorkspaceSizeAclnnStatus(error2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error2);
    }
    error2 = aclnnCast(workspaceAddr2, workspaceSize2, executor2, nullptr);
    CheckAclnnStatus(error2, "aclnnCast error");
    error2 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error2);
    return result;
}

NPUArray Cumprod(const NPUArray& a, int64_t axis, py::dtype dtype) {
    auto shape = a.shape;
    auto axis_scalar = aclCreateScalar(&axis, ACL_INT64);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnCumprodGetWorkspaceSize(a.tensorPtr, axis_scalar, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnCumprod(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnCumprod error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Cumsum(const NPUArray& a, int64_t axis, py::dtype dtype) {
    auto shape = a.shape;
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnCumsumGetWorkspaceSize(a.tensorPtr, axis, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnCumsum(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnCumsum error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}

NPUArray Nancumprod(const NPUArray& a, int64_t axis, py::dtype dtype) {
    auto shape = a.shape;
    auto axis_scalar = aclCreateScalar(&axis, ACL_INT64);
    float scalar = 1.0;
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, scalar, scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error1);
    }
    error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnNanToNum error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);

    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnCumprodGetWorkspaceSize(temp.tensorPtr, axis_scalar, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    CheckGetWorkspaceSizeAclnnStatus(error2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error2);
    }
    error2 = aclnnCumprod(workspaceAddr2, workspaceSize2, executor2, nullptr);
    CheckAclnnStatus(error2, "aclnnCumprod error");
    error2 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error2);
    return result;
}

NPUArray Nancumsum(const NPUArray& a, int64_t axis, py::dtype dtype) {
    auto shape = a.shape;
    float scalar = 0.0;
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, dtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, scalar, scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    CheckGetWorkspaceSizeAclnnStatus(error1);
    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error1);
    }
    error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
    CheckAclnnStatus(error1, "aclnnNanToNum error");
    error1 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error1);

    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnCumsumGetWorkspaceSize(temp.tensorPtr, axis, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
    CheckGetWorkspaceSizeAclnnStatus(error2);
    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error2);
    }
    error2 = aclnnCumsum(workspaceAddr2, workspaceSize2, executor2, nullptr);
    CheckAclnnStatus(error2, "aclnnCumsum error");
    error2 = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error2);
    return result;
}

NPUArray Cross(const NPUArray& a, const NPUArray& b, int64_t axisa, int64_t axisb, int64_t axisc, int64_t axis) {
    auto broadcast = GetBroadcastShape(a, b);
    auto result = NPUArray(broadcast, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnLinalgCrossGetWorkspaceSize(a.tensorPtr, b.tensorPtr, axis, result.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);
    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }
    error = aclnnLinalgCross(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "aclnnLinalgCross error");
    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);
    return result;
}