#include <asnumpy/math/miscellaneous.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/npu_ops_macros.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_flip.h>
#include <aclnnop/aclnn_convolution.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_pow.h>
#include <aclnnop/aclnn_nan_to_num.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_sign.h>
#include <aclnnop/aclnn_heaviside.h>
#include <aclnnop/aclnn_maximum.h>
#include <aclnnop/aclnn_minimum.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <limits>
#include <stdexcept>

/**NPUArray Convolve(const NPUArray& a, const NPUArray& v) {
    std::vector<int64_t> dims = {2};
    auto dims_acl = aclCreateIntArray(dims.data(), 1);
    auto shape1 = a.shape;
    auto temp = NPUArray(shape1, a.aclDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnFlipGetWorkspaceSize(a.tensorPtr, dims_acl, temp.tensorPtr, &workspaceSize1, &executor1);
    if (error1 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](convolve) aclnnFlipGetWorkspaceSize error = " + std::to_string(error1);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize1 < 0) {
        throw std::runtime_error("[miscellaneous.cpp](convolve) Invalid workspaceSize: " + std::to_string(workspaceSize1));
    }

    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error1 != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](convolve) aclrtMalloc error = " + std::to_string(error1);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error1 = aclnnFlip(workspaceAddr1, workspaceSize1, executor1, nullptr);
    if (error1 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](convolve) aclnnFlip error = " + std::to_string(error1);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr1) aclrtFree(workspaceAddr1);
        throw std::runtime_error(error_msg);
    }
    error1 = aclrtSynchronizeDevice();
    if (error1 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](convolve) aclrtSynchronizeDevice error = " + std::to_string(error1);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr1) aclrtFree(workspaceAddr1);
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr1) aclrtFree(workspaceAddr1);

    auto shape2 = v.shape;
    int64_t size = shape1[2] + shape2[2] - 1;
    std::vector<int64_t> shapeResult = {1, 1, size, 1};
    std::vector<int64_t> convStrides = {1, 1};
    std::vector<int64_t> convPads = {shape2[2] - 1, 0};
    std::vector<int64_t> convOutPads = {0, 0};
    std::vector<int64_t> convDilations = {1, 1};
    auto strides = aclCreateIntArray(convStrides.data(), 2);
    auto pads = aclCreateIntArray(convPads.data(), 2);
    auto outPads = aclCreateIntArray(convOutPads.data(), 2);
    auto dilations = aclCreateIntArray(convDilations.data(), 2);
    auto result = NPUArray(shapeResult, ACL_FLOAT);
    result.tensorPtr = aclCreateTensor(result.shape.data(), result.shape.size(), GetACLDataType(result.dtype), result.strides.data(), 0, ACL_FORMAT_NCHW, result.shape.data(), result.shape.size(), result.devicePtr);
    int8_t use_fp16 = 2;
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnConvolutionGetWorkspaceSize(temp.tensorPtr, v.tensorPtr, nullptr, strides, pads, dilations, false, outPads, 1, result.tensorPtr, use_fp16, &workspaceSize2, &executor2);
    if (error2 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](convolve) aclnnConvolutionGetWorkspaceSize error = " + std::to_string(error2);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize2 < 0) {
        throw std::runtime_error("[miscellaneous.cpp](convolve) Invalid workspaceSize: " + std::to_string(workspaceSize2));
    }

    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error2 != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](convolve) aclrtMalloc error = " + std::to_string(error2);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error2 = aclnnConvolution(workspaceAddr2, workspaceSize2, executor2, nullptr);
    if (error2 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](convolve) aclnnConvolution error = " + std::to_string(error2);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr2) aclrtFree(workspaceAddr2);
        throw std::runtime_error(error_msg);
    }

    error2 = aclrtSynchronizeDevice();
    if (error2 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](convolve) aclrtSynchronizeDevice error = " + std::to_string(error2);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr2) aclrtFree(workspaceAddr2);
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr2) aclrtFree(workspaceAddr2);
    return result;
}*/

NPUArray Clip(const NPUArray& a, const NPUArray& a_min, const NPUArray& a_max) {
    auto temp = GetBroadcastShape(a, a_min);
    auto x = NPUArray(temp, ACL_FLOAT);
    auto broadcast = GetBroadcastShape(x, a_max);
    auto result = NPUArray(broadcast, ACL_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnClampTensorGetWorkspaceSize(a.tensorPtr, a_min.tensorPtr, a_max.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampTensorGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](clip) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnClampTensor(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampTensor error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return result;
}

NPUArray Clip(const NPUArray& a, const py::object& a_min, const py::object& a_max) {
    auto shape = a.shape;
    double valueDouble1 = 0, valueDouble2 = 0;
    if (a_min.is_none() || a_max.is_none()) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Input is None");
    }
    try {
        valueDouble1 = py::cast<double>(a_min);
        valueDouble2 = py::cast<double>(a_max);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Conversion error: " + std::string(e.what()));
    }
    auto amin_scalar = aclCreateScalar(&valueDouble1, ACL_FLOAT);
    auto amax_scalar = aclCreateScalar(&valueDouble2, ACL_FLOAT);
    auto result = NPUArray(shape, ACL_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnClampGetWorkspaceSize(a.tensorPtr, amin_scalar, amax_scalar, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](clip) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnClamp(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClamp error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return result;
}

NPUArray Clip(const NPUArray& a, const py::object& a_min, const NPUArray& a_max) {
    auto shape = a.shape;
    double valueDouble = 0;
    if (a_min.is_none()) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Input is None");
    }
    try {
        valueDouble = py::cast<double>(a_min);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Conversion error: " + std::string(e.what()));
    }
    auto amin_scalar = aclCreateScalar(&valueDouble, ACL_FLOAT);
    auto temp = NPUArray(shape, ACL_FLOAT);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnClampMinGetWorkspaceSize(a.tensorPtr, amin_scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    if (error1 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampMinGetWorkspaceSize error = " + std::to_string(error1);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize1 < 0) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Invalid workspaceSize: " + std::to_string(workspaceSize1));
    }

    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error1 != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](clip) aclrtMalloc error = " + std::to_string(error1);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error1 = aclnnClampMin(workspaceAddr1, workspaceSize1, executor1, nullptr);
    if (error1 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampMin error = " + std::to_string(error1);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr1) aclrtFree(workspaceAddr1);
        throw std::runtime_error(error_msg);
    }
    error1 = aclrtSynchronizeDevice();
    if (error1 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclrtSynchronizeDevice error = " + std::to_string(error1);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr1) aclrtFree(workspaceAddr1);
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr1) aclrtFree(workspaceAddr1);

    auto broadcast = GetBroadcastShape(temp, a_max);
    auto result = NPUArray(broadcast, ACL_FLOAT);
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnClampMaxTensorGetWorkspaceSize(temp.tensorPtr, a_max.tensorPtr, result.tensorPtr, &workspaceSize2, &executor2);
    if (error2 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampMaxTensorGetWorkspaceSize error = " + std::to_string(error2);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize2 < 0) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Invalid workspaceSize: " + std::to_string(workspaceSize2));
    }

    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error2 != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](clip) aclrtMalloc error = " + std::to_string(error2);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error2 = aclnnClampMaxTensor(workspaceAddr2, workspaceSize2, executor2, nullptr);
    if (error2 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampMaxTensor error = " + std::to_string(error2);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr2) aclrtFree(workspaceAddr2);
        throw std::runtime_error(error_msg);
    }

    error2 = aclrtSynchronizeDevice();
    if (error2 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclrtSynchronizeDevice error = " + std::to_string(error2);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr2) aclrtFree(workspaceAddr2);
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr2) aclrtFree(workspaceAddr2);
    return result;
}

NPUArray Clip(const NPUArray& a, const NPUArray& a_min, const py::object& a_max) {
    auto shape = a.shape;
    double valueDouble = 0;
    if (a_max.is_none()) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Input is None");
    }
    try {
        valueDouble = py::cast<double>(a_max);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Conversion error: " + std::string(e.what()));
    }
    auto amax_scalar = aclCreateScalar(&valueDouble, ACL_FLOAT);
    auto temp = NPUArray(shape, ACL_FLOAT);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnClampMaxGetWorkspaceSize(a.tensorPtr, amax_scalar, temp.tensorPtr, &workspaceSize1, &executor1);
    if (error1 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampMaxGetWorkspaceSize error = " + std::to_string(error1);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize1 < 0) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Invalid workspaceSize: " + std::to_string(workspaceSize1));
    }

    void* workspaceAddr1 = nullptr;
    if(workspaceSize1 > 0) {
        error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error1 != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](clip) aclrtMalloc error = " + std::to_string(error1);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error1 = aclnnClampMax(workspaceAddr1, workspaceSize1, executor1, nullptr);
    if (error1 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampMax error = " + std::to_string(error1);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr1) aclrtFree(workspaceAddr1);
        throw std::runtime_error(error_msg);
    }
    error1 = aclrtSynchronizeDevice();
    if (error1 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclrtSynchronizeDevice error = " + std::to_string(error1);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr1) aclrtFree(workspaceAddr1);
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr1) aclrtFree(workspaceAddr1);

    auto broadcast = GetBroadcastShape(a_min, temp);
    auto result = NPUArray(broadcast, ACL_FLOAT);
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    auto error2 = aclnnClampMinTensorGetWorkspaceSize(temp.tensorPtr, a_min.tensorPtr, result.tensorPtr, &workspaceSize2, &executor2);
    if (error2 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampMinTensorGetWorkspaceSize error = " + std::to_string(error2);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize2 < 0) {
        throw std::runtime_error("[miscellaneous.cpp](clip) Invalid workspaceSize: " + std::to_string(workspaceSize2));
    }

    void* workspaceAddr2 = nullptr;
    if(workspaceSize2 > 0) {
        error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error2 != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](clip) aclrtMalloc error = " + std::to_string(error2);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error2 = aclnnClampMinTensor(workspaceAddr2, workspaceSize2, executor2, nullptr);
    if (error2 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclnnClampMinTensor error = " + std::to_string(error2);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr2) aclrtFree(workspaceAddr2);
        throw std::runtime_error(error_msg);
    }

    error2 = aclrtSynchronizeDevice();
    if (error2 != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](clip) aclrtSynchronizeDevice error = " + std::to_string(error2);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr2) aclrtFree(workspaceAddr2);
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr2) aclrtFree(workspaceAddr2);
    return result;
}

NPUArray Square(const NPUArray& x) {
    auto shape = x.shape;
    double two = 2.0;
    auto scalar = aclCreateScalar(&two, ACL_FLOAT);
    auto result = NPUArray(shape, ACL_FLOAT);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnPowTensorScalarGetWorkspaceSize(x.tensorPtr, scalar, result.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](square) aclnnPowTensorScalarGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[miscellaneous.cpp](square) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](square) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](square) aclnnPowTensorScalar error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](square) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return result;
}

DEFINE_UNARY_OP(Absolute, aclnnAbsGetWorkspaceSize, aclnnAbs)
DEFINE_UNARY_OP(Sign, aclnnSignGetWorkspaceSize, aclnnSign)
DEFINE_BINARY_OP(Heaviside, aclnnHeavisideGetWorkspaceSize, aclnnHeaviside)

NPUArray Fabs(const NPUArray& x){
    // absolute 处理所有数据类型（包括复数等） fabs只处理float和int，
    // 但aclnnAbs不支持复数，所以这里默认fabs=absolute
    return Absolute(x);
}

/**
 * @brief Replace NaN and infinities in an array using NPU.
 *
 * Creates an output array and applies aclnnNanToNum to replace NaN, +inf, and -inf.
 */
NPUArray Nan_to_num(const NPUArray& x,
                    double nan,
                    py::object posinf,
                    py::object neginf,
                    py::dtype dtype) {
    auto out = NPUArray(x.shape, dtype);

    // Convert optional posinf/neginf to doubles; use NaN as "not provided" sentinel.
    double pos_val = std::numeric_limits<double>::quiet_NaN();
    double neg_val = std::numeric_limits<double>::quiet_NaN();
    if (!posinf.is_none()) pos_val = posinf.cast<double>();
    if (!neginf.is_none()) neg_val = neginf.cast<double>();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto error = aclnnNanToNumGetWorkspaceSize(
        x.tensorPtr,          // input
        nan,                  // NaN replacement
        pos_val,              // +inf replacement (NaN sentinel means "use default")
        neg_val,              // -inf replacement (NaN sentinel means "use default")
        out.tensorPtr,        // output
        &workspaceSize,
        &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[floating_point_routines.cpp](nan_to_num) aclnnNanToNumGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[floating_point_routines.cpp](nan_to_num) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[floating_point_routines.cpp](nan_to_num) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnNanToNum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[floating_point_routines.cpp](nan_to_num) aclnnNanToNum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[floating_point_routines.cpp](nan_to_num) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

/**
 * @brief Element-wise maximum of two arrays.
 *
 * Creates an output array on NPU and computes element-wise max(x1, x2)
 * using the aclnnMaximum operator.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise maxima.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray maximum(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    // 输出形状应与广播结果一致；此处先以 x1.shape() 分配。
    // 如需显式广播形状，请在有统一工具后替换为广播形状。
    auto broadcast = GetBroadcastShape(x1, x2);
    auto out = NPUArray(broadcast, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnMaximumGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](maximum) aclnnMaximumGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[miscellaneous.cpp](maximum) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](maximum) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnMaximum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](maximum) aclnnMaximum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](maximum) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

/**
 * @brief Element-wise minimum of two arrays.
 *
 * Creates an output array on NPU and computes element-wise min(x1, x2)
 * using the aclnnMinimum operator.
 *
 * @param x1 First input array.
 * @param x2 Second input array.
 * @param dtype Target numpy dtype for the output array.
 * @return NPUArray Array with element-wise minima.
 * @throws std::runtime_error If ACL operation or memory allocation fails.
 */
NPUArray minimum(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    // 输出形状应与广播结果一致；此处先以 x1.shape() 分配。
    // 如需显式广播形状，请在有统一工具后替换为广播形状。
    auto broadcast = GetBroadcastShape(x1, x2);
    auto out = NPUArray(broadcast, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnMinimumGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](minimum) aclnnMinimumGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[miscellaneous.cpp](minimum) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](minimum) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnMinimum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](minimum) aclnnMinimum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](minimum) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

NPUArray fmax(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    auto broadcast = GetBroadcastShape(x1, x2);
    auto out = NPUArray(broadcast, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnMaximumGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmax) aclnnMaximumGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[miscellaneous.cpp](fmax) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](fmax) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnMaximum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmax) aclnnMaximum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmax) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

NPUArray fmin(const NPUArray& x1, const NPUArray& x2, py::dtype dtype) {
    auto broadcast = GetBroadcastShape(x1, x2);
    auto out = NPUArray(broadcast, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnMinimumGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmin) aclnnMinimumGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[miscellaneous.cpp](fmin) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[miscellaneous.cpp](fmin) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0)
                error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnMinimum(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmin) aclnnMinimum error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[miscellaneous.cpp](fmin) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0)
            error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}
