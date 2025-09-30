#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/math/trigonometric_functions.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_tan.h>
#include <aclnnop/aclnn_asin.h>
#include <aclnnop/aclnn_acos.h>
#include <aclnnop/aclnn_atan.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_sqrt.h>
#include <aclnnop/aclnn_atan2.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_foreach_mul_scalar.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>


NPUArray sin(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnSinGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](sin) aclnnSinGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) throw std::runtime_error("[math.cpp](sin) Invalid workspaceSize: " + std::to_string(workspaceSize));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](sin) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnSin(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](sin) aclnnSin error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](sin) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}


NPUArray cos(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnCosGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](cos) aclnnCosGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) throw std::runtime_error("[math.cpp](cos) Invalid workspaceSize: " + std::to_string(workspaceSize));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](cos) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnCos(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](cos) aclnnCos error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](cos) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}


NPUArray tan(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnTanGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](tan) aclnnTanGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) throw std::runtime_error("[math.cpp](tan) Invalid workspaceSize: " + std::to_string(workspaceSize));

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](tan) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnTan(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](tan) aclnnTan error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](tan) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);

    return out;
}


NPUArray arcsin(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAsinGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](arcsin) aclnnAsinGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](arcsin) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](arcsin) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnAsin(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](arcsin) aclnnAsin error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](arcsin) aclrtSynchronizeDevice error = " + std::to_string(error);
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


NPUArray arccos(const NPUArray& x, py::dtype dtype) {

    auto out = NPUArray(x.shape, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAcosGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](arccos) aclnnAcosGetWorkspaceSize error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error("[math.cpp](arccos) Invalid workspaceSize: " + std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = "[math.cpp](arccos) aclrtMalloc error = " + std::to_string(error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnAcos(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](arccos) aclnnAcos error = " + std::to_string(error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && std::strlen(detailed_msg) > 0) error_msg += " - " + std::string(detailed_msg);
        if (workspaceAddr) aclrtFree(workspaceAddr);
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = "[math.cpp](arccos) aclrtSynchronizeDevice error = " + std::to_string(error);
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


NPUArray Arctan(const NPUArray& x, py::dtype dtype) {
    auto out = NPUArray(x.shape, dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    auto error = aclnnAtanGetWorkspaceSize(x.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("[math.cpp](arctan) aclnnAtanGetWorkspaceSize error = {}", error));
    }

    if(workspaceSize < 0) {
        throw std::runtime_error(fmt::format("[math.cpp](arctan) Invalid workspaceSize: {}", workspaceSize));
    }

    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("[math.cpp](arctan) aclrtMalloc error = {}", error));
        }
    }

    error = aclnnAtan(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("[math.cpp](arctan) aclnnAtan error = {}", error));
    }

    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("[math.cpp](arctan) aclrtSynchronizeDevice error = {}", error));
    }

    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}


NPUArray Hypot(const NPUArray& a, const NPUArray& b) {
    // 检查输入形状是否匹配
    if (a.shape != b.shape) {
        throw std::invalid_argument("Hypot: a and b must have the same shape");
    }

    // 初始化结果数组
    auto shape = a.shape;
    auto dtype = a.dtype;
    NPUArray result(shape, dtype);

    // 步骤1: 计算a的平方 (a²)
    NPUArray a_squared(shape, dtype);
    uint64_t a_sq_workspace_size = 0;
    aclOpExecutor* a_sq_executor = nullptr;
    auto error = aclnnMulGetWorkspaceSize(
        a.tensorPtr, a.tensorPtr,
        a_squared.tensorPtr,
        &a_sq_workspace_size,
        &a_sq_executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Hypot: a² workspace size failed, error={}", error));
    }

    void* a_sq_workspace = nullptr;
    if (a_sq_workspace_size > 0) {
        error = aclrtMalloc(&a_sq_workspace, a_sq_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Hypot: a² workspace malloc failed, error={}", error));
        }
    }

    error = aclnnMul(a_sq_workspace, a_sq_workspace_size, a_sq_executor, nullptr);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Hypot: a² computation failed, error={}", error));
    }

    // 步骤2: 计算b的平方 (b²)
    NPUArray b_squared(shape, dtype);
    uint64_t b_sq_workspace_size = 0;
    aclOpExecutor* b_sq_executor = nullptr;
    error = aclnnMulGetWorkspaceSize(
        b.tensorPtr, b.tensorPtr,
        b_squared.tensorPtr,
        &b_sq_workspace_size,
        &b_sq_executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Hypot: b² workspace size failed, error={}", error));
    }

    void* b_sq_workspace = nullptr;
    if (b_sq_workspace_size > 0) {
        error = aclrtMalloc(&b_sq_workspace, b_sq_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Hypot: b² workspace malloc failed, error={}", error));
        }
    }

    error = aclnnMul(b_sq_workspace, b_sq_workspace_size, b_sq_executor, nullptr);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Hypot: b² computation failed, error={}", error));
    }

    // 步骤3: 计算平方和 (a² + b²)
    NPUArray sum_squares(shape, dtype);
    uint64_t add_workspace_size = 0;
    aclOpExecutor* add_executor = nullptr;
    int32_t alpha = 1;
    auto alpha_scalar = aclCreateScalar(&alpha, a.aclDtype);
    
    error = aclnnAddGetWorkspaceSize(
        a_squared.tensorPtr, b_squared.tensorPtr,
        alpha_scalar, sum_squares.tensorPtr,
        &add_workspace_size, &add_executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Hypot: sum workspace size failed, error={}", error));
    }

    void* add_workspace = nullptr;
    if (add_workspace_size > 0) {
        error = aclrtMalloc(&add_workspace, add_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Hypot: sum workspace malloc failed, error={}", error));
        }
    }

    error = aclnnAdd(add_workspace, add_workspace_size, add_executor, nullptr);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Hypot: sum computation failed, error={}", error));
    }

    // 步骤4: 计算平方根 (√(a² + b²))
    uint64_t sqrt_workspace_size = 0;
    aclOpExecutor* sqrt_executor = nullptr;
    error = aclnnSqrtGetWorkspaceSize(
        sum_squares.tensorPtr, result.tensorPtr,
        &sqrt_workspace_size, &sqrt_executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Hypot: sqrt workspace size failed, error={}", error));
    }

    void* sqrt_workspace = nullptr;
    if (sqrt_workspace_size > 0) {
        error = aclrtMalloc(&sqrt_workspace, sqrt_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Hypot: sqrt workspace malloc failed, error={}", error));
        }
    }

    error = aclnnSqrt(sqrt_workspace, sqrt_workspace_size, sqrt_executor, nullptr);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Hypot: sqrt computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    aclDestroyScalar(alpha_scalar);
    if (a_sq_workspace) aclrtFree(a_sq_workspace);
    if (b_sq_workspace) aclrtFree(b_sq_workspace);
    if (add_workspace) aclrtFree(add_workspace);
    if (sqrt_workspace) aclrtFree(sqrt_workspace);

    return result;
}


NPUArray Arctan2(const NPUArray& y, const NPUArray& x) {
    // 检查输入形状是否匹配
    if (y.shape != x.shape) {
        throw std::invalid_argument("Arctan2: y and x must have the same shape");
    }

    // 初始化结果数组
    auto shape = y.shape;
    auto dtype = y.dtype;
    NPUArray result(shape, dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAtan2GetWorkspaceSize(
        y.tensorPtr, x.tensorPtr,
        result.tensorPtr,
        &workspace_size, &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arctan2: workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Arctan2: workspace malloc failed, error={}", error));
        }
    }

    // 执行计算
    error = aclnnAtan2(workspace, workspace_size, executor, nullptr);
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Arctan2: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace) aclrtFree(workspace);

    return result;
}


NPUArray Radians(const NPUArray& x) {
    // 业务参数校验
    if (x.tensorSize == 0) {
        throw std::runtime_error("Radians: input tensor has no elements");
    }
    if (x.aclDtype != ACL_FLOAT && x.aclDtype != ACL_DOUBLE && x.aclDtype != ACL_FLOAT16) {
        throw std::runtime_error("Radians: input must be float, double or float16 type");
    }

    // 初始化结果张量（与输入同形状同类型）
    NPUArray result(x.shape, x.aclDtype);

    // 角度转弧度因子：π/180
    NPUArray scalar_factor({}, x.aclDtype);
    const double rad_factor = M_PI / 180.0;
    void* scalar_factor_ptr = nullptr;

    // 资源声明
    aclrtStream stream = nullptr;
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    void *workspace_addr = nullptr;
    aclTensorList *input_list = nullptr;  // 输入张量列表
    aclTensorList *output_list = nullptr; // 输出张量列表

    try {
        // 获取标量张量的设备指针
        auto error = aclGetRawTensorAddr(scalar_factor.tensorPtr, &scalar_factor_ptr);
        if (error != ACL_SUCCESS || !scalar_factor_ptr) {
            throw std::runtime_error(fmt::format("Failed to get scalar factor pointer, error: {}", error));
        }

        // 拷贝转换因子到设备（按数据类型适配）
        if (x.aclDtype == ACL_FLOAT) {
            float factor = static_cast<float>(rad_factor);
            aclrtMemcpy(scalar_factor_ptr, sizeof(float), &factor, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        } else if (x.aclDtype == ACL_DOUBLE) {
            aclrtMemcpy(scalar_factor_ptr, sizeof(double), &rad_factor, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
        } else if (x.aclDtype == ACL_FLOAT16) {
            float factor_float = static_cast<float>(rad_factor);
            uint32_t float_bits = *reinterpret_cast<uint32_t*>(&factor_float);
            uint16_t fp16_bits = static_cast<uint16_t>(
                ((float_bits >> 16) & 0x8000) |
                (((float_bits >> 13) - 0x1C000) & 0x7C00) |
                ((float_bits >> 13) & 0x03FF)
            );
            aclrtMemcpy(scalar_factor_ptr, sizeof(uint16_t), &fp16_bits, sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
        }

        // 创建执行流
        aclrtCreateStream(&stream);

        // 关键修复：将单个张量包装为张量列表（匹配接口参数要求）
        // 输入张量列表（包含1个元素）
        aclTensor* input_tensors[] = {x.tensorPtr};
        input_list = aclCreateTensorList(input_tensors, 1);  // 直接用数组初始化列表
        
        // 输出张量列表（包含1个元素）
        aclTensor* output_tensors[] = {result.tensorPtr};
        output_list = aclCreateTensorList(output_tensors, 1);

        // 获取工作空间大小（使用张量列表作为参数）
        aclnnForeachMulScalarGetWorkspaceSize(
            input_list,          // 输入张量列表（第一个参数类型匹配）
            scalar_factor.tensorPtr,
            output_list,         // 输出张量列表
            &workspace_size,
            &executor
        );

        // 分配工作空间
        if (workspace_size > 0) {
            aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        // 执行标量乘法
        aclnnForeachMulScalar(
            workspace_addr,
            workspace_size,
            executor,
            stream
        );
        aclrtSynchronizeStream(stream);
    }
    catch (const std::exception& e) {
        // 释放资源（包含张量列表）
        if (workspace_addr != nullptr) aclrtFree(workspace_addr);
        if (input_list != nullptr) aclDestroyTensorList(input_list);
        if (output_list != nullptr) aclDestroyTensorList(output_list);
        if (stream != nullptr) aclrtDestroyStream(stream);
        throw;
    }

    return result;
}
