#include <pybind11/pybind11.h>
#include <asnumpy/dtypes/acl_float_reg.hpp>
// 前向声明，避免额外头文件
namespace asnumpy { namespace dtypes { void InitAndRegisterDtypes(); } }

using namespace asnumpy::dtypes;

void bind_dtypes(pybind11::module_& dtypes) {
    dtypes.doc() = "ACL custom dtypes for NumPy";
    
    // 初始化并注册所有 dtypes（幂等且仅导入一次 NumPy C API）
    asnumpy::dtypes::InitAndRegisterDtypes();

    // 绑定所有注册的浮点类型对象到 Python 模块
    if (ACLFloatManager<float8_e5m2>::type_ptr != nullptr) {
        dtypes.attr("float8_e5m2") = pybind11::reinterpret_borrow<pybind11::object>(
            ACLFloatManager<float8_e5m2>::type_ptr);
    }
    
    if (ACLFloatManager<float8_e4m3fn>::type_ptr != nullptr) {
        dtypes.attr("float8_e4m3fn") = pybind11::reinterpret_borrow<pybind11::object>(
            ACLFloatManager<float8_e4m3fn>::type_ptr);
    }
    
    if (ACLFloatManager<float8_e8m0>::type_ptr != nullptr) {
        dtypes.attr("float8_e8m0") = pybind11::reinterpret_borrow<pybind11::object>(
            ACLFloatManager<float8_e8m0>::type_ptr);
    }
    
    if (ACLFloatManager<bfloat16>::type_ptr != nullptr) {
        dtypes.attr("bfloat16") = pybind11::reinterpret_borrow<pybind11::object>(
            ACLFloatManager<bfloat16>::type_ptr);
    }
    
    if (ACLFloatManager<float6_e2m3fn>::type_ptr != nullptr) {
        dtypes.attr("float6_e2m3fn") = pybind11::reinterpret_borrow<pybind11::object>(
            ACLFloatManager<float6_e2m3fn>::type_ptr);
    }
    
    if (ACLFloatManager<float6_e3m2fn>::type_ptr != nullptr) {
        dtypes.attr("float6_e3m2fn") = pybind11::reinterpret_borrow<pybind11::object>(
            ACLFloatManager<float6_e3m2fn>::type_ptr);
    }
    
    if (ACLFloatManager<float4_e2m1fn>::type_ptr != nullptr) {
        dtypes.attr("float4_e2m1fn") = pybind11::reinterpret_borrow<pybind11::object>(
            ACLFloatManager<float4_e2m1fn>::type_ptr);
    }
}