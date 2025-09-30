/******************************************************************************
 * Copyright [2024]-[2025] [HIT1920/asnumpy] Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
 

#include <pybind11/detail/common.h>
#include <pybind11/stl.h>
#include "array.hpp"
#include "creation.hpp"
#include "math.hpp"

namespace py = pybind11;

/**
 * @brief Define NPUArray module bindings for Python interface.
 * 
 * Contains Python binding definitions for the NPUArray module using pybind11 library.
 * This module provides Python interface to NPU array operations and ACL environment management.
 * 
 * @param m pybind11 module object for defining module contents.
 */
PYBIND11_MODULE(NPUArray, m) {
    /**
     * @brief Initialize ACL environment for NPU operations.
     * 
     * Initializes the ACL (Ascend Computing Language) environment required for NPU operations.
     * This function must be called before any NPU operations.
     * 
     * @param configPath Path to ACL configuration file, defaults to nullptr.
     */
    m.def("init", &aclInit, py::arg("configPath") = nullptr);
    /**
     * @brief Set current NPU device for operations.
     * 
     * Sets the current NPU device to use for subsequent operations.
     * 
     * @param deviceId Device ID of the NPU to use.
     */
    m.def("aclrt_set_device", &aclrtSetDevice, pybind11::arg("device_id"));
    
    // 暴露 ACL 数据类型常量（按照 ACL 枚举定义顺序）
    m.attr("dt_undefined") = static_cast<int>(ACL_DT_UNDEFINED);
    m.attr("float32") = static_cast<int>(ACL_FLOAT);
    m.attr("float16") = static_cast<int>(ACL_FLOAT16);
    m.attr("int8") = static_cast<int>(ACL_INT8);
    m.attr("int32") = static_cast<int>(ACL_INT32);
    m.attr("uint8") = static_cast<int>(ACL_UINT8);
    m.attr("int16") = static_cast<int>(ACL_INT16);
    m.attr("uint16") = static_cast<int>(ACL_UINT16);
    m.attr("uint32") = static_cast<int>(ACL_UINT32);
    m.attr("int64") = static_cast<int>(ACL_INT64);
    m.attr("uint64") = static_cast<int>(ACL_UINT64);
    m.attr("float64") = static_cast<int>(ACL_DOUBLE);
    m.attr("bool") = static_cast<int>(ACL_BOOL);
    m.attr("string") = static_cast<int>(ACL_STRING);
    m.attr("complex64") = static_cast<int>(ACL_COMPLEX64);
    m.attr("complex128") = static_cast<int>(ACL_COMPLEX128);
    m.attr("bf16") = static_cast<int>(ACL_BF16);
    m.attr("int4") = static_cast<int>(ACL_INT4);
    m.attr("uint1") = static_cast<int>(ACL_UINT1);
    m.attr("complex32") = static_cast<int>(ACL_COMPLEX32);
    m.attr("hifloat8") = static_cast<int>(ACL_HIFLOAT8);
    m.attr("float8_e5m2") = static_cast<int>(ACL_FLOAT8_E5M2);
    m.attr("float8_e4m3fn") = static_cast<int>(ACL_FLOAT8_E4M3FN);
    m.attr("float8_e8m0") = static_cast<int>(ACL_FLOAT8_E8M0);
    m.attr("float6_e3m2") = static_cast<int>(ACL_FLOAT6_E3M2);
    m.attr("float6_e2m3") = static_cast<int>(ACL_FLOAT6_E2M3);
    m.attr("float4_e2m1") = static_cast<int>(ACL_FLOAT4_E2M1);
    m.attr("float4_e1m2") = static_cast<int>(ACL_FLOAT4_E1M2);
    /**
     * @brief Bind NPUArray class to Python interface.
     * 
     * Defines the NPUArray class constructor, to_numpy method, and from_numpy static method
     * for Python users to create and manipulate NPU arrays.
     */
    py::class_<NPUArray>(m, "NPUArray")
        .def(py::init<const std::vector<int64_t>&, py::dtype>(),
            py::arg("shape"), py::arg("dtype"),
            "Constructs an empty NPUArray with the given shape and dtype.")
        .def(py::init<const NPUArray&>(), "Copy constructor for NPUArray")
        .def("to_numpy", &NPUArray::ToNumpy)
        .def_static("from_numpy", &NPUArray::FromNumpy, py::arg("host_data"))
        .def_property_readonly("shape", [](const NPUArray& self) { return self.shape; })
        .def_property_readonly("dtype", [](const NPUArray& self) { return self.dtype; })
        .def_property_readonly("aclDtype", [](const NPUArray& self) { return static_cast<int>(self.aclDtype); });
    /**
     * @brief Create an array filled with ones.
     * 
     * Creates an array stored on NPU filled with ones.
     * 
     * @param shape Vector containing array dimensions, defining the array shape.
     * @param dtype Data type defining the elements in the array.
     */
    m.def("ones", py::overload_cast<const std::vector<int64_t>&, py::dtype>(&Ones), 
          py::arg("shape"), py::arg("dtype"));
    m.def("ones", py::overload_cast<const std::vector<int64_t>&, py::object>(&Ones), 
          py::arg("shape"), py::arg("dtype"));
    /**
     * @brief Create an array filled with zeros.
     * 
     * Creates an array stored on NPU filled with zeros.
     * 
     * @param shape Vector containing array dimensions, defining the array shape.
     * @param dtype Data type defining the elements in the array.
     */
    m.def("zeros", py::overload_cast<const std::vector<int64_t>&, py::dtype>(&Zeros), 
          py::arg("shape"), py::arg("dtype"));
    m.def("zeros", py::overload_cast<const std::vector<int64_t>&, py::object>(&Zeros), 
          py::arg("shape"), py::arg("dtype"));
    m.def("add", &Add, py::arg("a"), py::arg("b"));
	m.def("sub", &Subtract, py::arg("a"), py::arg("b"));

    m.def("print", &Print, py::arg("a"));
    /**
     * @brief Create an array filled with specified value of given shape and data type.
     * 
     * Creates an array stored on NPU with all elements initialized to the specified scalar value.
     * 
     * @param shape Vector containing array dimensions, defining the array shape.
     * @param dtype Data type defining the elements in the array.
     * @param value Scalar value used to fill the array.
     */
    m.def("full", py::overload_cast<const std::vector<int64_t>&, py::dtype, const py::object&>(&Full), 
          py::arg("shape"), py::arg("dtype"), py::arg("value"));
    m.def("full", py::overload_cast<const std::vector<int64_t>&, py::object, const py::object&>(&Full), 
          py::arg("shape"), py::arg("dtype"), py::arg("value"));
    /**
     * @brief Create identity matrix (square matrix with ones on main diagonal).
     * 
     * Creates a square array (matrix) on NPU where elements on the main diagonal are 1,
     * and all other elements are 0.
     * 
     * @param n Number of rows and columns of output array (dimension of square matrix).
     * @param dtype Data type defining the elements in the array.
     */
    m.def("eye", py::overload_cast<int64_t, py::dtype>(&Eye), 
          py::arg("n"), py::arg("dtype"));
    m.def("eye", py::overload_cast<int64_t, py::object>(&Eye), 
          py::arg("n"), py::arg("dtype"));
    /**
     * @brief Create an uninitialized array.
     * 
     * Creates an array stored on NPU with uninitialized elements.
     * 
     * @param shape Vector containing array dimensions, defining the array shape.
     * @param dtype Data type defining the elements in the array.
     */
    m.def("empty", py::overload_cast<const std::vector<int64_t>&, py::dtype>(&Empty), 
          py::arg("shape"), py::arg("dtype"));
    m.def("empty", py::overload_cast<const std::vector<int64_t>&, py::object>(&Empty), 
          py::arg("shape"), py::arg("dtype"));
    /**
     * @brief Create an arithmetic sequence array.
     * 
     * Creates an arithmetic sequence array stored on NPU, similar to NumPy's arange function.
     * 
     * @param start Starting value (inclusive)
     * @param stop Ending value (exclusive)
     * @param step Step size (default 1)
     * @param dtype Data type defining the elements in the array.
     */
    m.def("arange", py::overload_cast<double, double, double, py::dtype>(&Arange), 
          py::arg("start"), py::arg("stop"), py::arg("step") = 1.0, py::arg("dtype"));
    m.def("arange", py::overload_cast<double, double, double, py::object>(&Arange), 
          py::arg("start"), py::arg("stop"), py::arg("step") = 1.0, py::arg("dtype"));
}
