#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <asnumpy/utils/npu_array.hpp>

void bind_utils(pybind11::module_& utils) {
    pybind11::class_<NPUArray>(utils, "ndarray")
        .def(py::init<const std::vector<int64_t>&, py::dtype>(),
            py::arg("shape"), py::arg("dtype"),
            "Constructs an empty NPUArray with the given shape and dtype.")
        .def(py::init<const NPUArray&>(), "Copy constructor for NPUArray")
        .def("to_numpy", &NPUArray::ToNumpy)
        .def_static("from_numpy", &NPUArray::FromNumpy, py::arg("host_data"))
        .def_property_readonly("shape", [](const NPUArray& self) { return self.shape; })
        .def_property_readonly("dtype", [](const NPUArray& self) { return self.dtype; })
        .def_property_readonly("aclDtype", [](const NPUArray& self) { return static_cast<int>(self.aclDtype); });
    utils.def("broadcast_shape", &GetBroadcastShape, py::arg("a"), py::arg("b"));
}