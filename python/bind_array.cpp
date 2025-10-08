#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <asnumpy/array/basic.hpp>

void bind_array(pybind11::module_& array) {
    array.doc() = "array module of asnumpy";
    array.def("zeros", &Zeros, py::arg("shape"), py::arg("dtype"));
    array.def("zeros_like", &Zeros_like, py::arg("other"), py::arg("dtype"));
    array.def("full", &Full, py::arg("shape"), py::arg("value"), py::arg("dtype"));
    array.def("full_like", &Full_like, py::arg("other"), py::arg("value"), py::arg("dtype"));
    array.def("empty", &Empty, py::arg("shape"), py::arg("dtype"));
    array.def("empty_like", &EmptyLike, py::arg("prototype"), py::arg("dtype")=py::none());
    array.def("eye", &Eye, py::arg("n"), py::arg("dtype"));
    array.def("ones", &Ones, py::arg("shape"), py::arg("dtype"));
    array.def("ones_like", &ones_like, py::arg("other"), py::arg("dtype"));
    array.def("identity", &Identity, py::arg("n"), py::arg("dtype"));
}
