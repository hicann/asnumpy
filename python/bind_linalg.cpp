#include <asnumpy/linalg/decompositions.hpp>
#include <asnumpy/linalg/norms.hpp>
#include <asnumpy/linalg/product.hpp>
#include <asnumpy/linalg/solving_inverting.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


// API with ap.linalg.xxxx usage
void bind_linalg(py::module_& linalg) {
	linalg.doc() = "linalg module of asnumpy";
	// product
	linalg.def("matrix_power", &Matrix_power, py::arg("a"), py::arg("n"));
	// decomposition
	linalg.def("qr", &Linalg_Qr, py::arg("a"), py::arg("mode") = "reduced");
	
	// norms
	linalg.def("norm", &Linalg_Norm, py::arg("a"), py::arg("ord") = py::none(), py::arg("axis") = py::none(),
			   py::arg("keepdims") = false);
	linalg.def("det", &Linalg_Det, py::arg("a"));
	linalg.def("slogdet", &Linalg_Slogdet, py::arg("a"));
	
	// solving equations
	linalg.def("inv", &Linalg_Inv, py::arg("a"));
	
}

// API with ap.xxxx usage
void bind_linalg_no_submodule(py::module_& linalg) {
	linalg.def("dot", &dot, py::arg("a"), py::arg("b"));
	linalg.def("inner", &inner, py::arg("a"), py::arg("b"));
	linalg.def("outer", &outer, py::arg("a"), py::arg("b"));
    linalg.def("vdot", &vdot, py::arg("a"), py::arg("b"));
	linalg.def("matmul", &Matmul, py::arg("x1"), py::arg("x2"));
	linalg.def("einsum", [](const char* subscripts, py::args operands) {
		std::vector<NPUArray> ops;
		for (auto op : operands) {
			ops.push_back(op.cast<NPUArray>());
		}
		return Einsum(subscripts, ops);
	}, py::arg("subscripts"));
}

