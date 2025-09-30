#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <asnumpy/math/trigonometric_functions.hpp>
#include <asnumpy/math/hyperbolic_functions.hpp>
#include <asnumpy/math/rounding.hpp>
#include <asnumpy/math/sums_products_differences.hpp>
#include <asnumpy/math/exponents_and_logarithms.hpp>
#include <asnumpy/math/other_special_functions.hpp>
#include <asnumpy/math/floating_point_routines.hpp>
#include <asnumpy/math/rational_routines.hpp>
#include <asnumpy/math/arithmetic_operations.hpp>
#include <asnumpy/math/handling_complex_numbers.hpp>
#include <asnumpy/math/miscellaneous.hpp>


namespace py = pybind11;

void bind_trigonometric_functions(py::module_& math);
void bind_hyperbolic_functions(py::module_& math);
void bind_rounding(py::module_& math);
void bind_sums_products_differences(py::module_& math);
void bind_exponents_and_logarithms(py::module_& math);
void bind_other_special_functions(py::module_& math);
void bind_floating_point_routines(py::module_& math);
void bind_rational_routines(py::module_& math);
void bind_arithmetic_operations(py::module_& math);
void bind_handling_complex_numbers(py::module_& math);
void bind_miscellaneous(py::module_& math);

void bind_math(py::module_& math) {
    math.doc() = "math module of asnumpy";
    //bind_trigonometric_functions(math);
    //bind_hyperbolic_functions(math);
    //bind_rounding(math);
    //bind_sums_products_differences(math);
    //bind_exponents_and_logarithms(math);
    //bind_other_special_functions(math);
    //bind_floating_point_routines(math);
    //bind_rational_routines(math);
    //bind_arithmetic_operations(math);
    //bind_handling_complex_numbers(math);
    bind_miscellaneous(math);
}


// void bind_trigonometric_functions(py::module_& math){
    
//     math.def("sin", &Sin, py::arg("x"));
//     math.def("cos", &Cos, py::arg("x"));
//     math.def("tan", &Tan, py::arg("x"));
//     math.def("arcsin",&Arcsin, py::arg("x"));
//     math.def("arccos",&Arccos, py::arg("x"));
//     math.def("arctan",&Arctan, py::arg("x"));
//     math.def("arctan2",&Arctan2, py::arg("x1"), py::arg("x2"));
//     math.def("hypot",&Hypot, py::arg("x1"), py::arg("x2"));
//     math.def("radians",&Radians, py::arg("x"));
// }


void bind_miscellaneous(py::module_& math){
    math.def("absolute", &Absolute, py::arg("x"));
    math.def("fabs", &Fabs, py::arg("x"));
    math.def("sign", &Sign, py::arg("x"));
    math.def("heaviside",&Heaviside, py::arg("x1"), py::arg("x2"));
}