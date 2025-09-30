#include <pybind11/pybind11.h>

void bind_testing(pybind11::module_& testing) {
    testing.doc() = "testing module of asnumpy";
}