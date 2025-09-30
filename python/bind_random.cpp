#include <pybind11/pybind11.h>

void bind_random(pybind11::module_& random) {
    random.doc() = "random module of asnumpy";
}