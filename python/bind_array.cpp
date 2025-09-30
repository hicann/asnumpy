#include <pybind11/pybind11.h>

void bind_array(pybind11::module_& array) {
    array.doc() = "array module of asnumpy";

}
