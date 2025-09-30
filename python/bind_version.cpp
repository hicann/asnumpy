#include <pybind11/pybind11.h>

void bind_version(pybind11::module_& version) {
    version.doc() = "version module of asnumpy";
}