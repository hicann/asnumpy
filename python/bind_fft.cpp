#include <pybind11/pybind11.h>

void bind_fft(pybind11::module_& fft) {
    fft.doc() = "fft module of asnumpy";
}