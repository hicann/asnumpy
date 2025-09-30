#include <asnumpy/cann/driver.hpp>
#include <pybind11/pybind11.h>
#include <acl/acl.h>

void bind_cann(pybind11::module_& cann) {
    cann.doc() = "cann module of asnumpy";
    cann.def("set_device", &aclrtSetDevice, pybind11::arg("device_id"));
    cann.def("reset_device", &aclrtResetDevice, pybind11::arg("device_id"));
    cann.def("reset_device_force", &aclrtResetDeviceForce, pybind11::arg("device_id"));
    cann.def("init", &asnumpy::cann::init);
}