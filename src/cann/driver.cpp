#include "asnumpy/cann/driver.hpp"
#include "fmt/base.h"

void asnumpy::cann::init() {
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        auto message = aclGetRecentErrMsg();
        fmt::println("{}", message);
    }
}