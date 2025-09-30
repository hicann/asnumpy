from .asnumpy_core import *
from .asnumpy_core.math import * 
from .asnumpy_core.cann import * 
from .asnumpy_core import linalg  
# linalg模块内部分需要ap.linalg.xxx调用，部分ap.yyy调用，
# yyy类函数分到了.asnumpy_core根模块中


__all__ = [
    "ndarray",
    "init",
    "set_device",
    "broadcast_shape",
    "absolute",
    "fabs",
    "sign",
    "heaviside",
    "linalg",  # linalg整个子模块
    "dot",
    "vdot",
    "inner",
    "outer",
    "matmul",
    "einsum"
]



