from test_data import UNARY_TEST_CASES, BINARY_TEST_CASES, BROADCAST_TEST_CASES, MATMUL_DOT_TEST_CASES

import numpy as np
import asnumpy as ap

# 函数注册表 =================================================

# 单操作数函数注册表 (函数名, numpy函数, asnumpy函数, 测试用例列表)
# 使用共享数据集UNARY_TEST_CASES，可额外添加自定义用例
UNARY_FUNCTIONS = [
    (
        "absolute",
        np.absolute,
        ap.absolute,
        UNARY_TEST_CASES
    ),
    (
        "sign",
        np.sign,
        ap.sign,
        UNARY_TEST_CASES
    ),
    (
        "fabs",
        np.fabs,
        ap.fabs,
        UNARY_TEST_CASES
    ),
]

# 双操作数函数注册表 (函数名, numpy函数, asnumpy函数, 测试用例列表)
# 基础用例BINARY_TEST_CASES + 广播用例BROADCAST_TEST_CASES
BINARY_FUNCTIONS = [
    (
        "heaviside",
        np.heaviside,
        ap.heaviside,
        BINARY_TEST_CASES + BROADCAST_TEST_CASES
    ),
    (
        "matmul",
        np.matmul,
        ap.matmul,
        MATMUL_DOT_TEST_CASES
    ),
    (
        "dot",
        np.dot,
        ap.dot,
        MATMUL_DOT_TEST_CASES
    )
]
