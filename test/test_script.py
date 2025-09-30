import numpy as np
import asnumpy as ap
from script_test_cases import *
"""
How to use:

在下面的XXX_FUNCTIONS中按模块增删你想要测的API
然后运行python3 test/test_script.py检查输出结果

"""
# ====================API注册表========================
# 格式：(函数名, numpy函数, asnumpy函数, 测试用例列表)

# 测试用例列表目前位于test/script_test_cases.py 由于不同API要求输入的测试数据格式不同，之后请根据自身需求增添
# 目前已有的：
# UNARY_TEST_CASES: 单操作数测试数据集
# BINARY_TEST_CASES: 普通双操作数测试数据集（加减乘除等
# BROADCAST_TEST_CASES: 双操作数广播测试数据集
# MATMUL_DOT_TEST_CASES、MATRIX_POWER_TEST_CASES: 矩阵乘法等对输入张量格式有要求的数据集

MATH_FUNCTIONS = [
    ("fabs", np.fabs, ap.fabs, UNARY_TEST_CASES),
    ("absolute", np.absolute, ap.absolute, UNARY_TEST_CASES),
    ("sign", np.sign, ap.sign, UNARY_TEST_CASES),
    ("heaviside", np.heaviside, ap.heaviside, BINARY_TEST_CASES + BROADCAST_TEST_CASES),
]

LINALG_FUNCTIONS = [
    ("qr", np.linalg.qr, ap.linalg.qr, QR_TEST_CASES),
    ("norm", np.linalg.norm, ap.linalg.norm, NORM_TEST_CASES),
    ("det",np.linalg.det,ap.linalg.det,DET_SLOGDET_TEST_CASES),
    ("slogdet",np.linalg.slogdet,ap.linalg.slogdet,DET_SLOGDET_TEST_CASES),
    ("matmul", np.matmul, ap.matmul, MATMUL_DOT_TEST_CASES),
    ("einsum", np.einsum, ap.einsum, EINSUM_TEST_CASES),
    ("matrix_power", np.linalg.matrix_power, ap.linalg.matrix_power, MATRIX_POWER_TEST_CASES),
    ("dot", np.dot, ap.dot, MATMUL_DOT_TEST_CASES),
    ("vdot", np.vdot, ap.vdot, MATMUL_DOT_TEST_CASES),
    ("inner", np.inner, ap.inner, INNER_TEST_CASES),
    ("outer", np.outer, ap.outer, OUTER_TEST_CASES),
    ("inv",np.linalg.inv,ap.linalg.inv,DET_SLOGDET_TEST_CASES)
]
# 总表
FUNCTIONS_TABLE = MATH_FUNCTIONS + LINALG_FUNCTIONS

# =========================测试函数本体======================

def test_functions():
    
    for name, np_func, ap_func, test_cases in FUNCTIONS_TABLE:
        print("=" * 50)
        print(f"Testing {name} function:")
        print("=" * 50)
        
        passed = 0
        total = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            # 将测试用例元组解包为参数
            try:
                # 处理单参数和多参数用例
                if isinstance(test_case, tuple):
                    # 参数里任意位置可能出现ndarray，需要转换为npuarray供asnumpy调用
                    converted_args = tuple(
                        ap.ndarray.from_numpy(arg) if isinstance(arg, np.ndarray) else arg
                        for arg in test_case
                    )
                    np_result = np_func(*test_case)
                    ap_result = ap_func(*converted_args)
                else:
                    # 单个参数的情况
                    converted_arg = ap.ndarray.from_numpy(test_case) if isinstance(test_case, np.ndarray) else test_case
                    np_result = np_func(test_case)
                    ap_result = ap_func(converted_arg)
                
                # 转换结果为numpy数组
                if hasattr(ap_result, 'to_numpy'):
                    ap_result_np = ap_result.to_numpy()
                else:
                    ap_result_np = ap_result
                
                # 对于多个返回值的函数（如qr）
                if isinstance(np_result, tuple) and isinstance(ap_result_np, tuple):
                    all_close = True
                    for np_res, ap_res in zip(np_result, ap_result_np):
                        if not np.allclose(np_res, ap_res.to_numpy() if hasattr(ap_res, 'to_numpy') else ap_res, equal_nan=True):
                            all_close = False
                    if not all_close:
                        print(f"Test {i+1} FAILED:")
                        print(f"  Input: {test_case}")
                        print(f"  NumPy result: {np_result}")
                        print(f"  AP result: {ap_result_np}")
                        print()
                    else:
                        passed += 1
                else:
                    if not np.allclose(np_result, ap_result_np, equal_nan=True):
                        print(f"Test {i+1} FAILED:")
                        print(f"  Input: {test_case}")
                        print(f"  NumPy result: {np_result}")
                        print(f"  AP result: {ap_result_np}")
                        print()
                    else:
                        passed += 1
            except Exception as e:
                print(f"Test {i+1} ERROR:")
                print(f"  Input: {test_case}")
                print(f"  Error: {e}")
                print()
        
        print(f"Passed: {passed}/{total} tests")
        print()
        

def main():
    print("Testing NPU Math Functions")
    print("=" * 60)
    print()
    
    test_functions()
    
    print("=" * 60)
    print("All tests completed")

if __name__ == "__main__":
    main()