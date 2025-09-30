import numpy as np
# 测试数据集 ===============================================

# 单操作数通用测试数据
UNARY_TEST_CASES = [
    np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32),
    np.array([-1.5, 2.7, -0.3, 4.0], dtype=np.float32),
    np.array([-10], dtype=np.float32),
    np.array([[1, -2], [-3, 4]], dtype=np.float32),
    np.array([-10, 0, 10], dtype=np.float32),
    np.array([[-1, 0], [0, 1]], dtype=np.float32),
    # 新增边缘测试用例
    np.array([np.inf, -np.inf, np.nan], dtype=np.float32),
    np.array([0.0, -0.0, 0.0], dtype=np.float32),  # 正负零混合
    # np.array([], dtype=np.float32),  # 空数组
    np.array([1e38, -1e38, 1e-38], dtype=np.float32),  # 极大极小值
    np.array([1.0, 1.0, 1.0], dtype=np.float32),  # 全正值
    np.array([-1.0, -1.0, -1.0], dtype=np.float32),  # 全负值
    np.array([[np.nan, 0], [0, np.inf]], dtype=np.float32),  # 特殊值矩阵
    np.array([2.2250738585072014e-308], dtype=np.float32)  # 最小正规格化浮点数
]

# 双操作数测试数据
BINARY_TEST_CASES = [
    (np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32),
     np.array([0.5], dtype=np.float32)),
    (np.array([-1.5, 0.0, 2.7], dtype=np.float32),
     np.array([1.0], dtype=np.float32)),
    (np.array([0, 0, 0], dtype=np.float32),
     np.array([0.5], dtype=np.float32)),
    (np.array([1, 2, 3], dtype=np.float32),
     np.array([0.0], dtype=np.float32)),
    (np.array([-1, -2, -3], dtype=np.float32),
     np.array([1.0], dtype=np.float32)),
    (np.array([[0, 1], [-1, 0]], dtype=np.float32),
     np.array([[0.5, 0.5]], dtype=np.float32)),
    # 新增边缘测试用例
    (np.array([np.nan, np.inf, -np.inf], dtype=np.float32),
     np.array([1.0, np.nan, 0.0], dtype=np.float32)),
    (np.array([0.0, 0.0, 0.0], dtype=np.float32),
     np.array([0.0, 1.0, -1.0], dtype=np.float32)),  # 零值处理
    (np.array([1e20, -1e20, 0], dtype=np.float32),
     np.array([1e-20, 1e20, 0], dtype=np.float32)),  # 大数和小数
    (np.array([], dtype=np.float32),
     np.array([], dtype=np.float32)),  # 空数组 npuarray暂不支持
    (np.array([[1, 2], [3, 4]], dtype=np.float32),
     np.array([[0], [0]], dtype=np.float32))  # 不同形状
]

# 广播测试数据
BROADCAST_TEST_CASES = [
    (np.array([[-1, 0, 1], [2, -2, 0]], dtype=np.float32),
     np.array([0.5, 1.0, 0.0], dtype=np.float32)),
    (np.array([1, 2, 3], dtype=np.float32),
     np.array([[0.5], [1.0]], dtype=np.float32)),
    (np.array([0], dtype=np.float32),
     np.array([0.1, 0.2, 0.3], dtype=np.float32)),
    # 新增边缘广播用例
    (np.array([[np.nan], [0], [1]], dtype=np.float32),
     np.array([0.5, np.inf, -1.0], dtype=np.float32)),
    (np.array([1e38, -1e38], dtype=np.float32),
     np.array([[0.0], [1.0]], dtype=np.float32)),
    (np.array([], dtype=np.float32),
    np.array([1.0, 2.0], dtype=np.float32))  # 空数组广播
]

# 矩阵乘法和点积用测试数据
MATMUL_DOT_TEST_CASES = [
    # 标量点积 (1x1)
    (np.array([2], dtype=np.float32),
     np.array([3], dtype=np.float32)),
    
    # 向量点积 (1D x 1D)
    (np.array([1, 2,9,6], dtype=np.float32),
     np.array([4, 5,10,8], dtype=np.float32)),
    
    # 矩阵乘法 (2D x 2D)
    (np.array([[1, 2], [3, 4]], dtype=np.float32),
     np.array([[5, 6], [7, 8]], dtype=np.float32)),
    
    # 高维张量乘法 (3D x 3D)
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
     np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float32)),
    
    # 边界情况：包含特殊值
    (np.array([[np.inf, 0], [np.nan, 1e38]], dtype=np.float32),
     np.array([[0, 1], [1, 0]], dtype=np.float32))
]

# 范数测试用例 (数组, ord, axis, keepdims)
NORM_TEST_CASES = [
    (np.array([1,2,3], dtype=np.float32), None, None, False),
    (np.array([[1,2],[3,4]], dtype=np.float32), 'fro', None, False),
    (np.array([[1,2],[3,4]], dtype=np.float32), 2, (0,1), True),
    (np.array([-1,0,1], dtype=np.float32), 1, None, False),
    (np.array([[np.nan,0],[0,1]], dtype=np.float32), None, None, False)
]

# QR分解测试用例 (数组, mode)
QR_TEST_CASES = [
    (np.array([[1,2],[3,4]], dtype=np.float32), 'complete'),
    (np.array([[1,2,3],[3,4,5],[5,6,7]], dtype=np.float32), 'complete'),
    (np.array([[0,1],[1,0]], dtype=np.float32), 'complete')
]

# Einsum测试用例 (表达式, 操作数列表)
EINSUM_TEST_CASES = [
    ('i,j->ij',np.array([1,2], dtype=np.float32),np.array([3,4], dtype=np.float32)),
]

# 矩阵乘方测试用例 (数组, 幂次)
MATRIX_POWER_TEST_CASES = [
    (np.array([[1,2],[3,4]], dtype=np.float32), 2),
    (np.array([[0,1],[1,0]], dtype=np.float32), 3),
    (np.array([[1,0],[0,1]], dtype=np.float32), 0),
    (np.array([[1,1],[0,1]], dtype=np.float32), -1)
]

# Inner函数测试数据
INNER_TEST_CASES = [
    # 1D vs 1D
    (np.array([1,2,3], dtype=np.float32),
     np.array([4,5,6], dtype=np.float32)),
     
    # 1D vs 2D
    (np.array([1,2], dtype=np.float32),
     np.array([[3,4],[5,6]], dtype=np.float32)),
     
    # 2D vs 1D
    (np.array([[1,2],[3,4]], dtype=np.float32),
     np.array([5,6], dtype=np.float32)),
     
    # 2D vs 2D
    (np.array([[1,2],[3,4]], dtype=np.float32),
     np.array([[5,6],[7,8]], dtype=np.float32)),
     
    # 包含特殊值
    (np.array([0,np.nan,1], dtype=np.float32),
     np.array([1,0,np.inf], dtype=np.float32)),
     
    # 不同形状
    (np.array([1,2,3,4], dtype=np.float32),
     np.array([5], dtype=np.float32))
]

# Outer函数测试数据
OUTER_TEST_CASES = [
    # 1D vs 1D
    (np.array([1,2,3], dtype=np.float32),
     np.array([4,5,6], dtype=np.float32)),
     
    # 1D vs 2D
    (np.array([1,2], dtype=np.float32),
     np.array([[3,4],[5,6]], dtype=np.float32)),
     
    # 2D vs 1D
    (np.array([[1,2],[3,4]], dtype=np.float32),
     np.array([5,6], dtype=np.float32)),
     
    # 包含特殊值
    (np.array([0,np.nan], dtype=np.float32),
     np.array([1,np.inf], dtype=np.float32)),
     
    # 标量
    (np.array([1], dtype=np.float32),
     np.array([2], dtype=np.float32))
]

# 行列式和符号对数行列式专用测试数据 (方阵)
DET_SLOGDET_TEST_CASES = [
    # 2x2矩阵
    np.array([[1, 2], [3, 4]], dtype=np.float32),  
    # 3x3矩阵
    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
    # 特殊值矩阵
    np.array([[np.nan, 0], [np.inf, 1]], dtype=np.float32),    
    # 奇异矩阵 (行列式=0)
    np.array([[1, 2], [2, 4]], dtype=np.float32),
]
