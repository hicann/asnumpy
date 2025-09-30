import numpy as np
import asnumpy as ap  

def check_case(shapes):
    arrays = [ap.ndarray.from_numpy(np.zeros(s, dtype=np.int32)) for s in shapes]

    try:
        result = ap.broadcast_shape(*arrays)
        result_shape = tuple(result)
    except Exception as e:
        result_shape = f"Exception: {e}"

    try:
        expected = np.broadcast_shapes(*shapes)
    except Exception as e:
        expected = f"Exception: {e}"

    print(f"shapes={shapes} -> ap: {result_shape}, numpy: {expected}")

def main():
    test_shapes = [
        [(2, 3), (3,)],          # 一维 + 二维
        [(3,), (3, 1)],          # 不同维度数
        [(3,), ()],              # 标量参与广播
        [(4, 1, 3), (1, 5, 1)],  # 高维广播
        [(2, 3), (2, 3)],        # 相同形状
        [(2, 3), (2, 4)],        # 不可广播
    ]

    for shapes in test_shapes:
        check_case(shapes)

if __name__ == "__main__":
    main()
