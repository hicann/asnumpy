import asnumpy
import numpy as np

if __name__ == "__main__":
    print("Hello NPUArray\n")

    # test full int32
    # b_int = asnumpy.full(shape=(10, 10), dtype=np.dtype(np.int32), value=5)
    b_int = asnumpy.full(shape=(10, 10), dtype=np.int32, value=5)
    print("type(b_int):", type(b_int))
    b_int_cpu = b_int.to_numpy()
    print("b_int_cpu:\n", b_int_cpu)
    print("b_int_cpu.shape:", b_int_cpu.shape)
    # test full
    # b = asnumpy.full(shape=(10, 10), dtype=np.dtype(np.float32), value=3.14)
    b = asnumpy.full(shape=(10, 10), dtype=np.float32, value=3.14)
    print("type(b):", type(b))
    b_cpu = b.to_numpy()
    print("type(b_cpu):", type(b_cpu))
    print("b_cpu:\n", b_cpu)
    print("b_cpu.shape:", b_cpu.shape)

    # numpy -> NPUArray -> numpy
    # c = np.full((40, 50, 60), 5, dtype=np.dtype(np.int32))
    c = np.full((40, 50, 60), 5, dtype=np.int32)
    c_device = asnumpy.NPUArray.from_numpy(c)
    d = c_device.to_numpy()
    print("d.shape:", d.shape)
    print("type(d):", type(d))