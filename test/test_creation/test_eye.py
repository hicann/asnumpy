import asnumpy
import numpy as np

if __name__ == "__main__":
        print("Hello NPUArray eye test\n")
        n = 8
        # dtype = np.dtype(np.float32)
        dtype = np.float32
        # Create an identity matrix on the NPU
        eye_npu = asnumpy.eye(n, dtype=dtype)
        print("NPUArray type:", type(eye_npu))

        # Convert to numpy
        eye_cpu = eye_npu.to_numpy()
        print("Numpy type:", type(eye_cpu))
        print("Numpy array:\n", eye_cpu)
        print("Shape:", eye_cpu.shape)

        # Compare with numpy's built-in eye
        eye_np = np.eye(n, dtype=dtype)
        print("Numpy eye:\n", eye_np)
        print("Allclose:", np.allclose(eye_cpu, eye_np))
