import asnumpy as ap
import numpy as np

if __name__ == "__main__":

    a = ap.ones((1024, 1024, 1024), dtype=np.dtype(np.int32))
    b = ap.full((1024, 1024, 1024), value=6, dtype=np.dtype(np.int32))
    ap.print(a)
    ap.print(b)
    c = ap.add(a, b)
    ap.print(c)
    c = ap.sub(a, b)
    ap.print(c)
