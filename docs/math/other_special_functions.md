**目前已完成的API：**
 sinc

### 1. <mark>sinc</mark>

- **参数：**
   x：NPUArray，输入数组
- **返回类型：**
   NPUArray
- **功能：**
   逐元素计算 sinc 函数（即 `sin(x)/x` 的归一化形式），对应 numpy.sinc，实现基于 `aclnnSinc`。