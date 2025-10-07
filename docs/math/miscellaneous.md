## miscellaneous

**目前已完成的API：**
 Clip (Tensor, Scalar, Scalar), Clip (Tensor, Scalar, Tensor), Clip (Tensor, Tensor, Scalar), Clip (Tensor, Tensor, Tensor), Fabs, Fmax, Fmin, Maximum, Minimum, Nan_to_num, Square

### 1. <mark>Clip（Tensor, Scalar, Scalar）</mark>

- **参数：**
   a：NPUArray，输入数组
   a_min：py::object，最小值（标量）
   a_max：py::object，最大值（标量）
- **返回类型：**
   NPUArray
- **功能：**
   将输入数组裁剪到区间 [a_min, a_max]。

### 2. <mark>Clip（Tensor, Scalar, Tensor）</mark>

- **参数：**
   a：NPUArray，输入数组
   a_min：py::object，最小值（标量）
   a_max：NPUArray，最大值
- **返回类型：**
   NPUArray
- **功能：**
   将输入数组逐元素裁剪，最小值为标量，最大值来自数组。

### 3. <mark>Clip（Tensor, Tensor, Scalar）</mark>

- **参数：**
   a：NPUArray，输入数组
   a_min：NPUArray，最小值
   a_max：py::object，最大值（标量）
- **返回类型：**
   NPUArray
- **功能：**
   将输入数组逐元素裁剪，最小值为数组，最大值为标量。

### 4. <mark>Clip（Tensor, Tensor, Tensor）</mark>

- **参数：**
   a：NPUArray，输入数组
   a_min：NPUArray，最小值
   a_max：NPUArray，最大值
- **返回类型：**
   NPUArray
- **功能：**
   将输入数组逐元素裁剪到对应区间。

### 5. <mark>Fabs</mark>

- **参数：**
   x：NPUArray，输入数组
- **返回类型：**
   NPUArray
- **功能：**
   计算逐元素绝对值（等同 Absolute）。

### 6. <mark>Fmax</mark>

- **参数：**
   x1：NPUArray，输入数组
   x2：NPUArray，输入数组
   dtype：py::dtype，结果类型
- **返回类型：**
   NPUArray
- **功能：**
   逐元素取较大值（忽略 NaN 行为同 maximum）。

### 7. <mark>Fmin</mark>

- **参数：**
   x1：NPUArray，输入数组
   x2：NPUArray，输入数组
   dtype：py::dtype，结果类型
- **返回类型：**
   NPUArray
- **功能：**
   逐元素取较小值（忽略 NaN 行为同 minimum）。

### 8. <mark>Maximum</mark>

- **参数：**
   x1：NPUArray，输入数组
   x2：NPUArray，输入数组
   dtype：py::dtype，结果类型
- **返回类型：**
   NPUArray
- **功能：**
   逐元素返回最大值。

### 9. <mark>Minimum</mark>

- **参数：**
   x1：NPUArray，输入数组
   x2：NPUArray，输入数组
   dtype：py::dtype，结果类型
- **返回类型：**
   NPUArray
- **功能：**
   逐元素返回最小值。

### 10. <mark>Nan_to_num</mark>

- **参数：**
   x：NPUArray，输入数组
   nan：double，用于替换 NaN 的值
   posinf：py::object，替换正无穷
   neginf：py::object，替换负无穷
- **返回类型：**
   NPUArray
- **功能：**
   替换数组中的 NaN、+inf 和 -inf。

### 11. <mark>Square</mark>

- **参数：**
   x：NPUArray，输入数组
- **返回类型：**
   NPUArray
- **功能：**
   逐元素计算平方。