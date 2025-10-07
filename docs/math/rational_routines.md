## rational_routines

**目前已完成的API：**
 Gcd, Lcm

------

### 1. <mark>Gcd</mark>

- **参数：**
   x1：NPUArray，输入数组（整数类型）
   x2：NPUArray，输入数组（整数类型）
- **返回类型：**
   NPUArray
- **功能：**
   逐元素计算最大公约数（Greatest Common Divisor），等同于 `numpy.gcd(x1, x2)`。

### 2. <mark>Lcm</mark>

- **参数：**
   x1：NPUArray，输入数组（整数类型）
   x2：NPUArray，输入数组（整数类型）
- **返回类型：**
   NPUArray
- **功能：**
   逐元素计算最小公倍数（Least Common Multiple），等同于 `numpy.lcm(x1, x2)`，实现方式为：
   `LCM(a, b) = |a * b| / GCD(a, b)`。