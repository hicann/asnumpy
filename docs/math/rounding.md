## rounding

**目前已完成的API：**
 Around, Ceil, Fix, Floor, Rint, Round, Trunc

### 1. <mark>Around</mark>

- **参数：**
   x：NPUArray，输入数组
   decimals：int，要四舍五入的小数位数；若为负数，则表示在小数点左侧取整
   dtype：py::dtype，可选，返回类型，默认与 x 相同
- **返回类型：**
   NPUArray
- **功能：**
   逐元素将 x 四舍五入到指定小数位数。

### 2. <mark>Ceil</mark>

- **参数：**
   x：NPUArray，输入数组
   dtype：py::dtype，可选，返回类型，默认与 x 相同
- **返回类型：**
   NPUArray
- **功能：**
   逐元素返回大于或等于 x 的最小整数（向上取整）。

### 3. <mark>Fix</mark>

- **参数：**
   x：NPUArray，输入数组
   dtype：py::dtype，可选，返回类型，默认与 x 相同
- **返回类型：**
   NPUArray
- **功能：**
   逐元素向零取整，截取整数部分。

### 4. <mark>Floor</mark>

- **参数：**
   x：NPUArray，输入数组
   dtype：py::dtype，可选，返回类型，默认与 x 相同
- **返回类型：**
   NPUArray
- **功能：**
   逐元素返回小于或等于 x 的最大整数（向下取整）。

### 5. <mark>Rint</mark>

- **参数：**
   x：NPUArray，输入数组
   dtype：py::dtype，可选，返回类型，默认与 x 相同
- **返回类型：**
   NPUArray
- **功能：**
   逐元素将 x 四舍五入到最近的整数。

### 6. <mark>Round</mark>

- **参数：**
   x：NPUArray，输入数组
   decimals：int，要四舍五入的小数位数；若为负数，则表示在小数点左侧取整
   dtype：py::dtype，可选，返回类型，默认与 x 相同
- **返回类型：**
   NPUArray
- **功能：**
   与 Around 相同，逐元素执行四舍五入。

### 7. <mark>Trunc</mark>

- **参数：**
   x：NPUArray，输入数组
   dtype：py::dtype，可选，返回类型，默认与 x 相同
- **返回类型：**
   NPUArray
- **功能：**
   逐元素截断小数部分，返回离零更近的整数值。