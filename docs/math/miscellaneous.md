/******************************************************************************
 * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

## miscellaneous  

**目前已完成的API：** [Clip (Tensor, Scalar, Scalar)](#Clip_Tensor_Scalar_Scalar), [Clip (Tensor, Scalar, Tensor)](#Clip_Tensor_Scalar_Tensor), [Clip (Tensor, Tensor, Scalar)](#Clip_Tensor_Tensor_Scalar), [Clip (Tensor, Tensor, Tensor)](#Clip_Tensor_Tensor_Tensor), [Fabs](#Fabs), [Fmax](#Fmax), [Fmin](#Fmin), [Maximum](#Maximum), [Minimum](#Minimum), [Nan_to_num](#Nan_to_num), [Square](#Square)  

<span id="Clip_Tensor_Scalar_Scalar">1. <mark>**Clip（Tensor, Scalar, Scalar）**</mark></span>  
- **参数：**  
    a：NPUArray，输入数组  
    a_min：py::object，最小值（标量）  
    a_max：py::object，最大值（标量）  
- **返回类型：**  
    NPUArray  
- **功能：**  
    将输入数组裁剪到区间 [a_min, a_max]。  

<span id="Clip_Tensor_Scalar_Tensor">2. <mark>**Clip（Tensor, Scalar, Tensor）**</mark></span>  
- **参数：**  
    a：NPUArray，输入数组  
    a_min：py::object，最小值（标量）  
    a_max：NPUArray，最大值  
- **返回类型：**  
    NPUArray  
- **功能：**  
    将输入数组逐元素裁剪，最小值为标量，最大值来自数组。  

<span id="Clip_Tensor_Tensor_Scalar">3. <mark>**Clip（Tensor, Tensor, Scalar）**</mark></span>  
- **参数：**  
    a：NPUArray，输入数组  
    a_min：NPUArray，最小值  
    a_max：py::object，最大值（标量）  
- **返回类型：**  
    NPUArray  
- **功能：**  
    将输入数组逐元素裁剪，最小值为数组，最大值为标量。  

<span id="Clip_Tensor_Tensor_Tensor">4. <mark>**Clip（Tensor, Tensor, Tensor）**</mark></span>  
- **参数：**  
    a：NPUArray，输入数组  
    a_min：NPUArray，最小值  
    a_max：NPUArray，最大值  
- **返回类型：**  
    NPUArray  
- **功能：**  
    将输入数组逐元素裁剪到对应区间。  

<span id="Fabs">5. <mark>**Fabs**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
- **返回类型：**  
    NPUArray  
- **功能：**  
    计算逐元素绝对值（等同 Absolute）。  

<span id="Fmax">6. <mark>**Fmax**</mark></span>  
- **参数：**  
    x1：NPUArray，输入数组  
    x2：NPUArray，输入数组  
    dtype：py::dtype，结果类型  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素取较大值（忽略 NaN 行为同 maximum）。  

<span id="Fmin">7. <mark>**Fmin**</mark></span>  
- **参数：**  
    x1：NPUArray，输入数组  
    x2：NPUArray，输入数组  
    dtype：py::dtype，结果类型  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素取较小值（忽略 NaN 行为同 minimum）。  

<span id="Maximum">8. <mark>**Maximum**</mark></span>  
- **参数：**  
    x1：NPUArray，输入数组  
    x2：NPUArray，输入数组  
    dtype：py::dtype，结果类型  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素返回最大值。  

<span id="Minimum">9. <mark>**Minimum**</mark></span>  
- **参数：**  
    x1：NPUArray，输入数组  
    x2：NPUArray，输入数组  
    dtype：py::dtype，结果类型  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素返回最小值。  

<span id="Nan_to_num">10. <mark>**Nan_to_num**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
    nan：double，用于替换 NaN 的值  
    posinf：py::object，替换正无穷  
    neginf：py::object，替换负无穷  
- **返回类型：**  
    NPUArray  
- **功能：**  
    替换数组中的 NaN、+inf 和 -inf。  

<span id="Square">11. <mark>**Square**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素计算平方。  
