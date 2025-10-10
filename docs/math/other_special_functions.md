/******************************************************************************
 * Copyright [2024]-[2025] [CANN/asnumpy] Authors. All Rights Reserved.
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

## sinc_functions  

**目前已完成的API：** [Sinc](#Sinc)  

<span id="Sinc">1. <mark>**Sinc**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素计算 Sinc 函数（即 `sin(x)/x` 的归一化形式），对应 `numpy.sinc(x)`。  
