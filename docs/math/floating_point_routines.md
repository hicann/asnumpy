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

## floating_point_routines  

**目前已完成的API：** [Signbit](#Signbit)  

<span id="Signbit">1. <mark>**Signbit**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组（数值类型）  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    逐元素判断输入值是否为负数，相当于 `numpy.signbit(x)`，返回与输入形状相同的布尔数组，值为 True 表示该元素符号位为 1（即负数）  
