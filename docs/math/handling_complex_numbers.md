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

## handling_complex_numbers
**目前已完成的API：** [real](#real)  
  
<span id="real">1. <mark> **real** </mark></span>
- **参数：**  
    val：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 x 的实数部分，逐元素计算，如果val是实数，则直接输出该数。如果 val 是实数，则使用 val 的类型作为输出；如果 val 包含复数元素，则返回类型为浮点数。