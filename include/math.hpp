/******************************************************************************
 * Copyright [2024]-[2025] [HIT1920/asnumpy] Authors. All Rights Reserved.
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
 *****************************************************************************/
 
#pragma once

#include "array.hpp"
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_sub.h>

/**
 * @brief Perform element-wise addition of two NPUArray objects.
 *
 * This function adds two NPUArray objects element-wise and returns a new NPUArray containing the result.
 *
 * @param a First NPUArray
 * @param b Second NPUArray
 * @return NPUArray Resulting NPUArray after element-wise addition
 */
NPUArray Add(const NPUArray& a, const NPUArray& b);
NPUArray NewAdd(const NPUArray& a, const NPUArray& b);

// *** TODO ***

// Math operations
NPUArray Subtract(const NPUArray& a, const NPUArray& b);
NPUArray Multiply(const NPUArray& a, const NPUArray& b);
NPUArray Matmul(const NPUArray& a, const NPUArray& b);
NPUArray Divide(const NPUArray& a, const NPUArray& b);
NPUArray Logaddexp(const NPUArray& a, const NPUArray& b);
NPUArray Logaddexp2(const NPUArray& a, const NPUArray& b);
NPUArray TrueDivide(const NPUArray& a, const NPUArray& b);
NPUArray FloorDivide(const NPUArray& a, const NPUArray& b);
NPUArray Power(const NPUArray& a, const NPUArray& b);
NPUArray FloatPower(const NPUArray& a, const NPUArray& b);
NPUArray Remainder(const NPUArray& a, const NPUArray& b);
NPUArray Mod(const NPUArray& a, const NPUArray& b);
NPUArray FMod(const NPUArray& a, const NPUArray& b);
NPUArray GCD(const NPUArray& a, const NPUArray& b);
NPUArray LCM(const NPUArray& a, const NPUArray& b);

NPUArray Negative(const NPUArray& a);
NPUArray Positive(const NPUArray& a);
NPUArray Exp(const NPUArray& a);
NPUArray Exp2(const NPUArray& a);
NPUArray Log(const NPUArray& a);
NPUArray Log2(const NPUArray& a);
NPUArray Log10(const NPUArray& a);
NPUArray Sqrt(const NPUArray& a);


// Trigonometric functions
NPUArray Sin(const NPUArray& a);
NPUArray Cos(const NPUArray& a);
NPUArray Tan(const NPUArray& a);
NPUArray Arcsin(const NPUArray& a);
NPUArray Arccos(const NPUArray& a);
NPUArray Arctan(const NPUArray& a);
NPUArray Sinh(const NPUArray& a);
NPUArray Cosh(const NPUArray& a);
NPUArray Tanh(const NPUArray& a);


/**
 * @brief Print the contents of an NPUArray.
 *
 * This function copies the data from the NPUArray to the host, converts it to a NumPy array, and prints the first 10 elements.
 *
 * @param a NPUArray object to be printed
 */
void Print(const NPUArray& a);
