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
 *****************************************************************************/

#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <utility>

NPUArray Linalg_Norm(const NPUArray& a, double ord, const std::vector<int64_t>& axis, bool keepdims);

NPUArray Linalg_Det(const NPUArray& a);

std::vector<NPUArray> Linalg_Slogdet(const NPUArray& a);