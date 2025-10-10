# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

import numpy as np
from typing import Optional

def init(configPath=None): ...

def aclrt_set_device(device_id: int) -> int: ...

class NPUArray:
    def __init__(self, shape: list, dtype: np.dtype) -> None: ...
    def to_numpy(self) -> np.ndarray: ...
    
    @staticmethod
    def from_numpy(host_data: np.ndarray) -> NPUArray: ...


def ones(shape: list, dtype: np.dtype) -> NPUArray: ...

def zeros(shape: list, dtype: np.dtype) -> NPUArray: ...

def add(a: NPUArray, b: NPUArray) -> NPUArray: ...

def sub(a: NPUArray, b: NPUArray) -> NPUArray: ...

def print(a: NPUArray) -> None: ...

def eye(n: int, dtype: np.dtype) -> NPUArray: ...

def full(shape: list, dtype: np.dtype, value: int) -> NPUArray: ...

def empty(shape: list, dtype: np.dtype) -> NPUArray: ...

def arange(start: float, stop: float, step: float = 1.0, dtype: Optional[np.dtype] = None) -> NPUArray: ...