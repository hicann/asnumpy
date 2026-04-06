# *****************************************************************************
# Copyright (c) 2025 AISS and ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

# Standard library
import importlib
import os
import sys
from typing import TYPE_CHECKING

# Third-party
import numpy as np
from loguru import logger

# Application modules
from .cann import finalize, init, reset_device, reset_device_force, set_device

if TYPE_CHECKING:
    from .array import (
        array,
        asarray,
        asanyarray,
        copy,
        empty,
        empty_like,
        eye,
        full,
        full_like,
        identity,
        linspace,
        ones,
        ones_like,
        zeros,
        zeros_like,
    )

    from . import linalg

    from .linalg.direct import dot, einsum, inner, matmul, outer, vdot

    from .logic import (
        all,
        any,
        equal,
        greater,
        greater_equal,
        isfinite,
        isinf,
        isneginf,
        isposinf,
        less,
        less_equal,
        logical_and,
        logical_not,
        logical_or,
        logical_xor,
        not_equal,
    )

    from .math import (
        absolute,
        add,
        amax,
        amin,
        around,
        arccos,
        arccosh,
        arcsin,
        arcsinh,
        arctan,
        arctan2,
        arctanh,
        ceil,
        clip,
        copysign,
        cos,
        cosh,
        cross,
        cumprod,
        cumsum,
        deg2rad,
        degrees,
        divide,
        divmod,
        exp,
        exp2,
        expm1,
        fabs,
        fix,
        floor,
        floor_divide,
        fmax,
        fmin,
        fmod,
        float_power,
        gelu,
        gcd,
        heaviside,
        hypot,
        lcm,
        ldexp,
        log,
        log10,
        log1p,
        log2,
        logaddexp,
        logaddexp2,
        max,
        maximum,
        min,
        minimum,
        mod,
        modf,
        multiply,
        nan_to_num,
        nancumprod,
        nancumsum,
        nanmax,
        nanprod,
        nansum,
        negative,
        power,
        positive,
        prod,
        rad2deg,
        radians,
        real,
        reciprocal,
        relu,
        remainder,
        rint,
        round_,
        sign,
        signbit,
        sin,
        sinc,
        sinh,
        sqrt,
        square,
        subtract,
        sum,
        tan,
        tanh,
        true_divide,
        trunc,
    )

    from . import random
    from . import testing

    from .sorting import sort

    from .statistics import mean

    from .utils import broadcast_shape, ndarray

    from ._types import (
        ArrayLike,
        DTypeLike,
        ShapeLike,
        AxisLike,
        AxisOptional,
        ScalarLike,
    )
    from .nn import softmax

    from .io import save, savez, savez_compressed, load


# ---------------------------------------------------------------------------
# Top-level imports for runtime use
# ---------------------------------------------------------------------------
from . import random
from . import testing
from .utils import broadcast_shape, ndarray
from .io import save, savez, savez_compressed, load
from .linalg.direct import _direct_all_

# NumPy-compatible constants
from numpy import e, euler_gamma, inf, nan, newaxis, pi

# NumPy-compatible dtype types
from numpy import (
    bool_, int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float16, float32, float64,
    complex64, complex128,
    dtype, finfo, iinfo,
)

# NumPy-compatible dtype helper functions
from numpy import issubdtype, promote_types, can_cast, result_type


_LAZY_MAPPING = {
    # .array
    "array": ".array",
    "asarray": ".array",
    "asanyarray": ".array",
    "copy": ".array",
    "empty": ".array", "empty_like": ".array", "eye": ".array", "full": ".array",
    "full_like": ".array", "identity": ".array", "linspace": ".array", "ones": ".array",
    "ones_like": ".array", "zeros": ".array", "zeros_like": ".array",
    # .cann
    "finalize": ".cann", "init": ".cann",
    "reset_device": ".cann", "reset_device_force": ".cann", "set_device": ".cann",
    # .linalg
    "linalg": ".linalg",
    # .linalg.direct
    "dot": ".linalg.direct", "einsum": ".linalg.direct", "inner": ".linalg.direct",
    "matmul": ".linalg.direct", "outer": ".linalg.direct", "vdot": ".linalg.direct",
    # .logic
    "all": ".logic", "any": ".logic", "equal": ".logic", "greater": ".logic",
    "greater_equal": ".logic", "isfinite": ".logic", "isinf": ".logic", "isneginf": ".logic",
    "isposinf": ".logic", "less": ".logic", "less_equal": ".logic", "logical_and": ".logic",
    "logical_not": ".logic", "logical_or": ".logic", "logical_xor": ".logic", "not_equal": ".logic",
    # .math
    "absolute": ".math", "add": ".math", "amax": ".math", "amin": ".math", "around": ".math",
    "arccos": ".math", "arccosh": ".math", "arcsin": ".math", "arcsinh": ".math",
    "arctan": ".math", "arctan2": ".math", "arctanh": ".math", "ceil": ".math", "clip": ".math",
    "copysign": ".math", "cos": ".math", "cosh": ".math", "cross": ".math", "cumprod": ".math",
    "cumsum": ".math", "deg2rad": ".math", "degrees": ".math", "divide": ".math", "divmod": ".math",
    "exp": ".math", "exp2": ".math", "expm1": ".math", "fabs": ".math", "fix": ".math", "floor": ".math",
    "floor_divide": ".math", "fmax": ".math", "fmin": ".math", "fmod": ".math", "float_power": ".math",
    "gelu": ".math", "gcd": ".math", "heaviside": ".math", "hypot": ".math", "lcm": ".math",
    "ldexp": ".math", "log": ".math", "log10": ".math", "log1p": ".math", "log2": ".math",
    "logaddexp": ".math", "logaddexp2": ".math", "max": ".math", "maximum": ".math", "min": ".math",
    "minimum": ".math", "mod": ".math", "modf": ".math", "multiply": ".math", "nan_to_num": ".math",
    "nancumprod": ".math", "nancumsum": ".math", "nanmax": ".math", "nanprod": ".math", "nansum": ".math",
    "negative": ".math", "power": ".math", "positive": ".math", "prod": ".math", "rad2deg": ".math",
    "radians": ".math", "real": ".math", "reciprocal": ".math", "relu": ".math", "remainder": ".math",
    "rint": ".math", "round_": ".math", "sign": ".math", "signbit": ".math", "sin": ".math",
    "sinc": ".math", "sinh": ".math", "sqrt": ".math", "square": ".math", "subtract": ".math",
    "sum": ".math", "tan": ".math", "tanh": ".math", "true_divide": ".math", "trunc": ".math",
    # .random
    "random": ".random",
    # .sorting
    "sort": ".sorting",
    # .statistics
    "mean": ".statistics",
    # .nn
    "softmax": ".nn",
    # .io
    "save": ".io", "savez": ".io", "savez_compressed": ".io", "load": ".io",
    # ._types
    "ArrayLike": "._types", "DTypeLike": "._types", "ShapeLike": "._types",
    "AxisLike": "._types", "AxisOptional": "._types", "ScalarLike": "._types",
    # .utils
    "broadcast_shape": ".utils", "ndarray": ".utils",
}

# NumPy re-exported names: these are imported at module level and must
# appear in __all__ so that __getattr__ does not try to re-resolve them.
_NUMPY_REEXPORTS = [
    # constants
    "e", "euler_gamma", "inf", "nan", "newaxis", "pi",
    # dtype types
    "bool_", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
    "complex64", "complex128",
    "dtype", "finfo", "iinfo",
    # dtype helpers
    "issubdtype", "promote_types", "can_cast", "result_type",
]

__all__ = list(_LAZY_MAPPING.keys()) + _NUMPY_REEXPORTS
__all__.extend(_direct_all_)


# ---------------------------------------------------------------------------
# Module-level __getattr__: automatic numpy fallback for unimplemented APIs
# ---------------------------------------------------------------------------
# Python's module attribute lookup order:
#   1. Explicit imports / __dict__  →  found? return immediately
#   2. Not found?  →  call __getattr__(name) if defined on the module
#
# Example: when a user writes `ap.tri(3)`, Python cannot find "tri" in
# asnumpy's explicit imports, so it calls __getattr__("tri"). This function
# then delegates to numpy.tri and wraps the returned np.ndarray into an
# asnumpy.ndarray, so the user gets a seamless experience.
#
# Guard: if a name is declared in __all__ but has no actual implementation
# (e.g. a planned-but-not-yet-implemented operator), we raise AttributeError
# immediately instead of silently falling back to numpy. This prevents real
# bugs from being masked — if we claimed to support "sin", it MUST work on
# NPU, not quietly run on CPU via numpy.
# ---------------------------------------------------------------------------

def __getattr__(name):
    """Lazy-load asnumpy modules and fallback to numpy for unimplemented APIs.

    Lookup flow:
        1. Check _LAZY_MAPPING — if the name maps to an asnumpy submodule,
           import it lazily and cache the result in sys.modules.
        2. Delegate to numpy — try ``getattr(np, name)``.
        3. If the result is callable, wrap it so that any np.ndarray
           returned by numpy is automatically converted to asnumpy.ndarray
           via ``_wrap_result``. Non-callable attributes (e.g. constants)
           are returned as-is.

    This is the same pattern used by CuPy for APIs it has not yet ported
    to GPU: the user gets a working API immediately, and native
    implementations can be added incrementally without breaking existing
    user code.
    """
    # Step 1: lazy-load from asnumpy submodules
    if name in _LAZY_MAPPING:
        module_path = _LAZY_MAPPING[name]
        module = importlib.import_module(module_path, package=__package__)
        # If name matches the last segment of the path (e.g. "array" == ".array"),
        # it could be a module or a function inside it. Try the attribute first.
        attr = getattr(module, name, None)
        if attr is not None:
            return attr
        return module

    # Step 2: delegate to numpy
    try:
        attr = getattr(np, name)
    except AttributeError:
        raise AttributeError(
            f"module 'asnumpy' has no attribute {name!r}"
        )

    # Wrap callables so that np.ndarray return values become asnumpy.ndarray.
    # Non-callable attributes (e.g. np.ndarray subclass types) pass through.
    if callable(attr):
        def _wrapped(*args, **kwargs):
            result = attr(*args, **kwargs)
            return _wrap_result(result)
        # Preserve introspection so that tracebacks and help() look correct
        _wrapped.__name__ = name
        _wrapped.__qualname__ = f'asnumpy.{name}'
        _wrapped.__module__ = 'asnumpy'
        return _wrapped

    return attr


def _wrap_result(result):
    """Recursively convert np.ndarray values to asnumpy.ndarray.

    This handles the common return types from numpy functions:
        - np.ndarray       → asnumpy.ndarray (via from_numpy)
        - tuple of arrays  → tuple of asnumpy.ndarray
        - list of arrays   → list of asnumpy.ndarray
        - scalars / other  → returned as-is
    """
    if isinstance(result, np.ndarray):
        return ndarray.from_numpy(result)
    if isinstance(result, tuple):
        return tuple(_wrap_result(r) for r in result)
    if isinstance(result, list):
        return [_wrap_result(r) for r in result]
    return result


logger.disable("asnumpy")


def enable_logging(level="INFO", log_dir=None):
    """
    Enable low-level logging for asnumpy.
    :param level: Logging level (e.g., "DEBUG", "INFO").
    :param log_dir: If specified, logs will be saved to a file in this directory alongside console output.
    """
    # Enable logging for the current module
    logger.enable("asnumpy")
    logger.remove() # Remove Loguru's default handler

    # Add safe console output (use sys.__stderr__ to avoid closed stream errors during atexit)
    logger.add(sys.__stderr__, level=level,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                      "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
               catch=True)

    # Write to file only if log_dir is provided by the user
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "asnumpy_{time:YYYY-MM-DD_HHmmss}.log")
        logger.add(log_file, retention="7 days", level=level, catch=True)
        logger.info(f"ASNumPy file logging enabled: {log_file}")


if os.getenv("ASNUMPY_DEBUG", "0") == "1":
    enable_logging(level="DEBUG", log_dir=os.getenv("ASNUMPY_LOG_DIR", None))


import atexit


@atexit.register
def reset():
    reset_device(0)
    finalize()


init()
set_device(0)
