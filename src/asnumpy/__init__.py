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
import atexit
import functools
import importlib
import os
import sys
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

# NumPy constants — always CPU, re-exported for drop-in compatibility
from numpy import (
    can_cast,
    dtype,
    e,
    euler_gamma,
    finfo,
    iinfo,
    inf,
    issubdtype,
    nan,
    newaxis,
    pi,
    promote_types,
    result_type,
)

from .cann import finalize, init, reset_device, reset_device_force, set_device

if TYPE_CHECKING:
    from . import linalg, random
    from ._types import (
        ArrayLike,
        AxisLike,
        AxisOptional,
        DTypeLike,
        ScalarLike,
        ShapeLike,
    )
    from .array import (
        array,
        asanyarray,
        asarray,
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
    from .io import load, save, savez, savez_compressed
    from .linalg.direct import _direct_all_, dot, einsum, inner, matmul, outer, vdot
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
        arccos,
        arccosh,
        arcsin,
        arcsinh,
        arctan,
        arctan2,
        arctanh,
        around,
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
        float_power,
        floor,
        floor_divide,
        fmax,
        fmin,
        fmod,
        gcd,
        gelu,
        heaviside,
        hypot,
        lcm,
        ldexp,
        log,
        log1p,
        log2,
        log10,
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
        positive,
        power,
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
    from .nn import softmax
    from .sorting import sort
    from .statistics import mean
    from .utils import broadcast_shape, ndarray


# Common NumPy dtype aliases accessible as ap.float32, ap.int32, etc.
_NUMPY_DTYPE_NAMES = {
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "complex64",
    "complex128",
    "bool_",
}

_LAZY_MAPPING = {
    # .array
    "array": ".array",
    "asarray": ".array",
    "asanyarray": ".array",
    "copy": ".array",
    "empty": ".array",
    "empty_like": ".array",
    "eye": ".array",
    "full": ".array",
    "full_like": ".array",
    "identity": ".array",
    "linspace": ".array",
    "ones": ".array",
    "ones_like": ".array",
    "zeros": ".array",
    "zeros_like": ".array",
    # .linalg
    "linalg": ".linalg",
    # .linalg.direct
    "dot": ".linalg.direct",
    "einsum": ".linalg.direct",
    "inner": ".linalg.direct",
    "matmul": ".linalg.direct",
    "outer": ".linalg.direct",
    "vdot": ".linalg.direct",
    # .logic
    "all": ".logic",
    "any": ".logic",
    "equal": ".logic",
    "greater": ".logic",
    "greater_equal": ".logic",
    "isfinite": ".logic",
    "isinf": ".logic",
    "isneginf": ".logic",
    "isposinf": ".logic",
    "less": ".logic",
    "less_equal": ".logic",
    "logical_and": ".logic",
    "logical_not": ".logic",
    "logical_or": ".logic",
    "logical_xor": ".logic",
    "not_equal": ".logic",
    # .math
    "absolute": ".math",
    "add": ".math",
    "amax": ".math",
    "amin": ".math",
    "around": ".math",
    "arccos": ".math",
    "arccosh": ".math",
    "arcsin": ".math",
    "arcsinh": ".math",
    "arctan": ".math",
    "arctan2": ".math",
    "arctanh": ".math",
    "ceil": ".math",
    "clip": ".math",
    "copysign": ".math",
    "cos": ".math",
    "cosh": ".math",
    "cross": ".math",
    "cumprod": ".math",
    "cumsum": ".math",
    "deg2rad": ".math",
    "degrees": ".math",
    "divide": ".math",
    "divmod": ".math",
    "exp": ".math",
    "exp2": ".math",
    "expm1": ".math",
    "fabs": ".math",
    "fix": ".math",
    "floor": ".math",
    "floor_divide": ".math",
    "fmax": ".math",
    "fmin": ".math",
    "fmod": ".math",
    "float_power": ".math",
    "gelu": ".math",
    "gcd": ".math",
    "heaviside": ".math",
    "hypot": ".math",
    "lcm": ".math",
    "ldexp": ".math",
    "log": ".math",
    "log10": ".math",
    "log1p": ".math",
    "log2": ".math",
    "logaddexp": ".math",
    "logaddexp2": ".math",
    "max": ".math",
    "maximum": ".math",
    "min": ".math",
    "minimum": ".math",
    "mod": ".math",
    "modf": ".math",
    "multiply": ".math",
    "nan_to_num": ".math",
    "nancumprod": ".math",
    "nancumsum": ".math",
    "nanmax": ".math",
    "nanprod": ".math",
    "nansum": ".math",
    "negative": ".math",
    "power": ".math",
    "positive": ".math",
    "prod": ".math",
    "rad2deg": ".math",
    "radians": ".math",
    "real": ".math",
    "reciprocal": ".math",
    "relu": ".math",
    "remainder": ".math",
    "rint": ".math",
    "round_": ".math",
    "sign": ".math",
    "signbit": ".math",
    "sin": ".math",
    "sinc": ".math",
    "sinh": ".math",
    "sqrt": ".math",
    "square": ".math",
    "subtract": ".math",
    "sum": ".math",
    "tan": ".math",
    "tanh": ".math",
    "true_divide": ".math",
    "trunc": ".math",
    # .random
    "random": ".random",
    # .sorting
    "sort": ".sorting",
    # .statistics
    "mean": ".statistics",
    # .testing
    "testing": ".testing",
    # .nn
    "softmax": ".nn",
    # .io
    "save": ".io",
    "savez": ".io",
    "savez_compressed": ".io",
    "load": ".io",
    # ._types
    "ArrayLike": "._types",
    "DTypeLike": "._types",
    "ShapeLike": "._types",
    "AxisLike": "._types",
    "AxisOptional": "._types",
    "ScalarLike": "._types",
    # .utils
    "broadcast_shape": ".utils",
    "ndarray": ".utils",
}


_EAGER_EXPORTS = [
    "finalize",
    "init",
    "reset_device",
    "reset_device_force",
    "set_device",
]

_NUMPY_CONSTANTS = [
    "pi",
    "e",
    "inf",
    "nan",
    "newaxis",
    "euler_gamma",
]

_NUMPY_DTYPE_UTILS = [
    "dtype",
    "finfo",
    "iinfo",
    "issubdtype",
    "can_cast",
    "promote_types",
    "result_type",
]

__all__ = _EAGER_EXPORTS + _NUMPY_CONSTANTS + _NUMPY_DTYPE_UTILS + list(_LAZY_MAPPING.keys())


# Get version from package metadata
try:
    from importlib.metadata import version

    __version__ = version("asnumpy")
except Exception:
    __version__ = "0.2.0"


def _wrap_result(result):
    """Recursively wrap np.ndarray -> asnumpy.ndarray in fallback results.

    Handles direct ndarray returns and list/tuple containers.
    Non-array objects are returned as-is.
    """
    if isinstance(result, np.ndarray):
        from .utils import ndarray

        return ndarray.from_numpy(result)
    if isinstance(result, (list, tuple)):
        return type(result)(_wrap_result(item) for item in result)
    return result


def __getattr__(name):
    """Lazy-load asnumpy modules and fallback to numpy for unimplemented APIs.

    Lookup flow:
        1. Check _LAZY_MAPPING -- if the name maps to an asnumpy submodule,
           import it lazily and cache the result in sys.modules.
        2. Check for asnumpy subpackages -- try importing ``.{name}`` as
           an asnumpy subpackage (handles modules not yet in _LAZY_MAPPING).
        3. Check _NUMPY_DTYPE_NAMES -- dtype aliases like float32, int64
           are resolved to numpy dtypes.
        4. Delegate to numpy -- try ``getattr(np, name)``.
           - If the result is callable, wrap it so that any np.ndarray
             returned by numpy is automatically converted to asnumpy.ndarray
             via ``_wrap_result``, and emit a warning about the fallback.
           - Non-callable attributes are returned as-is.
        5. If numpy doesn't have it either, raise AttributeError.
    """
    # Step 1: lazy-load from asnumpy submodules
    if name in _LAZY_MAPPING:
        module_path = _LAZY_MAPPING[name]
        module = importlib.import_module(module_path, package=__package__)
        if name == module_path.strip("."):
            # The import above sets sys.modules[__name__].name = module
            # (e.g. asnumpy.array = <module>). When the module has a
            # function with the same name, replace it so future lookups
            # find the function instead of the module.
            if hasattr(module, name):
                result = getattr(module, name)
            else:
                result = module
        else:
            result = getattr(module, name)
        # Cache the resolved attribute to avoid re-running __getattr__
        sys.modules[__name__].__dict__[name] = result
        return result

    # Step 2: numpy dtype aliases (e.g., ap.float32 -> np.float32)
    if name in _NUMPY_DTYPE_NAMES:
        return getattr(np, name)

    # Step 4: general numpy fallback for unimplemented APIs
    try:
        numpy_attr = getattr(np, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    # Constants, dtype utilities, etc. are already handled by explicit
    # imports at the module level. This branch catches callable functions
    # not yet implemented on NPU.
    if callable(numpy_attr):
        logger.warning(
            f"'{name}' is not yet implemented on NPU, falling back to numpy. "
            f"This may cause performance degradation."
        )

        @functools.wraps(numpy_attr)
        def _wrapped(*args, **kwargs):
            return _wrap_result(numpy_attr(*args, **kwargs))

        return _wrapped

    return numpy_attr


def __dir__():
    return __all__ + ["__version__"] + list(_NUMPY_DTYPE_NAMES)


logger.disable("asnumpy")


def enable_logging(level="INFO", log_dir=None):
    """
    Enable low-level logging for asnumpy.
    :param level: Logging level (e.g., "DEBUG", "INFO").
    :param log_dir: If specified, logs will be saved to a file in this directory alongside console output.
    """
    # Enable logging for the current module
    logger.enable("asnumpy")
    logger.remove()  # Remove Loguru's default handler

    # Add safe console output (use sys.__stderr__ to avoid closed stream errors during atexit)
    logger.add(
        sys.__stderr__,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        catch=True,
    )

    # Write to file only if log_dir is provided by the user
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "asnumpy_{time:YYYY-MM-DD_HHmmss}.log")
        logger.add(log_file, retention="7 days", level=level, catch=True)
        logger.info(f"ASNumPy file logging enabled: {log_file}")


if os.getenv("ASNUMPY_DEBUG", "0") == "1":
    enable_logging(level="DEBUG", log_dir=os.getenv("ASNUMPY_LOG_DIR", None))


@atexit.register
def reset():
    reset_device(0)
    finalize()


init()
set_device(0)
