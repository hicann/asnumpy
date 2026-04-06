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
"""Tests for the __getattr__ fallback mechanism."""

import numpy
import pytest

import asnumpy as ap


class TestFallbackCallable:
    """Test that unimplemented numpy functions fall back correctly."""

    def test_binary_repr(self):
        """Pure CPU function should fall back to numpy."""
        assert ap.binary_repr(10) == numpy.binary_repr(10)

    def test_binary_repr_width(self):
        assert ap.binary_repr(10, width=8) == numpy.binary_repr(10, width=8)

    def test_base_repr(self):
        assert ap.base_repr(5, base=2) == numpy.base_repr(5, base=2)

    def test_isscalar(self):
        assert ap.isscalar(3.14) == numpy.isscalar(3.14)
        assert ap.isscalar([1, 2]) == numpy.isscalar([1, 2])


class TestFallbackArrayWrapping:
    """Test that fallback functions returning arrays wrap them to ndarray."""

    def test_fromfunction_returns_ndarray(self):
        """fromfunction should return a wrapped ndarray."""
        result = ap.fromfunction(lambda i, j: i + j, (3, 3), dtype=numpy.float32)
        assert isinstance(result, ap.ndarray)
        numpy.testing.assert_allclose(
            result.to_numpy(),
            numpy.fromfunction(lambda i, j: i + j, (3, 3), dtype=numpy.float32),
        )

    def test_tri_returns_ndarray(self):
        """tri should return a wrapped ndarray."""
        result = ap.tri(3, dtype=numpy.float32)
        assert isinstance(result, ap.ndarray)
        numpy.testing.assert_allclose(
            result.to_numpy(),
            numpy.tri(3, dtype=numpy.float32),
        )


class TestFallbackAttributeError:
    """Test that accessing non-existent attributes raises AttributeError."""

    def test_nonexistent_attribute(self):
        """Accessing an attribute that doesn't exist in numpy should raise."""
        with pytest.raises(AttributeError):
            _ = ap.this_definitely_does_not_exist_xyz123

    def test_native_functions_not_fallback(self):
        """Native asnumpy functions should not be numpy's version."""
        assert ap.sin is not numpy.sin


class TestFallbackLogging:
    """Test that fallback calls emit a loguru warning."""

    def test_fallback_emits_warning(self):
        """Calling an unimplemented function should log a warning to loguru."""
        from loguru import logger

        # Enable the asnumpy logger (disabled by default)
        logger.enable("asnumpy")
        captured = []

        def capture_sink(message):
            captured.append(message.record["message"])

        handler_id = logger.add(capture_sink, level="WARNING", format="{message}")
        try:
            ap.binary_repr(10)
            assert any("not yet implemented on NPU" in msg for msg in captured)
            assert any("binary_repr" in msg for msg in captured)
        finally:
            logger.remove(handler_id)
            logger.disable("asnumpy")

    def test_native_function_no_warning(self):
        """Calling a native asnumpy function should NOT log a fallback warning."""
        from loguru import logger

        logger.enable("asnumpy")
        captured = []

        def capture_sink(message):
            captured.append(message.record["message"])

        handler_id = logger.add(capture_sink, level="WARNING", format="{message}")
        try:
            ap.zeros((2,), dtype=numpy.float32)
            assert not any("not yet implemented on NPU" in msg for msg in captured)
        finally:
            logger.remove(handler_id)
            logger.disable("asnumpy")
