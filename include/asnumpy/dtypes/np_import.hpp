// Prevent multiple inclusion
#pragma once

#ifdef PyArray_Type
#error "NumPy headers must not be included before np_import.hpp"
#endif

// Disallow NumPy 1.7 deprecated symbols
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

// Unique NumPy C-API symbol for this extension
#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL _asnumpy_numpy_api
#endif

// Only the one translation unit defining ASNUMPY_IMPORT_NUMPY should import the API
#ifndef ASNUMPY_IMPORT_NUMPY
#ifndef NO_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif
#endif

// Place <locale> before <Python.h> for macOS compatibility (harmless elsewhere)
#include <locale>
#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>

// NumPy 2.x compatibility shim for descriptor prototype type name
#ifndef NPY_ABI_VERSION
#error "NumPy headers did not define NPY_ABI_VERSION"
#endif

#if NPY_ABI_VERSION < 0x02000000
#define PyArray_DescrProto PyArray_Descr
#endif

namespace asnumpy{
    namespace dtypes{
        void ImportNumpy();
    }
}