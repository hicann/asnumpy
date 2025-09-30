#define ASNUMPY_IMPORT_NUMPY
#include <asnumpy/dtypes/np_import.hpp>

namespace asnumpy{
    namespace dtypes{
        void ImportNumpy(){
            if (!PyArray_API) { import_array1(); }
        }
    }
}